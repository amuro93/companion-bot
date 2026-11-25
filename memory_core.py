"""Isolated SQLite companion memory core (P1a).

Standard-library-only storage engine providing owner/character-scoped messages,
derived memories with explicit source links, read-only character sharing ACLs,
atomic correction lineages, hard-delete with content-free idempotency tombstones,
literal substring search, and content-free index job queueing.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterator, Sequence
import uuid

MAX_ID_LENGTH = 128
MAX_TEXT_LENGTH = 100_000
MAX_QUERY_LENGTH = 1_024
MAX_META_KEY_LENGTH = 64
MAX_LIMIT = 1_000

VALID_ROLES = frozenset({"user", "assistant"})
VALID_KINDS = frozenset({"preference", "event", "promise", "fact"})
VALID_DOMAINS = frozenset({"real", "roleplay"})
VALID_SOURCES = frozenset({"manual", "auto"})

_SCHEMA_SQL = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS messages (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL UNIQUE,
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    UNIQUE (owner_id, character_id, external_id),
    UNIQUE (message_id, owner_id, character_id)
);

CREATE TABLE IF NOT EXISTS memories (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL UNIQUE,
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    root_id TEXT NOT NULL,
    parent_id TEXT,
    superseded_by TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('preference', 'event', 'promise', 'fact')),
    domain TEXT NOT NULL CHECK (domain IN ('real', 'roleplay')),
    text TEXT NOT NULL,
    meta_key TEXT,
    source TEXT NOT NULL DEFAULT 'auto',
    status TEXT NOT NULL CHECK (status IN ('active', 'superseded')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (owner_id, character_id, external_id),
    UNIQUE (memory_id, owner_id, character_id)
);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (memory_id, message_id),
    FOREIGN KEY (memory_id, owner_id, character_id)
        REFERENCES memories(memory_id, owner_id, character_id)
        ON DELETE CASCADE,
    FOREIGN KEY (message_id, owner_id, character_id)
        REFERENCES messages(message_id, owner_id, character_id)
);

CREATE TABLE IF NOT EXISTS memory_acl (
    memory_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, character_id),
    FOREIGN KEY (memory_id)
        REFERENCES memories(memory_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_idempotency (
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    external_id TEXT NOT NULL,
    op_type TEXT NOT NULL CHECK (op_type IN ('save', 'correct')),
    memory_id TEXT NOT NULL,
    target_memory_id TEXT,
    root_id TEXT NOT NULL,
    payload_hash TEXT,
    is_deleted INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner_id, character_id, external_id)
);

CREATE TABLE IF NOT EXISTS memory_key_blocks (
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    meta_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner_id, character_id, meta_key)
);

CREATE TABLE IF NOT EXISTS index_jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    memory_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('upsert', 'delete')),
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_owner_status
    ON memories(owner_id, status, domain, created_at DESC, seq DESC);

CREATE INDEX IF NOT EXISTS idx_memories_root
    ON memories(root_id, owner_id, character_id);

CREATE INDEX IF NOT EXISTS idx_memory_acl_lookup
    ON memory_acl(memory_id, owner_id, character_id);

CREATE INDEX IF NOT EXISTS idx_memory_sources_mem
    ON memory_sources(memory_id, position ASC);

CREATE INDEX IF NOT EXISTS idx_index_jobs_scope
    ON index_jobs(owner_id, character_id, seq ASC);


COMMIT;
"""


def _utc_now_str() -> str:
    """Return current UTC timestamp as fixed-width ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalize_timestamp(value: str | datetime | None) -> tuple[str | None, str]:
    """Validate and normalize an optional timestamp to UTC ISO-8601.

    Returns (explicit_normalized_or_none, effective_utc_str).
    """
    if value is None:
        return None, _utc_now_str()

    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("occurred_at must not be empty")
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid occurred_at ISO timestamp: {value!r}") from exc
    else:
        raise ValueError("occurred_at must be an ISO-8601 string, datetime, or None")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    normalized = dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return normalized, normalized


def _validate_meta_key(value: Any) -> str | None:
    """Validate the optional P2o memory key ("<category>/<topic>")."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("meta_key must be a string or None")
    cleaned = value.strip()
    if not cleaned or "\x00" in cleaned or len(cleaned) > MAX_META_KEY_LENGTH:
        raise ValueError("invalid meta_key")
    return cleaned


def _validate_source(value: Any) -> str:
    """Validate the writer of a memory: manual (user typed) or auto (extractor)."""
    if not isinstance(value, str) or value not in VALID_SOURCES:
        raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}")
    return value


def _validate_id(value: Any, name: str) -> str:
    """Validate that an identifier is a non-empty bounded string without control chars."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value or not value.strip() or value != value.strip():
        raise ValueError(
            f"{name} must be a non-empty string without leading or trailing whitespace"
        )
    if len(value) > MAX_ID_LENGTH:
        raise ValueError(f"{name} exceeds maximum length of {MAX_ID_LENGTH}")
    if any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _validate_text(value: Any, name: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Validate that a text field is a non-empty bounded string without null bytes."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain null bytes")
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length of {max_length}")
    return value


def _validate_limit(limit: Any) -> int:
    """Validate that limit is a positive integer (rejecting booleans)."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer, not bool or other type")
    if limit <= 0 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    return limit


def _validate_source_ids(source_ids: Any) -> list[str]:
    """Validate that source_ids is a non-empty sequence of valid message IDs."""
    if isinstance(source_ids, (str, bytes, bytearray, dict)) or not isinstance(
        source_ids, (list, tuple, set, frozenset)
    ):
        raise ValueError("source_ids must be a non-empty list or tuple of IDs")
    if not source_ids:
        raise ValueError("source_ids must not be empty")

    validated: list[str] = []
    seen: set[str] = set()
    for idx, raw_id in enumerate(source_ids):
        sid = _validate_id(raw_id, f"source_ids[{idx}]")
        if sid not in seen:
            seen.add(sid)
            validated.append(sid)
    return validated


def _validate_visible_to(
    creator_character_id: str, visible_to: Any
) -> list[str]:
    """Validate visible_to and return canonical list of reader character IDs."""
    if visible_to is None:
        return [creator_character_id]
    if isinstance(visible_to, (str, bytes, bytearray, dict)) or not isinstance(
        visible_to, (list, tuple, set, frozenset)
    ):
        raise ValueError("visible_to must be a list/tuple of character IDs or None")

    readers: list[str] = [creator_character_id]
    seen: set[str] = {creator_character_id}
    for idx, raw_cid in enumerate(visible_to):
        cid = _validate_id(raw_cid, f"visible_to[{idx}]")
        if cid not in seen:
            seen.add(cid)
            readers.append(cid)
    return sorted(readers)


def _hash_payload(payload: dict[str, Any]) -> str:
    """Return SHA-256 hex digest of a canonical JSON payload without storing raw text."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _escape_like(query: str) -> str:
    """Escape SQLite LIKE wildcards (%, _, and backslash)."""
    return (
        query.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


class MemoryStore:
    """Isolated SQLite companion memory store.

    Caller context (`owner_id` and `character_id`) is treated as trusted input
    derived by an external authentication layer, never raw client JSON proof.
    """

    def __init__(self, db_path: str | os.PathLike[str]) -> None:
        if not isinstance(db_path, (str, os.PathLike)):
            raise ValueError("db_path must be a string or PathLike")
        raw_path = os.fspath(db_path)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("db_path must be a non-empty path string")

        self.db_path: str = raw_path
        self._lock = threading.RLock()
        self._is_memory_db = raw_path == ":memory:"
        self._memory_conn: sqlite3.Connection | None = None
        self._active_named_transaction: sqlite3.Connection | None = None
        self._active_named_thread: int | None = None

        if self._is_memory_db:
            self._memory_conn = self._create_connection()

        with self._connect() as conn:
            self._initialize_schema(conn)

    def _initialize_schema(
        self, conn: sqlite3.Connection, schema_sql: str | None = None
    ) -> None:
        script = _SCHEMA_SQL if schema_sql is None else schema_sql
        stripped = script.lstrip()
        if not stripped[:5].upper().startswith("BEGIN"):
            script = f"BEGIN IMMEDIATE;\n{script}\nCOMMIT;"
        try:
            conn.executescript(script)
            if conn.in_transaction:
                conn.execute("COMMIT")
            self._migrate_columns(conn)
            conn.execute("PRAGMA foreign_keys = ON")
        except Exception:
            self._safe_rollback(conn)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
            except sqlite3.Error:
                pass
            raise

    _ADDED_COLUMNS = (
        ("memories", "meta_key", "ALTER TABLE memories ADD COLUMN meta_key TEXT"),
        (
            "memories",
            "source",
            "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'auto'",
        ),
        (
            "messages",
            "excluded_from_memory",
            "ALTER TABLE messages ADD COLUMN excluded_from_memory INTEGER NOT NULL DEFAULT 0 CHECK(excluded_from_memory IN (0,1))",
        ),
    )

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        """Idempotently add P2o columns and the active-key index to older databases."""
        for table, column, ddl in self._ADDED_COLUMNS:
            existing = {
                str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                conn.execute(ddl)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_key_blocks (
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                meta_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (owner_id, character_id, meta_key)
            )
            """
        )
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_memories_active_meta_key'"
        ).fetchone()
        # A course topic describes a bucket, not a unique event. Keep the
        # topic for retrieval and use explicit record IDs for corrections.
        create_index = """
            CREATE UNIQUE INDEX idx_memories_active_meta_key
                ON memories(owner_id, character_id, meta_key)
                WHERE status = 'active' AND meta_key IS NOT NULL
                  AND instr(meta_key, '/') > 0 AND meta_key <> '約定與計畫/子女課程'
        """
        if row is None:
            conn.execute(create_index.replace('CREATE UNIQUE INDEX ', 'CREATE UNIQUE INDEX IF NOT EXISTS ', 1))
        else:
            normalize = lambda sql: "".join(sql.split()).lower().replace('ifnotexists', '')
            if normalize(str(row[0] or '')) != normalize(create_index):
                savepoint = "migrate_meta_key_idx"
                conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    conn.execute("DROP INDEX idx_memories_active_meta_key")
                    conn.execute(create_index)
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                except Exception:
                    try:
                        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    except Exception:
                        pass
                    raise

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._is_memory_db:
                if self._memory_conn is None:
                    self._memory_conn = self._create_connection()
                self._memory_conn.execute("PRAGMA foreign_keys = ON")
                yield self._memory_conn
            else:
                conn = self._create_connection()
                try:
                    yield conn
                finally:
                    conn.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Use one write-locked connection for key check and write."""
        with self._connect() as conn:
            if self._active_named_transaction is not None:
                raise RuntimeError("Nested named-key transaction is not allowed")
            self._begin_immediate(conn)
            self._active_named_transaction = conn
            self._active_named_thread = threading.get_ident()
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                self._safe_rollback(conn)
                raise
            finally:
                self._active_named_transaction = None
                self._active_named_thread = None

    def _require_named_transaction(self, conn: sqlite3.Connection) -> None:
        if (conn is not self._active_named_transaction
                or self._active_named_thread != threading.get_ident()
                or not conn.in_transaction):
            raise ValueError("Invalid internal transaction connection")

    @staticmethod
    def _begin_immediate(conn: sqlite3.Connection) -> None:
        for attempt in range(50):
            try:
                conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError:
                if attempt == 49:
                    raise
                time.sleep(0.01)

    @staticmethod
    def _safe_rollback(conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass

    def close(self) -> None:
        """Close persistent SQLite connection if held."""
        with self._lock:
            if self._memory_conn is not None:
                try:
                    self._memory_conn.close()
                finally:
                    self._memory_conn = None

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def append_message(
        self,
        owner_id: str,
        character_id: str,
        external_id: str,
        role: str,
        content: str,
        occurred_at: str | datetime | None = None,
    ) -> str:
        """Append a conversation message idempotently and return its UUID message_id."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        external_id = _validate_id(external_id, "external_id")
        if not isinstance(role, str) or role not in VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(VALID_ROLES)}")
        content = _validate_text(content, "content")
        explicit_occurred_at, effective_occurred_at = _normalize_timestamp(occurred_at)

        payload_hash = _hash_payload(
            {
                "role": role,
                "content": content,
                "occurred_at": explicit_occurred_at,
            }
        )

        with self._connect() as conn:
            self._begin_immediate(conn)
            try:
                existing = conn.execute(
                    """
                    SELECT message_id, payload_hash
                    FROM messages
                    WHERE owner_id = ? AND character_id = ? AND external_id = ?
                    """,
                    (owner_id, character_id, external_id),
                ).fetchone()

                if existing is not None:
                    if existing["payload_hash"] != payload_hash:
                        raise ValueError(
                            "Idempotency conflict: message payload changed for external_id"
                        )
                    conn.execute("COMMIT")
                    return str(existing["message_id"])

                message_id = str(uuid.uuid4())
                now = _utc_now_str()
                conn.execute(
                    """
                    INSERT INTO messages (
                        message_id, owner_id, character_id, external_id,
                        role, content, occurred_at, created_at, payload_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        owner_id,
                        character_id,
                        external_id,
                        role,
                        content,
                        effective_occurred_at,
                        now,
                        payload_hash,
                    ),
                )
                conn.execute("COMMIT")
                return message_id
            except Exception:
                self._safe_rollback(conn)
                raise

    def _verify_sources_in_scope(
        self,
        conn: sqlite3.Connection,
        owner_id: str,
        character_id: str,
        source_ids: Sequence[str],
    ) -> None:
        """Ensure all source_ids exist in messages for the same owner and character and are not excluded."""
        placeholders = ",".join("?" for _ in source_ids)
        rows = conn.execute(
            f"""
            SELECT message_id, excluded_from_memory
            FROM messages
            WHERE owner_id = ?
              AND character_id = ?
              AND message_id IN ({placeholders})
            """,
            (owner_id, character_id, *source_ids),
        ).fetchall()
        found = { str(row["message_id"]) for row in rows }
        if len(found) != len(source_ids):
            missing = [sid for sid in source_ids if sid not in found]
            raise ValueError(
                f"Source message IDs missing or out of owner/character scope: {missing}"
            )
        excluded_set = {
            str(row["message_id"])
            for row in rows
            if int(row["excluded_from_memory"] or 0) == 1
        }
        if excluded_set:
            excluded = [sid for sid in dict.fromkeys(source_ids) if sid in excluded_set]
            raise ValueError(
                f"Source message IDs excluded from memory: {excluded}"
            )

    def _queue_job(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        action: str,
        owner_id: str,
        character_id: str,
        created_at: str,
    ) -> None:
        """Insert a content-free index job inside the current transaction."""
        conn.execute(
            """
            INSERT INTO index_jobs (
                job_id, memory_id, action, owner_id, character_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), memory_id, action, owner_id, character_id, created_at),
        )

    def save_memory(
        self,
        owner_id: str,
        character_id: str,
        external_id: str,
        kind: str,
        text: str,
        source_ids: Sequence[str],
        domain: str = "real",
        visible_to: Sequence[str] | None = None,
        meta_key: str | None = None,
        source: str = "auto",
        _transaction_conn: sqlite3.Connection | None = None,
    ) -> str:
        """Persist a derived memory atomically with source links, ACL, and index job."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        external_id = _validate_id(external_id, "external_id")
        if not isinstance(kind, str) or kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
        if not isinstance(domain, str) or domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}")
        text = _validate_text(text, "text")
        norm_sources = _validate_source_ids(source_ids)
        norm_acl = _validate_visible_to(character_id, visible_to)
        meta_key = _validate_meta_key(meta_key)
        source = _validate_source(source)

        payload_hash = _hash_payload(
            {
                "op": "save",
                "kind": kind,
                "domain": domain,
                "text": text,
                "source_ids": norm_sources,
                "visible_to": norm_acl,
                "meta_key": meta_key,
                "source": source,
            }
        )
        if _transaction_conn is not None:
            self._require_named_transaction(_transaction_conn)

        with (self._connect() if _transaction_conn is None else nullcontext(_transaction_conn)) as conn:
            if _transaction_conn is None:
                self._begin_immediate(conn)
            try:
                idem = conn.execute(
                    """
                    SELECT op_type, memory_id, payload_hash, is_deleted
                    FROM memory_idempotency
                    WHERE owner_id = ? AND character_id = ? AND external_id = ?
                    """,
                    (owner_id, character_id, external_id),
                ).fetchone()

                if idem is not None:
                    if idem["is_deleted"] == 1:
                        raise ValueError(
                            "Cannot replay or resurrect a deleted memory external_id"
                        )
                    if idem["op_type"] != "save" or idem["payload_hash"] != payload_hash:
                        raise ValueError(
                            "Idempotency conflict: memory payload changed for external_id"
                        )
                    if _transaction_conn is None:
                        conn.execute("COMMIT")
                    return str(idem["memory_id"])

                self._verify_sources_in_scope(conn, owner_id, character_id, norm_sources)

                memory_id = str(uuid.uuid4())
                now = _utc_now_str()

                conn.execute(
                    """
                    INSERT INTO memories (
                        memory_id, owner_id, character_id, external_id,
                        root_id, parent_id, superseded_by,
                        kind, domain, text, meta_key, source,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        memory_id,
                        owner_id,
                        character_id,
                        external_id,
                        memory_id,
                        kind,
                        domain,
                        text,
                        meta_key,
                        source,
                        now,
                        now,
                    ),
                )

                for pos, sid in enumerate(norm_sources):
                    conn.execute(
                        """
                        INSERT INTO memory_sources (
                            memory_id, message_id, owner_id, character_id, position
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (memory_id, sid, owner_id, character_id, pos),
                    )

                for reader_cid in norm_acl:
                    conn.execute(
                        """
                        INSERT INTO memory_acl (memory_id, owner_id, character_id)
                        VALUES (?, ?, ?)
                        """,
                        (memory_id, owner_id, reader_cid),
                    )

                conn.execute(
                    """
                    INSERT INTO memory_idempotency (
                        owner_id, character_id, external_id, op_type,
                        memory_id, target_memory_id, root_id,
                        payload_hash, is_deleted, created_at
                    ) VALUES (?, ?, ?, 'save', ?, NULL, ?, ?, 0, ?)
                    """,
                    (
                        owner_id,
                        character_id,
                        external_id,
                        memory_id,
                        memory_id,
                        payload_hash,
                        now,
                    ),
                )

                self._queue_job(conn, memory_id, "upsert", owner_id, character_id, now)

                if _transaction_conn is None:
                    conn.execute("COMMIT")
                return memory_id
            except Exception:
                if _transaction_conn is None:
                    self._safe_rollback(conn)
                raise

    def block_key(self, owner_id: str, character_id: str, meta_key: str) -> None:
        """Stop automatic writes for this key until the user writes it manually (P2o)."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        meta_key = _validate_meta_key(meta_key)
        if meta_key is None:
            raise ValueError("meta_key is required")
        with self._connect() as conn:
            self._begin_immediate(conn)
            try:
                conn.execute(
                    """
                    INSERT INTO memory_key_blocks (owner_id, character_id, meta_key, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(owner_id, character_id, meta_key) DO NOTHING
                    """,
                    (owner_id, character_id, meta_key, _utc_now_str()),
                )
                conn.execute("COMMIT")
            except Exception:
                self._safe_rollback(conn)
                raise

    def unblock_key(self, owner_id: str, character_id: str, meta_key: str) -> bool:
        """Lift a block; returns True when a row was removed."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        meta_key = _validate_meta_key(meta_key)
        if meta_key is None:
            raise ValueError("meta_key is required")
        with self._connect() as conn:
            self._begin_immediate(conn)
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM memory_key_blocks
                    WHERE owner_id = ? AND character_id = ? AND meta_key = ?
                    """,
                    (owner_id, character_id, meta_key),
                )
                removed = cursor.rowcount > 0
                conn.execute("COMMIT")
                return removed
            except Exception:
                self._safe_rollback(conn)
                raise

    def is_key_blocked(self, owner_id: str, character_id: str, meta_key: str) -> bool:
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        meta_key = _validate_meta_key(meta_key)
        if meta_key is None:
            raise ValueError("meta_key is required")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM memory_key_blocks
                WHERE owner_id = ? AND character_id = ? AND meta_key = ?
                """,
                (owner_id, character_id, meta_key),
            ).fetchone()
            return row is not None

    def is_message_excluded(
        self, owner_id: str, character_id: str, message_id: str
    ) -> bool:
        """Return True if message exists in exact scope and is excluded from memory."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        message_id = _validate_id(message_id, "message_id")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT excluded_from_memory
                FROM messages
                WHERE owner_id = ? AND character_id = ? AND message_id = ?
                """,
                (owner_id, character_id, message_id),
            ).fetchone()
            if row is None:
                return False
            return int(row["excluded_from_memory"] or 0) == 1

    def find_by_external_id(
        self, owner_id: str, character_id: str, external_id: str
    ) -> dict[str, Any] | None:
        """Return the memory already written for this external_id, or None (P2o-4).

        Deleted external_ids return None: the tombstone must still trip the
        "cannot resurrect" guard inside save_memory.
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        external_id = _validate_id(external_id, "external_id")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT m.memory_id, m.owner_id, m.character_id,
                       m.kind, m.domain, m.text, m.meta_key, m.source,
                       m.status, m.created_at, m.updated_at
                FROM memory_idempotency i
                JOIN memories m
                  ON m.memory_id = i.memory_id
                 AND m.owner_id = i.owner_id
                 AND m.character_id = i.character_id
                WHERE i.owner_id = ?
                  AND i.character_id = ?
                  AND i.external_id = ?
                  AND i.is_deleted = 0
                """,
                (owner_id, character_id, external_id),
            ).fetchone()
            if row is None:
                return None
            return self._hydrate_memory_row(conn, row)

    def find_active_by_key(
        self, owner_id: str, character_id: str, meta_key: str
    ) -> dict[str, Any] | None:
        """Return active memory for key: named slot unique, category bucket may contain many, returns newest only for backward compatibility."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        meta_key = _validate_meta_key(meta_key)
        if meta_key is None:
            raise ValueError("meta_key is required")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    m.memory_id, m.owner_id, m.character_id,
                    m.kind, m.domain, m.text, m.meta_key, m.source,
                    m.created_at, m.updated_at
                FROM memories m
                WHERE m.owner_id = ?
                  AND m.character_id = ?
                  AND m.meta_key = ?
                  AND m.status = 'active'
                ORDER BY m.created_at DESC, m.seq DESC
                """,
                (owner_id, character_id, meta_key),
            ).fetchone()
            if row is None:
                return None
            return self._hydrate_memory_row(conn, row)

    def save_keyed_memory(
        self,
        owner_id: str,
        character_id: str,
        external_id: str,
        kind: str,
        text: str,
        source_ids: Sequence[str],
        meta_key: str,
        domain: str = "real",
        visible_to: Sequence[str] | None = None,
        source: str = "auto",
    ) -> dict[str, Any]:
        """Insert a keyed memory, or supersede the active one holding the same key.

        Returns {"action": "insert" | "correct" | "unchanged" | "skipped", ...}.
        A manual memory is never overwritten by an automatic one (P2o).
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        external_id = _validate_id(external_id, "external_id")
        meta_key = _validate_meta_key(meta_key)
        if meta_key is None:
            raise ValueError("meta_key is required")
        if not isinstance(kind, str) or kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {sorted(VALID_KINDS)}")
        if not isinstance(domain, str) or domain not in VALID_DOMAINS:
            raise ValueError(f"domain must be one of {sorted(VALID_DOMAINS)}")
        source = _validate_source(source)
        text = _validate_text(text, "text")
        norm_sources = _validate_source_ids(source_ids)
        norm_acl = _validate_visible_to(character_id, visible_to)

        is_course_bucket = meta_key == "約定與計畫/子女課程"
        if "/" not in meta_key or is_course_bucket:
            payload_hash = _hash_payload(
                {
                    "op": "save",
                    "kind": kind,
                    "domain": domain,
                    "text": text,
                    "source_ids": norm_sources,
                    "visible_to": norm_acl,
                    "meta_key": meta_key,
                    "source": source,
                }
            )
            with (self._write_transaction() if is_course_bucket else self._connect()) as conn:
                idem = conn.execute(
                    """
                    SELECT op_type, memory_id, target_memory_id, payload_hash, is_deleted
                    FROM memory_idempotency
                    WHERE owner_id = ? AND character_id = ? AND external_id = ?
                    """,
                    (owner_id, character_id, external_id),
                ).fetchone()

                if idem is not None:
                    if idem["is_deleted"] == 1:
                        raise ValueError(
                            "Cannot replay or resurrect a deleted memory external_id"
                        )
                    if idem["op_type"] == "save":
                        if idem["payload_hash"] != payload_hash:
                            raise ValueError(
                                "Idempotency conflict: memory payload changed for external_id"
                            )
                        return {
                            "action": "insert",
                            "memory_id": str(idem["memory_id"]),
                            "meta_key": meta_key,
                        }
                    elif idem["op_type"] == "correct":
                        legacy_payload = {
                            "op": "correct",
                            "target_memory_id": (
                                str(idem["target_memory_id"])
                                if idem["target_memory_id"] is not None
                                else None
                            ),
                            "text": text,
                            "source_ids": norm_sources,
                            "meta_key": meta_key,
                            "source": source,
                        }
                        if idem["payload_hash"] != _hash_payload(legacy_payload):
                            raise ValueError(
                                "Idempotency conflict: memory payload changed for external_id"
                            )
                        corr_mem = conn.execute(
                            """
                            SELECT memory_id, owner_id, character_id, kind, domain,
                                   text, meta_key, source, status
                            FROM memories
                            WHERE memory_id = ?
                            """,
                            (idem["memory_id"],),
                        ).fetchone()
                        if corr_mem is None:
                            raise ValueError(
                                "Cannot replay or resurrect a deleted memory external_id"
                            )
                        if (
                            corr_mem["owner_id"] != owner_id
                            or corr_mem["character_id"] != character_id
                            or corr_mem["kind"] != kind
                            or corr_mem["domain"] != domain
                            or corr_mem["meta_key"] != meta_key
                            or corr_mem["text"] != text
                            or corr_mem["source"] != source
                        ):
                            raise ValueError(
                                "Idempotency conflict: memory payload changed for external_id"
                            )
                        acl_rows = conn.execute(
                            """
                            SELECT character_id
                            FROM memory_acl
                            WHERE memory_id = ?
                            ORDER BY character_id ASC
                            """,
                            (idem["memory_id"],),
                        ).fetchall()
                        existing_acl = sorted({str(r["character_id"]) for r in acl_rows})
                        if existing_acl != norm_acl:
                            raise ValueError(
                                "Idempotency conflict: memory payload changed for external_id"
                            )
                        return {
                            "action": "unchanged",
                            "memory_id": str(idem["memory_id"]),
                            "meta_key": meta_key,
                        }
                    else:
                        raise ValueError(
                            "Idempotency conflict: memory payload changed for external_id"
                        )

                self._verify_sources_in_scope(
                    conn, owner_id, character_id, norm_sources
                )

                norm_text = " ".join(text.split())
                rows = conn.execute(
                    """
                    SELECT memory_id, text
                    FROM memories
                    WHERE owner_id = ?
                      AND character_id = ?
                      AND meta_key = ?
                      AND domain = ?
                      AND kind = ?
                      AND status = 'active'
                    ORDER BY seq ASC
                    """,
                    (owner_id, character_id, meta_key, domain, kind),
                ).fetchall()

                for r in rows:
                    if " ".join(str(r["text"]).split()) == norm_text:
                        return {
                            "action": "unchanged",
                            "memory_id": str(r["memory_id"]),
                            "meta_key": meta_key,
                        }

                if is_course_bucket:
                    memory_id = self.save_memory(
                        owner_id, character_id, external_id, kind, text, norm_sources,
                        domain=domain, visible_to=norm_acl, meta_key=meta_key,
                        source=source, _transaction_conn=conn,
                    )
                    return {"action": "insert", "memory_id": memory_id, "meta_key": meta_key}
            memory_id = self.save_memory(
                owner_id,
                character_id,
                external_id,
                kind,
                text,
                norm_sources,
                domain=domain,
                visible_to=norm_acl,
                meta_key=meta_key,
                source=source,
            )
            return {"action": "insert", "memory_id": memory_id, "meta_key": meta_key}

        # Named-key path: select and mutate under the same SQLite write lock.
        with self._write_transaction() as conn:
            tombstone = conn.execute(
                """SELECT is_deleted FROM memory_idempotency
                   WHERE owner_id = ? AND character_id = ? AND external_id = ?""",
                (owner_id, character_id, external_id),
            ).fetchone()
            if tombstone is not None and tombstone["is_deleted"] == 1:
                raise ValueError("Cannot replay or resurrect a deleted memory external_id")
            row = conn.execute(
                """SELECT memory_id, text, source FROM memories
                   WHERE owner_id = ? AND character_id = ? AND meta_key = ?
                     AND status = 'active'
                   ORDER BY created_at DESC, seq DESC LIMIT 1""",
                (owner_id, character_id, meta_key),
            ).fetchone()
            if row is None:
                memory_id = self.save_memory(
                    owner_id, character_id, external_id, kind, text, norm_sources,
                    domain=domain, visible_to=norm_acl, meta_key=meta_key,
                    source=source, _transaction_conn=conn,
                )
                return {"action": "insert", "memory_id": memory_id, "meta_key": meta_key}
            if row["source"] == "manual" and source == "auto":
                return {"action": "skipped", "reason": "manual takes precedence",
                        "memory_id": str(row["memory_id"]), "meta_key": meta_key}
            if " ".join(str(row["text"]).split()) == " ".join(text.split()):
                return {"action": "unchanged", "memory_id": str(row["memory_id"]),
                        "meta_key": meta_key}
            new_id = self.correct_memory(
                owner_id, character_id, str(row["memory_id"]), external_id,
                text, norm_sources, meta_key=meta_key, source=source,
                _transaction_conn=conn,
            )
            return {"action": "correct", "memory_id": new_id,
                    "superseded": str(row["memory_id"]), "meta_key": meta_key}

    def _hydrate_memory_row(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        memory_id = str(row["memory_id"])
        source_rows = conn.execute(
            """
            SELECT message_id
            FROM memory_sources
            WHERE memory_id = ?
            ORDER BY position ASC
            """,
            (memory_id,),
        ).fetchall()
        source_ids = [str(r["message_id"]) for r in source_rows]

        acl_rows = conn.execute(
            """
            SELECT character_id
            FROM memory_acl
            WHERE memory_id = ?
            ORDER BY character_id ASC
            """,
            (memory_id,),
        ).fetchall()
        visible_to = [str(r["character_id"]) for r in acl_rows]

        creator_char = str(row["character_id"])
        row_keys = set(row.keys())
        meta_key = str(row["meta_key"]) if "meta_key" in row_keys and row["meta_key"] is not None else None
        row_source = str(row["source"]) if "source" in row_keys and row["source"] is not None else "auto"
        return {
            "id": memory_id,
            "memory_id": memory_id,
            "owner_id": str(row["owner_id"]),
            "character_id": creator_char,
            "creator_character": creator_char,
            "creator_character_id": creator_char,
            "kind": str(row["kind"]),
            "domain": str(row["domain"]),
            "text": str(row["text"]),
            "meta_key": meta_key,
            "source": row_source,
            "source_ids": source_ids,
            "visible_to": visible_to,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def list_memories(
        self, owner_id: str, character_id: str, limit: int = 10, offset: int = 0
    ) -> dict[str, Any]:
        """Newest-first page of active memories visible to (owner_id, character_id)."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("invalid limit")
        if type(offset) is not int or not 0 <= offset <= 100000:
            raise ValueError("invalid offset")

        visible = """
            m.owner_id = ?
              AND m.status = 'active'
              AND (
                  m.character_id = ?
                  OR EXISTS (
                      SELECT 1 FROM memory_acl a
                      WHERE a.memory_id = m.memory_id
                        AND a.owner_id = m.owner_id
                        AND a.character_id = ?
                  )
              )
        """
        with self._connect() as conn:
            total = int(conn.execute(
                f"SELECT COUNT(*) FROM memories m WHERE {visible}",
                (owner_id, character_id, character_id),
            ).fetchone()[0])
            rows = conn.execute(
                f"""
                SELECT
                    m.memory_id, m.owner_id, m.character_id,
                    m.kind, m.domain, m.text, m.meta_key, m.source,
                    m.created_at, m.updated_at
                FROM memories m
                WHERE {visible}
                ORDER BY m.rowid DESC
                LIMIT ? OFFSET ?
                """,
                (owner_id, character_id, character_id, limit, offset),
            ).fetchall()
            items = [self._hydrate_memory_row(conn, row) for row in rows]
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def get_memory(
        self, owner_id: str, character_id: str, memory_id: str
    ) -> dict[str, Any] | None:
        """Retrieve an active memory if visible to (owner_id, character_id), else None."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        memory_id = _validate_id(memory_id, "memory_id")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    m.memory_id, m.owner_id, m.character_id,
                    m.kind, m.domain, m.text, m.meta_key, m.source,
                    m.created_at, m.updated_at
                FROM memories m
                WHERE m.memory_id = ?
                  AND m.owner_id = ?
                  AND m.status = 'active'
                  AND (
                      m.character_id = ?
                      OR EXISTS (
                          SELECT 1 FROM memory_acl a
                          WHERE a.memory_id = m.memory_id
                            AND a.owner_id = m.owner_id
                            AND a.character_id = ?
                      )
                  )
                """,
                (memory_id, owner_id, character_id, character_id),
            ).fetchone()

            if row is None:
                return None
            return self._hydrate_memory_row(conn, row)

    def search(
        self,
        owner_id: str,
        character_id: str,
        query: str,
        limit: int = 5,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """Literal substring search over active visible memories, ordered newest-first."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        query = _validate_text(query, "query", max_length=MAX_QUERY_LENGTH)
        limit = _validate_limit(limit)
        if domain is not None and (
            not isinstance(domain, str) or domain not in VALID_DOMAINS
        ):
            raise ValueError(f"domain must be None or one of {sorted(VALID_DOMAINS)}")

        like_pattern = f"%{_escape_like(query)}%"
        params: list[Any] = [owner_id, character_id, character_id, like_pattern]
        domain_clause = ""
        if domain is not None:
            domain_clause = "AND m.domain = ?"
            params.append(domain)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    m.memory_id, m.owner_id, m.character_id,
                    m.kind, m.domain, m.text, m.meta_key, m.source,
                    m.created_at, m.updated_at
                FROM memories m
                WHERE m.owner_id = ?
                  AND m.status = 'active'
                  AND (
                      m.character_id = ?
                      OR EXISTS (
                          SELECT 1 FROM memory_acl a
                          WHERE a.memory_id = m.memory_id
                            AND a.owner_id = m.owner_id
                            AND a.character_id = ?
                      )
                  )
                  AND m.text LIKE ? ESCAPE '\\'
                  {domain_clause}
                ORDER BY m.created_at DESC, m.seq DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

            return [self._hydrate_memory_row(conn, row) for row in rows]

    @staticmethod
    def is_course_reschedule(text: str) -> bool:
        import re
        return bool(re.search(r"課[^。！？!?\n]{0,12}(?:改到|改成|改為|改期|調到)", text))

    @staticmethod
    def _course_target(conn, owner_id, character_id, text):
        import re
        day = r"(?:每)?(?:週|星期)[一二三四五六日天]"
        clock = r"(?:(?:早上|上午|中午|下午|晚上)?(?:[0-9]{1,2}[:：][0-9]{2}|[一二三四五六七八九十兩0-9]{1,3}點(?:半|[一二三四五六七八九十0-9]{1,3}分)?))?"
        child = r"(?:我(?:的)?(?:兒子|女兒))?(?P<child>[\u4e00-\u9fff]{2,4}?)"
        course = r"(?P<course>[\u4e00-\u9fff]{1,12}課)"
        change = re.fullmatch(child + r"的" + course + r"(?:改到|改成|改為|調到)" + day + clock + r"[。！!]??", text.strip())
        if not change or change['child'] in {'他的', '她的', '他們', '她們', '孩子', '小孩', '兒子', '女兒', '他家', '她家'}:
            return None
        identity = change['child'], change['course']
        rows = conn.execute(
            "SELECT memory_id, text, source FROM memories WHERE owner_id=? AND character_id=? "
            "AND domain='real' AND status='active' AND meta_key=?",
            (owner_id, character_id, '約定與計畫/子女課程'),
        )
        scheduled = child + day + clock + r"上" + course + r"[。！!]??"
        matches = []
        for row in rows:
            previous = re.fullmatch(scheduled, row['text']) or re.fullmatch(change.re, row['text'])
            if previous and (previous['child'], previous['course']) == identity:
                matches.append(row)
        if len(matches) != 1 or matches[0]['source'] == 'manual':
            return None
        return matches[0]['memory_id']

    def needs_course_clarification(self, owner_id, character_id, text):
        if not self.is_course_reschedule(text):
            return False
        with self._connect() as conn:
            return self._course_target(conn, owner_id, character_id, text) is None

    def reschedule_course(self, owner_id, character_id, external_id, text, message_id):
        with self._write_transaction() as conn:
            self._verify_sources_in_scope(conn, owner_id, character_id, [message_id])
            if conn.execute(
                "SELECT 1 FROM memory_idempotency WHERE owner_id=? AND character_id=? AND external_id=?",
                (owner_id, character_id, external_id),
            ).fetchone():
                return None  # A replay must not retarget a newer active revision.
            if conn.execute(
                'SELECT 1 FROM memory_key_blocks WHERE owner_id=? AND character_id=? AND meta_key=?',
                (owner_id, character_id, '約定與計畫/子女課程'),
            ).fetchone():
                return None
            target = self._course_target(conn, owner_id, character_id, text)
            if target is None:
                return None
            return self.correct_memory(
                owner_id, character_id, target, external_id, text, [message_id],
                meta_key='約定與計畫/子女課程', source='auto', _transaction_conn=conn,
            )

    def correct_memory(
        self,
        owner_id: str,
        character_id: str,
        memory_id: str,
        external_id: str,
        text: str,
        source_ids: Sequence[str],
        meta_key: str | None = None,
        source: str | None = None,
        _transaction_conn: sqlite3.Connection | None = None,
    ) -> str:
        """Atomically supersede an active memory and insert its corrected replacement."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        memory_id = _validate_id(memory_id, "memory_id")
        external_id = _validate_id(external_id, "external_id")
        text = _validate_text(text, "text")
        norm_sources = _validate_source_ids(source_ids)
        new_meta_key = _validate_meta_key(meta_key)
        new_source = _validate_source(source) if source is not None else None

        payload_hash = _hash_payload(
            {
                "op": "correct",
                "target_memory_id": memory_id,
                "text": text,
                "source_ids": norm_sources,
                "meta_key": new_meta_key,
                "source": new_source,
            }
        )
        if _transaction_conn is not None:
            self._require_named_transaction(_transaction_conn)

        with (self._connect() if _transaction_conn is None else nullcontext(_transaction_conn)) as conn:
            if _transaction_conn is None:
                self._begin_immediate(conn)
            try:
                target = conn.execute(
                    """
                    SELECT memory_id, owner_id, character_id, root_id, kind, domain,
                           meta_key, source, status
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                ).fetchone()

                if target is not None:
                    if (
                        target["owner_id"] != owner_id
                        or target["character_id"] != character_id
                    ):
                        raise PermissionError(
                            "Only the creating owner and character may correct a memory"
                        )
                else:
                    tombstone = conn.execute(
                        """
                        SELECT owner_id, character_id, is_deleted
                        FROM memory_idempotency
                        WHERE memory_id = ?
                        """,
                        (memory_id,),
                    ).fetchone()
                    if tombstone is not None and (
                        tombstone["owner_id"] != owner_id
                        or tombstone["character_id"] != character_id
                    ):
                        raise PermissionError(
                            "Only the creating owner and character may correct a memory"
                        )

                idem = conn.execute(
                    """
                    SELECT op_type, memory_id, payload_hash, is_deleted
                    FROM memory_idempotency
                    WHERE owner_id = ? AND character_id = ? AND external_id = ?
                    """,
                    (owner_id, character_id, external_id),
                ).fetchone()

                if idem is not None:
                    if idem["is_deleted"] == 1:
                        raise ValueError(
                            "Cannot replay or resurrect a deleted memory external_id"
                        )
                    if (
                        idem["op_type"] != "correct"
                        or idem["payload_hash"] != payload_hash
                    ):
                        raise ValueError(
                            "Idempotency conflict: correction payload changed for external_id"
                        )
                    if _transaction_conn is None:
                        conn.execute("COMMIT")
                    return str(idem["memory_id"])

                if target is None:
                    raise ValueError("Target memory does not exist or has been deleted")

                if target["status"] != "active":
                    raise ValueError("Target memory is already superseded")

                self._verify_sources_in_scope(conn, owner_id, character_id, norm_sources)

                acl_rows = conn.execute(
                    """
                    SELECT character_id
                    FROM memory_acl
                    WHERE memory_id = ?
                    ORDER BY character_id ASC
                    """,
                    (memory_id,),
                ).fetchall()
                acl_characters = [str(r["character_id"]) for r in acl_rows]
                if character_id not in acl_characters:
                    acl_characters.append(character_id)

                new_memory_id = str(uuid.uuid4())
                now = _utc_now_str()
                root_id = str(target["root_id"])
                kind = str(target["kind"])
                domain = str(target["domain"])
                # P2o: a correction keeps the key and the origin (manual stays manual).
                target_meta_key = (
                    str(target["meta_key"]) if target["meta_key"] is not None else None
                )
                target_source = str(target["source"])
                # P2o: the caller may promote the replacement (auto -> manual).
                if new_meta_key is not None:
                    target_meta_key = new_meta_key
                if new_source is not None:
                    target_source = new_source

                conn.execute(
                    """
                    UPDATE memories
                    SET status = 'superseded',
                        superseded_by = ?,
                        updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (new_memory_id, now, memory_id),
                )

                conn.execute(
                    """
                    INSERT INTO memories (
                        memory_id, owner_id, character_id, external_id,
                        root_id, parent_id, superseded_by,
                        kind, domain, text, meta_key, source,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        new_memory_id,
                        owner_id,
                        character_id,
                        external_id,
                        root_id,
                        memory_id,
                        kind,
                        domain,
                        text,
                        target_meta_key,
                        target_source,
                        now,
                        now,
                    ),
                )

                for pos, sid in enumerate(norm_sources):
                    conn.execute(
                        """
                        INSERT INTO memory_sources (
                            memory_id, message_id, owner_id, character_id, position
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (new_memory_id, sid, owner_id, character_id, pos),
                    )

                for reader_cid in acl_characters:
                    conn.execute(
                        """
                        INSERT INTO memory_acl (memory_id, owner_id, character_id)
                        VALUES (?, ?, ?)
                        """,
                        (new_memory_id, owner_id, reader_cid),
                    )

                conn.execute(
                    """
                    INSERT INTO memory_idempotency (
                        owner_id, character_id, external_id, op_type,
                        memory_id, target_memory_id, root_id,
                        payload_hash, is_deleted, created_at
                    ) VALUES (?, ?, ?, 'correct', ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        owner_id,
                        character_id,
                        external_id,
                        new_memory_id,
                        memory_id,
                        root_id,
                        payload_hash,
                        now,
                    ),
                )

                self._queue_job(conn, memory_id, "delete", owner_id, character_id, now)
                self._queue_job(
                    conn, new_memory_id, "upsert", owner_id, character_id, now
                )

                if _transaction_conn is None:
                    conn.execute("COMMIT")
                return new_memory_id
            except Exception:
                if _transaction_conn is None:
                    self._safe_rollback(conn)
                raise

    def delete_memory(
        self, owner_id: str, character_id: str, memory_id: str
    ) -> bool:
        """Hard-delete a memory correction lineage while leaving content-free tombstones.

        Raises PermissionError if the target memory belongs to a different owner
        or was created by a different character (e.g. a read-only shared character).
        Returns True if the lineage was deleted, or False if not found / already deleted.
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        memory_id = _validate_id(memory_id, "memory_id")

        with self._connect() as conn:
            self._begin_immediate(conn)
            try:
                target = conn.execute(
                    """
                    SELECT memory_id, owner_id, character_id, root_id
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                ).fetchone()

                if target is None:
                    tombstone = conn.execute(
                        """
                        SELECT owner_id, character_id
                        FROM memory_idempotency
                        WHERE memory_id = ?
                        """,
                        (memory_id,),
                    ).fetchone()
                    if tombstone is not None and (
                        tombstone["owner_id"] != owner_id
                        or tombstone["character_id"] != character_id
                    ):
                        raise PermissionError(
                            "Only the creating owner and character may delete a memory"
                        )
                    conn.execute("COMMIT")
                    return False

                if (
                    target["owner_id"] != owner_id
                    or target["character_id"] != character_id
                ):
                    raise PermissionError(
                        "Only the creating owner and character may delete a memory"
                    )

                root_id = str(target["root_id"])
                lineage_rows = conn.execute(
                    """
                    SELECT memory_id
                    FROM memories
                    WHERE root_id = ? AND owner_id = ? AND character_id = ?
                    ORDER BY seq ASC
                    """,
                    (root_id, owner_id, character_id),
                ).fetchall()
                lineage_ids = [str(r["memory_id"]) for r in lineage_rows]
                placeholders = ",".join("?" for _ in lineage_ids)

                conn.execute(
                    f"""
                    UPDATE messages
                    SET excluded_from_memory = 1
                    WHERE owner_id = ?
                      AND character_id = ?
                      AND message_id IN (
                          SELECT message_id
                          FROM memory_sources
                          WHERE owner_id = ?
                            AND character_id = ?
                            AND memory_id IN ({placeholders})
                      )
                    """,
                    (owner_id, character_id, owner_id, character_id, *lineage_ids),
                )

                conn.execute(
                    f"DELETE FROM memory_sources WHERE memory_id IN ({placeholders})",
                    tuple(lineage_ids),
                )
                conn.execute(
                    f"DELETE FROM memory_acl WHERE memory_id IN ({placeholders})",
                    tuple(lineage_ids),
                )
                conn.execute(
                    """
                    DELETE FROM memories
                    WHERE root_id = ? AND owner_id = ? AND character_id = ?
                    """,
                    (root_id, owner_id, character_id),
                )

                conn.execute(
                    """
                    UPDATE memory_idempotency
                    SET is_deleted = 1,
                        payload_hash = NULL
                    WHERE root_id = ? AND owner_id = ? AND character_id = ?
                    """,
                    (root_id, owner_id, character_id),
                )

                conn.execute(
                    f"""
                    DELETE FROM index_jobs
                    WHERE owner_id = ?
                      AND character_id = ?
                      AND action = 'upsert'
                      AND memory_id IN ({placeholders})
                    """,
                    (owner_id, character_id, *lineage_ids),
                )

                existing_delete_rows = conn.execute(
                    f"""
                    SELECT memory_id
                    FROM index_jobs
                    WHERE owner_id = ?
                      AND character_id = ?
                      AND action = 'delete'
                      AND memory_id IN ({placeholders})
                    """,
                    (owner_id, character_id, *lineage_ids),
                ).fetchall()
                existing_deletes = {str(r["memory_id"]) for r in existing_delete_rows}

                now = _utc_now_str()
                for lid in lineage_ids:
                    if lid not in existing_deletes:
                        self._queue_job(
                            conn, lid, "delete", owner_id, character_id, now
                        )

                conn.execute("COMMIT")
                return True
            except Exception:
                self._safe_rollback(conn)
                raise

    def pending_jobs(
        self, owner_id: str, character_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Return pending content-free index jobs for the exact creating (owner_id, character_id)."""
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        limit = _validate_limit(limit)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id, memory_id, action, owner_id, character_id, created_at
                FROM index_jobs
                WHERE owner_id = ? AND character_id = ?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (owner_id, character_id, limit),
            ).fetchall()

            return [
                {
                    "id": str(row["job_id"]),
                    "job_id": str(row["job_id"]),
                    "memory_id": str(row["memory_id"]),
                    "action": str(row["action"]),
                    "owner_id": str(row["owner_id"]),
                    "character_id": str(row["character_id"]),
                    "created_at": str(row["created_at"]),
                }
                for row in rows
            ]

    def get_index_record(
        self, owner_id: str, character_id: str, memory_id: str
    ) -> dict[str, Any] | None:
        """Return active creator-owned canonical record for indexing, or None.

        Trusted-internal helper for index synchronization (NOT a public user-read API).
        Returns None if the memory does not exist, is not active, or was not created
        by the exact (owner_id, character_id).
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        memory_id = _validate_id(memory_id, "memory_id")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    memory_id, owner_id, character_id,
                    kind, domain, text, meta_key, source, created_at, updated_at
                FROM memories
                WHERE memory_id = ?
                  AND owner_id = ?
                  AND character_id = ?
                  AND status = 'active'
                """,
                (memory_id, owner_id, character_id),
            ).fetchone()

            if row is None:
                return None

            acl_rows = conn.execute(
                """
                SELECT character_id
                FROM memory_acl
                WHERE memory_id = ? AND owner_id = ?
                ORDER BY character_id ASC
                """,
                (memory_id, owner_id),
            ).fetchall()
            creator_char = str(row["character_id"])
            visible_set = {creator_char, *(str(r["character_id"]) for r in acl_rows)}
            visible_to = sorted(visible_set)

            return {
                "id": str(row["memory_id"]),
                "memory_id": str(row["memory_id"]),
                "owner_id": str(row["owner_id"]),
                "character_id": creator_char,
                "creator_character": creator_char,
                "creator_character_id": creator_char,
                "kind": str(row["kind"]),
                "domain": str(row["domain"]),
                "text": str(row["text"]),
                "visible_to": visible_to,
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }

    def ack_index_job(
        self, owner_id: str, character_id: str, job_id: str
    ) -> bool:
        """Delete an exact scoped index job upon successful processing.

        Returns True if the scoped job existed and was deleted, or False otherwise.
        Any failure rolls back and leaves index_jobs unchanged.
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        job_id = _validate_id(job_id, "job_id")

        with self._connect() as conn:
            self._begin_immediate(conn)
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM index_jobs
                    WHERE job_id = ? AND owner_id = ? AND character_id = ?
                    """,
                    (job_id, owner_id, character_id),
                )
                deleted = cursor.rowcount > 0
                conn.execute("COMMIT")
                return deleted
            except Exception:
                self._safe_rollback(conn)
                raise

    def rollback_memory(
        self, owner_id: str, character_id: str, memory_id: str
    ) -> dict[str, Any]:
        """Atomically roll back an active memory by restoring its parent revision as a NEW revision.

        Does not mutate historical revisions or toggle active/superseded flags backward.
        The current active memory becomes superseded, and a new active revision is created
        containing the parent revision's text, source message IDs, and source origin,
        while preserving the target's kind, domain, meta_key, and current reader ACL.
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        memory_id = _validate_id(memory_id, "memory_id")

        with self._lock:
            with self._connect() as conn:
                target = conn.execute(
                    """
                    SELECT memory_id, owner_id, character_id, root_id, parent_id,
                           kind, domain, text, meta_key, source, status
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (memory_id,),
                ).fetchone()

                if target is not None:
                    if (
                        target["owner_id"] != owner_id
                        or target["character_id"] != character_id
                    ):
                        raise PermissionError(
                            "Only the creating owner and character may roll back a memory"
                        )
                    if target["status"] != "active":
                        raise ValueError(
                            f"Target memory is already superseded or inactive: {memory_id}"
                        )
                else:
                    tombstone = conn.execute(
                        """
                        SELECT owner_id, character_id, is_deleted
                        FROM memory_idempotency
                        WHERE memory_id = ?
                        """,
                        (memory_id,),
                    ).fetchone()
                    if tombstone is not None and (
                        tombstone["owner_id"] != owner_id
                        or tombstone["character_id"] != character_id
                    ):
                        raise PermissionError(
                            "Only the creating owner and character may roll back a memory"
                        )
                    raise ValueError(
                        f"Target memory does not exist or has been deleted: {memory_id}"
                    )

                parent_id = target["parent_id"]
                if parent_id is None:
                    raise ValueError(
                        f"Target memory has no parent revision to roll back to: {memory_id}"
                    )
                if parent_id == memory_id:
                    raise ValueError(
                        f"Target memory cannot be its own parent: {memory_id}"
                    )

                parent = conn.execute(
                    """
                    SELECT memory_id, owner_id, character_id, root_id, parent_id,
                           kind, domain, text, meta_key, source, status
                    FROM memories
                    WHERE memory_id = ?
                    """,
                    (parent_id,),
                ).fetchone()

                if parent is None:
                    raise ValueError(
                        f"Parent revision does not exist or has been deleted: {parent_id}"
                    )

                if (
                    parent["owner_id"] != owner_id
                    or parent["character_id"] != character_id
                    or parent["root_id"] != target["root_id"]
                    or parent["memory_id"] == target["memory_id"]
                ):
                    raise ValueError(
                        f"Inconsistent version chain between target and parent: {memory_id} -> {parent_id}"
                    )

                source_rows = conn.execute(
                    """
                    SELECT message_id
                    FROM memory_sources
                    WHERE memory_id = ?
                    ORDER BY position ASC
                    """,
                    (parent_id,),
                ).fetchall()
                parent_source_ids = [str(r["message_id"]) for r in source_rows]
                if not parent_source_ids:
                    raise ValueError(
                        f"Parent memory has no source messages: {parent_id}"
                    )

                parent_text = str(parent["text"])
                parent_source = (
                    str(parent["source"]) if parent["source"] is not None else "auto"
                )
                target_meta_key = (
                    str(target["meta_key"]) if target["meta_key"] is not None else None
                )

            rollback_external_id = f"rollback-{uuid.uuid4()}"
            new_id = self.correct_memory(
                owner_id=owner_id,
                character_id=character_id,
                memory_id=memory_id,
                external_id=rollback_external_id,
                text=parent_text,
                source_ids=parent_source_ids,
                meta_key=target_meta_key,
                source=parent_source,
            )

            return {
                "action": "rollback",
                "memory_id": new_id,
                "restored_from": str(parent_id),
                "superseded": memory_id,
            }

