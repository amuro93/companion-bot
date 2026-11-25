"""SQLite-only, model-bound vector backend and canonical-authorized semantic search.

Provides:
- `SQLiteVectorBackend`: implements the `memory_index.VectorBackend` protocol
  (`ensure_collection`, `upsert`, `delete`) on top of the *same* SQLite database
  that a supplied `MemoryStore` owns (via `MemoryStore._connect`). Vectors are
  stored as packed little-endian float32 BLOBs in `memory_vectors`; a single
  `memory_vector_config` row binds the database to exactly one
  `(model_id, dimension)` pair. Canonical memory text is never duplicated.
- `SQLiteSemanticSearcher`: fail-closed semantic search. The payload handed to
  `upsert` is only a consistency claim and is never an authority: owner,
  creator, ACL, status and domain are always re-read from the canonical
  `memories` / `memory_acl` tables, both while selecting candidates (before any
  ranking) and again through `MemoryStore.get_memory` before returning results.

Standard library only. No HTTP, network, async work, model loading, FTS5 or
bootstrap logic lives here; the embedder is injected by the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import operator
import sqlite3
import struct
from typing import Any, Iterator

from memory_core import (
    MAX_LIMIT,
    MAX_QUERY_LENGTH,
    VALID_DOMAINS,
    MemoryStore,
    _utc_now_str,
    _validate_id,
    _validate_limit,
    _validate_source_ids,
    _validate_text,
)
from memory_index import (
    _validate_dimension,
    _validate_index_payload,
    _validate_uuid,
    _validate_vector,
)
from memory_search import SemanticSearchError

CONFIG_TABLE = "memory_vector_config"
VECTOR_TABLE = "memory_vectors"

_CONFIG_COLUMNS = frozenset({"id", "model_id", "dimension", "created_at"})
_VECTOR_COLUMNS = frozenset(
    {"memory_id", "model_id", "dimension", "vector", "updated_at"}
)

_FLOAT_BYTES = 4
# Tolerance for float rounding when clamping a cosine into [-1, 1].
_COSINE_TOLERANCE = 1e-6


class SQLiteVectorError(ValueError):
    """Sanitized validation/authority failure (never contains user text or payload)."""


class SQLiteVectorConfigMismatchError(SQLiteVectorError):
    """The requested or persisted model/dimension binding does not match."""


class SQLiteVectorAuthorityError(SQLiteVectorError):
    """The claimed metadata is not backed by the current canonical active record."""


class SQLiteVectorBackendError(RuntimeError):
    """Sanitized runtime failure: SQLite error, missing/corrupt table, corrupt BLOB."""


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _prepare_vector(
    vector: Any, dimension: int
) -> tuple[bytes, tuple[float, ...], float]:
    """Validate a vector and return `(blob, float32_values, l2_norm)`.

    Rejects non-sequences, bools, NaN/Inf, wrong length, float32 overflow,
    float32 underflow of a non-zero element, and (float32-rounded) zero vectors.
    The returned values are the float32-rounded numbers actually stored.
    """
    try:
        cleaned = _validate_vector(vector, dimension)
    except Exception:
        raise SQLiteVectorError("vector is invalid") from None
    fmt = f"<{dimension}f"
    try:
        blob = struct.pack(fmt, *cleaned)
    except (OverflowError, struct.error):
        raise SQLiteVectorError("vector value is outside float32 range") from None
    stored = struct.unpack(fmt, blob)
    for original, rounded in zip(cleaned, stored):
        if not math.isfinite(rounded):
            raise SQLiteVectorError("vector value is outside float32 range")
        if rounded == 0.0 and original != 0.0:
            raise SQLiteVectorError("vector value underflows float32")
    norm = math.sqrt(math.fsum(value * value for value in stored))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SQLiteVectorError("vector must be finite and non-zero")
    return blob, stored, norm


def _decode_blob(blob: Any, dimension: int) -> tuple[tuple[float, ...], float]:
    """Decode a stored BLOB; raise `SQLiteVectorBackendError` on any corruption."""
    if type(blob) is not bytes or len(blob) != dimension * _FLOAT_BYTES:
        raise SQLiteVectorBackendError("stored vector is corrupt")
    values = struct.unpack(f"<{dimension}f", blob)
    for value in values:
        if not math.isfinite(value):
            raise SQLiteVectorBackendError("stored vector is corrupt")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SQLiteVectorBackendError("stored vector is corrupt")
    return values, norm


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SQLiteVectorBackend:
    """Model-bound vector storage inside the supplied `MemoryStore` database."""

    def __init__(self, store: MemoryStore, dimension: int, model_id: str) -> None:
        if store is None or not (
            callable(getattr(store, "_connect", None))
            and callable(getattr(store, "get_index_record", None))
        ):
            raise SQLiteVectorError("store must provide _connect and get_index_record")
        try:
            self._dimension = _validate_dimension(dimension, "dimension")
            self._model_id = _validate_id(model_id, "model_id")
        except Exception:
            raise SQLiteVectorConfigMismatchError(
                "invalid model_id or dimension"
            ) from None
        self._store = store
        self._initialize()

    # -- public attributes -------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    # -- transaction helpers ----------------------------------------------

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass

    @contextmanager
    def _write_txn(self) -> Iterator[sqlite3.Connection]:
        """`BEGIN IMMEDIATE` ... `COMMIT` on the store connection; rollback on error."""
        with self._store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                self._rollback(conn)
                raise

    # -- schema / config ---------------------------------------------------

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
        return frozenset(
            str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
        )

    def _check_schema(self, conn: sqlite3.Connection) -> None:
        if (
            self._columns(conn, CONFIG_TABLE) != _CONFIG_COLUMNS
            or self._columns(conn, VECTOR_TABLE) != _VECTOR_COLUMNS
        ):
            raise SQLiteVectorBackendError("vector index tables missing or invalid")

    def _read_config(self, conn: sqlite3.Connection) -> tuple[str, int] | None:
        rows = conn.execute(
            f"SELECT id, model_id, dimension FROM {CONFIG_TABLE}"
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise SQLiteVectorBackendError("vector index config is corrupt")
        stored_id, stored_model, stored_dim = rows[0][0], rows[0][1], rows[0][2]
        if (
            type(stored_id) is not int
            or stored_id != 1
            or not isinstance(stored_model, str)
            or type(stored_dim) is not int
            or stored_dim <= 0
        ):
            raise SQLiteVectorBackendError("vector index config is corrupt")
        return stored_model, stored_dim

    def _verify_ready(self, conn: sqlite3.Connection) -> None:
        """Verify tables exist and the persisted binding equals this backend's."""
        self._check_schema(conn)
        config = self._read_config(conn)
        if config is None:
            raise SQLiteVectorBackendError("vector index config is missing")
        if config != (self._model_id, self._dimension):
            raise SQLiteVectorConfigMismatchError(
                "vector index is bound to a different model or dimension"
            )

    def _initialize(self) -> None:
        """Transactional, idempotent table creation; never drops or rebuilds data."""
        try:
            with self._write_txn() as conn:
                existing = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND name IN (?, ?)",
                        (CONFIG_TABLE, VECTOR_TABLE),
                    )
                }
                if not existing:
                    conn.execute(
                        f"""
                        CREATE TABLE {CONFIG_TABLE} (
                            id INTEGER PRIMARY KEY CHECK (id = 1),
                            model_id TEXT NOT NULL,
                            dimension INTEGER NOT NULL CHECK (dimension > 0),
                            created_at TEXT NOT NULL
                        )
                        """
                    )
                    conn.execute(
                        f"""
                        CREATE TABLE {VECTOR_TABLE} (
                            memory_id TEXT PRIMARY KEY,
                            model_id TEXT NOT NULL,
                            dimension INTEGER NOT NULL CHECK (dimension > 0),
                            vector BLOB NOT NULL,
                            updated_at TEXT NOT NULL
                        )
                        """
                    )
                elif existing != {CONFIG_TABLE, VECTOR_TABLE}:
                    raise SQLiteVectorBackendError(
                        "vector index tables are partially present"
                    )
                self._check_schema(conn)
                config = self._read_config(conn)
                if config is None:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {VECTOR_TABLE}"
                    ).fetchone()[0]
                    if count:
                        raise SQLiteVectorBackendError(
                            "vector index config is missing"
                        )
                    conn.execute(
                        f"INSERT INTO {CONFIG_TABLE} "
                        "(id, model_id, dimension, created_at) VALUES (1, ?, ?, ?)",
                        (self._model_id, self._dimension, _utc_now_str()),
                    )
                elif config != (self._model_id, self._dimension):
                    raise SQLiteVectorConfigMismatchError(
                        "vector index is bound to a different model or dimension"
                    )
        except (SQLiteVectorError, SQLiteVectorBackendError):
            raise
        except sqlite3.Error:
            raise SQLiteVectorBackendError("vector index initialization failed") from None

    # -- VectorBackend protocol -------------------------------------------

    def ensure_collection(self, dimension: int, model_id: str) -> None:
        """Verify caller config, this backend's config, and the persisted binding."""
        try:
            req_dim = _validate_dimension(dimension, "dimension")
            req_model = _validate_id(model_id, "model_id")
        except Exception:
            raise SQLiteVectorConfigMismatchError(
                "invalid model_id or dimension"
            ) from None
        if req_dim != self._dimension or req_model != self._model_id:
            raise SQLiteVectorConfigMismatchError(
                "requested model or dimension does not match the vector backend"
            )
        try:
            with self._store._connect() as conn:
                self._verify_ready(conn)
        except (SQLiteVectorError, SQLiteVectorBackendError):
            raise
        except sqlite3.Error:
            raise SQLiteVectorBackendError("vector index verification failed") from None

    def upsert(
        self, memory_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None:
        """Validate, check canonical authority, then insert-or-update the vector."""
        try:
            valid_id = _validate_uuid(memory_id, "memory_id")
        except Exception:
            raise SQLiteVectorError("memory_id is invalid") from None
        try:
            meta = _validate_index_payload(
                payload,
                expected_memory_id=valid_id,
                expected_model_id=self._model_id,
            )
        except Exception:
            raise SQLiteVectorError("payload is invalid") from None
        blob, _stored, _norm = _prepare_vector(vector, self._dimension)

        try:
            with self._write_txn() as conn:
                self._verify_ready(conn)
                self._assert_canonical_authority(valid_id, meta)
                conn.execute(
                    f"""
                    INSERT INTO {VECTOR_TABLE}
                        (memory_id, model_id, dimension, vector, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        model_id = excluded.model_id,
                        dimension = excluded.dimension,
                        vector = excluded.vector,
                        updated_at = excluded.updated_at
                    """,
                    (valid_id, self._model_id, self._dimension, blob, _utc_now_str()),
                )
        except (SQLiteVectorError, SQLiteVectorBackendError):
            raise
        except sqlite3.Error:
            raise SQLiteVectorBackendError("vector upsert failed") from None

    def _assert_canonical_authority(self, memory_id: str, meta: dict[str, Any]) -> None:
        """Reject payloads that differ from the current active canonical record."""
        try:
            record = self._store.get_index_record(
                meta["owner_id"], meta["creator_character_id"], memory_id
            )
        except Exception:
            raise SQLiteVectorBackendError("canonical store read failed") from None
        if record is None:
            raise SQLiteVectorAuthorityError(
                "memory is not an active canonical record for the claimed scope"
            )
        try:
            creator = record.get("creator_character_id", record.get("character_id"))
            matches = (
                record.get("memory_id") == memory_id
                and record.get("owner_id") == meta["owner_id"]
                and creator == meta["creator_character_id"]
                and record.get("domain") == meta["domain"]
                and list(record.get("visible_to", [])) == meta["visible_to"]
            )
        except Exception:
            raise SQLiteVectorBackendError("canonical record is invalid") from None
        if not matches:
            raise SQLiteVectorAuthorityError(
                "payload metadata does not match the canonical record"
            )

    def delete(self, memory_id: str) -> None:
        """Idempotently remove the vector for `memory_id`; never touches `memories`."""
        try:
            valid_id = _validate_uuid(memory_id, "memory_id")
        except Exception:
            raise SQLiteVectorError("memory_id is invalid") from None
        try:
            with self._write_txn() as conn:
                self._verify_ready(conn)
                conn.execute(
                    f"DELETE FROM {VECTOR_TABLE} WHERE memory_id = ?", (valid_id,)
                )
        except (SQLiteVectorError, SQLiteVectorBackendError):
            raise
        except sqlite3.Error:
            raise SQLiteVectorBackendError("vector delete failed") from None


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------


class SQLiteSemanticSearcher:
    """Fail-closed semantic search; caller supplies authenticated owner/reader IDs."""

    def __init__(
        self, store: Any, embedder: Any, backend: SQLiteVectorBackend | None
    ) -> None:
        if store is None or not (
            callable(getattr(store, "get_memory", None))
            and callable(getattr(store, "_connect", None))
        ):
            raise ValueError("store must provide get_memory and _connect")
        if backend is not None and (
            not isinstance(backend, SQLiteVectorBackend) or backend.store is not store
        ):
            raise ValueError("backend must be a SQLiteVectorBackend over the same store")
        self.store = store
        self.embedder = embedder
        self.backend = backend

    def search(
        self,
        owner_id: str,
        character_id: str,
        query: str,
        limit: int = 5,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        query = _validate_text(query, "query", MAX_QUERY_LENGTH)
        limit = _validate_limit(limit)
        if domain is not None and (
            not isinstance(domain, str) or domain not in VALID_DOMAINS
        ):
            raise ValueError("invalid domain")

        embedder, backend = self.embedder, self.backend
        if embedder is None or backend is None:
            raise SemanticSearchError("search dependency unavailable")
        if not callable(getattr(embedder, "embed", None)):
            raise SemanticSearchError("embedder unavailable")

        try:
            model_id = _validate_id(embedder.model_id, "embedder.model_id")
            dimension = _validate_dimension(embedder.dimension, "embedder.dimension")
        except Exception:
            raise SemanticSearchError("embedding configuration invalid") from None
        if model_id != backend.model_id or dimension != backend.dimension:
            raise SemanticSearchError("embedding configuration mismatch")

        try:
            raw_vector = embedder.embed(query)
            _blob, query_vector, query_norm = _prepare_vector(raw_vector, dimension)
        except Exception:
            raise SemanticSearchError("embedding failed") from None

        try:
            ranked = self._rank_candidates(
                owner_id, character_id, domain, model_id, dimension,
                query_vector, query_norm,
            )
        except Exception:
            raise SemanticSearchError("vector search failed") from None

        # Canonical recheck of every returned record (revoked/deleted rows drop out).
        results: list[dict[str, Any]] = []
        try:
            for score, memory_id in ranked[: limit + MAX_LIMIT]:
                record = self.store.get_memory(owner_id, character_id, memory_id)
                if record is None:
                    continue
                visible = record.get("visible_to", [])
                if (
                    record.get("memory_id") != memory_id
                    or record.get("owner_id") != owner_id
                    or (domain is not None and record.get("domain") != domain)
                    or (
                        record.get("creator_character_id") != character_id
                        and character_id not in visible
                    )
                ):
                    continue
                sources = record.get("source_ids")
                try:
                    if not sources or not isinstance(sources, list):
                        continue
                    _validate_source_ids(sources)
                except Exception:
                    continue
                results.append({**record, "score": score})
                if len(results) == limit:
                    break
        except Exception:
            raise SemanticSearchError("canonical search verification failed") from None
        return results

    def _rank_candidates(
        self,
        owner_id: str,
        character_id: str,
        domain: str | None,
        model_id: str,
        dimension: int,
        query_vector: tuple[float, ...],
        query_norm: float,
    ) -> list[tuple[float, str]]:
        """Score every canonical-authorized candidate; any corrupt candidate raises.

        Authorization (owner, active status, domain, creator/ACL) is part of the
        SQL JOIN against the current canonical tables, so ranking only ever sees
        authorized rows and unauthorized/stale rows cannot starve results.
        An incomplete index (missing vector for an authorized active memory)
        or corruption raises SQLiteVectorBackendError immediately.
        """
        params: list[Any] = [owner_id]
        domain_clause = ""
        if domain is not None:
            domain_clause = "AND m.domain = ?"
            params.append(domain)
        params.extend([character_id, character_id])

        with self.store._connect() as conn:
            if conn.in_transaction:
                raise SQLiteVectorBackendError('search cannot own an existing transaction')
            conn.execute("BEGIN")
            try:
                if self.backend is None:
                    raise SQLiteVectorBackendError("vector backend unavailable")
                self.backend._verify_ready(conn)
                rows = conn.execute(
                    f"""
                    SELECT
                        m.memory_id,
                        v.memory_id,
                        v.model_id,
                        v.dimension,
                        v.vector
                    FROM memories AS m
                    LEFT JOIN {VECTOR_TABLE} AS v ON v.memory_id = m.memory_id
                    WHERE m.owner_id = ?
                      AND m.status = 'active'
                      {domain_clause}
                      AND (
                          m.character_id = ?
                          OR EXISTS (
                              SELECT 1 FROM memory_acl AS a
                              WHERE a.memory_id = m.memory_id
                                AND a.owner_id = m.owner_id
                                AND a.character_id = ?
                          )
                      )
                    ORDER BY m.memory_id
                    """,
                    tuple(params),
                ).fetchall()
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    try:
                        conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                raise

        scored: list[tuple[float, str]] = []
        for row in rows:
            canonical_mid = _validate_uuid(row[0], "memory_id")
            vector_mid = row[1]
            if vector_mid is None:
                raise SQLiteVectorBackendError("vector index incomplete: missing vector row")
            if vector_mid != canonical_mid:
                raise SQLiteVectorBackendError("vector index corrupt: memory_id mismatch")
            if row[2] != model_id or type(row[3]) is not int or row[3] != dimension:
                raise SQLiteVectorBackendError("stored vector model binding mismatch")
            values, norm = _decode_blob(row[4], dimension)
            dot = sum(map(operator.mul, query_vector, values))
            score = dot / (query_norm * norm)
            if not math.isfinite(score) or abs(score) > 1.0 + _COSINE_TOLERANCE:
                raise SQLiteVectorBackendError("stored vector score is invalid")
            scored.append((max(-1.0, min(1.0, float(score))), canonical_mid))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored
