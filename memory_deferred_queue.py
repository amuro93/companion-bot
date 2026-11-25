"""Durable deferred-memory queue storage (TASK A-5a).

Standard-library-only queue storage engine providing scope-isolated
capture, recovery, leasing, completion, and retry persistence for
deferred memory extraction pipelines.
"""

from __future__ import annotations

from functools import wraps
import math
import os
import sqlite3
import threading
import time
from typing import Any, Callable
import uuid

from memory_core import MemoryStore, _validate_id, _validate_text
from memory_trigger import message_external_id

__all__ = ["DeferredMemoryQueue", "DeferredQueueError"]

VALID_ERROR_CATEGORIES = frozenset(
    {"processor_failed", "index_failed", "invalid_result"}
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS deferred_memory_scopes (
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    scan_cursor INTEGER NOT NULL DEFAULT 0,
    last_activity REAL NOT NULL,
    last_success REAL,
    last_error_category TEXT,
    PRIMARY KEY (owner_id, character_id)
);

CREATE TABLE IF NOT EXISTS deferred_memory_jobs (
    owner_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    source_seq INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'processing', 'done', 'excluded')),
    attempts INTEGER NOT NULL DEFAULT 0,
    claim_token TEXT,
    lease_until REAL,
    next_attempt_at REAL,
    completed_at REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_id, character_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_deferred_jobs_scope_state
    ON deferred_memory_jobs (owner_id, character_id, state, source_seq ASC);
"""


class DeferredQueueError(Exception):
    """Sanitized exception raised for storage, clock, or caller-transaction failures."""

    pass


def _sanitize_storage(method: Callable[..., Any]) -> Callable[..., Any]:
    """Sanitize SQLite and OS storage errors to fixed DeferredQueueError."""

    @wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except (sqlite3.Error, OSError):
            raise DeferredQueueError("Queue storage unavailable") from None

    return wrapper


class DeferredMemoryQueue:
    """Isolated SQLite companion queue storage for deferred extraction pipelines.

    Bound strictly to a trusted (owner_id, character_id) scope. Concurrency and
    leasing coordination are managed through SQLite immediate transactions.
    """

    @_sanitize_storage
    def __init__(
        self,
        store: MemoryStore,
        owner_id: str,
        character_id: str,
        idle_seconds: float = 180,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if store is None or not hasattr(store, "_connect"):
            raise ValueError("store must be a valid MemoryStore instance")

        self.owner_id = _validate_id(owner_id, "owner_id")
        self.character_id = _validate_id(character_id, "character_id")

        if isinstance(idle_seconds, bool) or not isinstance(idle_seconds, (int, float)):
            raise ValueError("idle_seconds must be a finite positive number <= 86400")
        if (
            math.isnan(idle_seconds)
            or math.isinf(idle_seconds)
            or idle_seconds <= 0
            or idle_seconds > 86400
        ):
            raise ValueError("idle_seconds must be a finite positive number <= 86400")
        self.idle_seconds = float(idle_seconds)

        if not callable(clock):
            raise ValueError("clock must be callable")
        self._clock = clock
        self.store = store
        self._lock = threading.RLock()

        now = self._now()

        with self._lock:
            with self.store._connect() as conn:
                if conn.in_transaction:
                    raise DeferredQueueError("Caller transaction already active")
                self.store._begin_immediate(conn)
                try:
                    self._ensure_schema(conn)
                    row = conn.execute(
                        """
                        SELECT scan_cursor, last_activity
                        FROM deferred_memory_scopes
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (self.owner_id, self.character_id),
                    ).fetchone()

                    if row is None:
                        max_seq_row = conn.execute(
                            """
                            SELECT COALESCE(MAX(seq), 0)
                            FROM messages
                            WHERE owner_id = ? AND character_id = ?
                            """,
                            (self.owner_id, self.character_id),
                        ).fetchone()
                        initial_cursor = int(max_seq_row[0]) if max_seq_row else 0
                        conn.execute(
                            """
                            INSERT INTO deferred_memory_scopes (
                                owner_id, character_id, scan_cursor, last_activity, last_success, last_error_category
                            ) VALUES (?, ?, ?, ?, NULL, NULL)
                            """,
                            (self.owner_id, self.character_id, initial_cursor, now),
                        )
                    conn.execute("COMMIT")
                except Exception as exc:
                    self.store._safe_rollback(conn)
                    if isinstance(exc, (DeferredQueueError, ValueError)):
                        raise
                    raise DeferredQueueError("Queue initialization failed") from None

            self.recover()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        for statement in _SCHEMA_SQL.split(";"):
            sql = statement.strip()
            if sql:
                conn.execute(sql)

    def _now(self) -> float:
        try:
            val = self._clock()
        except Exception:
            raise DeferredQueueError("Clock invocation failed") from None
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise DeferredQueueError("Clock returned non-numeric value")
        if math.isnan(val) or math.isinf(val):
            raise DeferredQueueError("Clock returned non-finite value")
        return float(val)

    def _check_no_caller_transaction(self) -> None:
        with self.store._connect() as conn:
            if conn.in_transaction:
                raise DeferredQueueError("Caller transaction already active")

    def _touch_activity(self, now: float) -> None:
        with self.store._connect() as conn:
            if conn.in_transaction:
                raise DeferredQueueError("Caller transaction already active")
            self.store._begin_immediate(conn)
            try:
                conn.execute(
                    """
                    UPDATE deferred_memory_scopes
                    SET last_activity = MAX(last_activity, ?)
                    WHERE owner_id = ? AND character_id = ?
                    """,
                    (now, self.owner_id, self.character_id),
                )
                conn.execute("COMMIT")
            except Exception as exc:
                self.store._safe_rollback(conn)
                if isinstance(exc, (DeferredQueueError, ValueError)):
                    raise
                raise DeferredQueueError("Failed to update scope activity") from None

    @_sanitize_storage
    def capture(self, content: str) -> str:
        """Append user message idempotently, update activity, and recover queue."""
        content = _validate_text(content, "content", max_length=1024)

        with self._lock:
            self._check_no_caller_transaction()
            ext_id = message_external_id(content)
            message_id = self.store.append_message(
                self.owner_id,
                self.character_id,
                ext_id,
                "user",
                content,
            )
            now = self._now()
            self._touch_activity(now)
            self.recover()
            return message_id

    @_sanitize_storage
    def recover(self) -> int:
        """Transactionally insert missing jobs and advance scan cursor."""
        with self._lock:
            with self.store._connect() as conn:
                if conn.in_transaction:
                    raise DeferredQueueError("Caller transaction already active")
                self.store._begin_immediate(conn)
                try:
                    scope = conn.execute(
                        """
                        SELECT scan_cursor
                        FROM deferred_memory_scopes
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (self.owner_id, self.character_id),
                    ).fetchone()

                    if scope is None:
                        raise DeferredQueueError("Scope not registered")

                    current_cursor = int(scope["scan_cursor"])

                    rows = conn.execute(
                        """
                        SELECT seq, message_id, role, external_id, excluded_from_memory
                        FROM messages
                        WHERE owner_id = ? AND character_id = ? AND seq > ?
                        ORDER BY seq ASC
                        """,
                        (self.owner_id, self.character_id, current_cursor),
                    ).fetchall()

                    if not rows:
                        conn.execute("COMMIT")
                        return 0

                    inserted_count = 0
                    max_scanned_seq = current_cursor
                    now = self._now()

                    for row in rows:
                        seq = int(row["seq"])
                        if seq > max_scanned_seq:
                            max_scanned_seq = seq

                        role = str(row["role"])
                        ext_id = str(row["external_id"])
                        if role != "user" or not ext_id.startswith("tg-"):
                            continue

                        msg_id = str(row["message_id"])
                        is_excluded = int(row["excluded_from_memory"] or 0) == 1
                        job_state = "excluded" if is_excluded else "pending"
                        completed_at = now if is_excluded else None

                        cursor = conn.execute(
                            """
                            INSERT OR IGNORE INTO deferred_memory_jobs (
                                owner_id, character_id, message_id, source_seq,
                                state, attempts, claim_token, lease_until,
                                next_attempt_at, completed_at, created_at
                            ) VALUES (?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?)
                            """,
                            (
                                self.owner_id,
                                self.character_id,
                                msg_id,
                                seq,
                                job_state,
                                completed_at,
                                now,
                            ),
                        )
                        if cursor.rowcount > 0:
                            inserted_count += 1

                    if inserted_count > 0:
                        conn.execute(
                            """
                            UPDATE deferred_memory_scopes
                            SET scan_cursor = ?,
                                last_activity = MAX(last_activity, ?)
                            WHERE owner_id = ? AND character_id = ?
                            """,
                            (max_scanned_seq, now, self.owner_id, self.character_id),
                        )

                    conn.execute("COMMIT")
                    return inserted_count
                except Exception as exc:
                    self.store._safe_rollback(conn)
                    if isinstance(exc, (DeferredQueueError, ValueError)):
                        raise
                    raise DeferredQueueError("Recover operation failed") from None

    @_sanitize_storage
    def claim_due(self, lease_seconds: float = 300) -> dict[str, Any] | None:
        """Claim the earliest nonterminal job if idle debounce and retry delay have passed."""
        if isinstance(lease_seconds, bool) or not isinstance(
            lease_seconds, (int, float)
        ):
            raise ValueError("lease_seconds must be a finite positive number <= 86400")
        if (
            math.isnan(lease_seconds)
            or math.isinf(lease_seconds)
            or lease_seconds <= 0
            or lease_seconds > 86400
        ):
            raise ValueError("lease_seconds must be a finite positive number <= 86400")
        lease_seconds = float(lease_seconds)

        with self._lock:
            now = self._now()
            with self.store._connect() as conn:
                if conn.in_transaction:
                    raise DeferredQueueError("Caller transaction already active")
                self.store._begin_immediate(conn)
                try:
                    scope = conn.execute(
                        """
                        SELECT last_activity
                        FROM deferred_memory_scopes
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (self.owner_id, self.character_id),
                    ).fetchone()

                    if scope is None:
                        raise DeferredQueueError("Scope not registered")

                    last_activity = float(scope["last_activity"])
                    if now < last_activity + self.idle_seconds:
                        conn.execute("COMMIT")
                        return None

                    while True:
                        job = conn.execute(
                            """
                            SELECT message_id, source_seq, state, attempts, claim_token, lease_until, next_attempt_at
                            FROM deferred_memory_jobs
                            WHERE owner_id = ? AND character_id = ? AND state IN ('pending', 'processing')
                            ORDER BY source_seq ASC
                            LIMIT 1
                            """,
                            (self.owner_id, self.character_id),
                        ).fetchone()

                        if job is None:
                            conn.execute("COMMIT")
                            return None

                        msg_id = str(job["message_id"])

                        msg = conn.execute(
                            """
                            SELECT content, excluded_from_memory
                            FROM messages
                            WHERE owner_id = ? AND character_id = ? AND message_id = ?
                            """,
                            (self.owner_id, self.character_id, msg_id),
                        ).fetchone()

                        if msg is None:
                            raise DeferredQueueError(
                                "Canonical message missing for queued job"
                            )

                        if int(msg["excluded_from_memory"] or 0) == 1:
                            conn.execute(
                                """
                                UPDATE deferred_memory_jobs
                                SET state = 'excluded',
                                    completed_at = ?,
                                    claim_token = NULL,
                                    lease_until = NULL
                                WHERE owner_id = ? AND character_id = ? AND message_id = ?
                                """,
                                (now, self.owner_id, self.character_id, msg_id),
                            )
                            continue

                        state = str(job["state"])
                        if state == "processing":
                            lease_until = job["lease_until"]
                            if lease_until is not None and now < float(lease_until):
                                conn.execute("COMMIT")
                                return None
                        elif state == "pending":
                            next_attempt_at = job["next_attempt_at"]
                            if next_attempt_at is not None and now < float(
                                next_attempt_at
                            ):
                                conn.execute("COMMIT")
                                return None

                        claim_token = str(uuid.uuid4())
                        new_attempts = int(job["attempts"]) + 1
                        new_lease_until = now + lease_seconds

                        conn.execute(
                            """
                            UPDATE deferred_memory_jobs
                            SET state = 'processing',
                                claim_token = ?,
                                lease_until = ?,
                                attempts = ?,
                                next_attempt_at = NULL
                            WHERE owner_id = ? AND character_id = ? AND message_id = ?
                            """,
                            (
                                claim_token,
                                new_lease_until,
                                new_attempts,
                                self.owner_id,
                                self.character_id,
                                msg_id,
                            ),
                        )
                        conn.execute("COMMIT")
                        return {
                            "message_id": msg_id,
                            "content": str(msg["content"]),
                            "claim_token": claim_token,
                            "attempts": new_attempts,
                        }
                except Exception as exc:
                    self.store._safe_rollback(conn)
                    if isinstance(exc, (DeferredQueueError, ValueError)):
                        raise
                    raise DeferredQueueError("Claim operation failed") from None

    @_sanitize_storage
    def complete(self, message_id: str, claim_token: str) -> bool:
        """Mark exact processing job done and record success in scope."""
        message_id = _validate_id(message_id, "message_id")
        claim_token = _validate_id(claim_token, "claim_token")

        with self._lock:
            now = self._now()
            with self.store._connect() as conn:
                if conn.in_transaction:
                    raise DeferredQueueError("Caller transaction already active")
                self.store._begin_immediate(conn)
                try:
                    row = conn.execute(
                        """
                        SELECT state, claim_token
                        FROM deferred_memory_jobs
                        WHERE owner_id = ? AND character_id = ? AND message_id = ?
                        """,
                        (self.owner_id, self.character_id, message_id),
                    ).fetchone()

                    if (
                        row is None
                        or row["state"] != "processing"
                        or row["claim_token"] != claim_token
                    ):
                        conn.execute("COMMIT")
                        return False

                    conn.execute(
                        """
                        UPDATE deferred_memory_jobs
                        SET state = 'done',
                            completed_at = ?,
                            claim_token = NULL,
                            lease_until = NULL
                        WHERE owner_id = ? AND character_id = ? AND message_id = ? AND claim_token = ?
                        """,
                        (now, self.owner_id, self.character_id, message_id, claim_token),
                    )
                    conn.execute(
                        """
                        UPDATE deferred_memory_scopes
                        SET last_success = ?,
                            last_error_category = NULL
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (now, self.owner_id, self.character_id),
                    )
                    conn.execute("COMMIT")
                    return True
                except Exception as exc:
                    self.store._safe_rollback(conn)
                    if isinstance(exc, (DeferredQueueError, ValueError)):
                        raise
                    raise DeferredQueueError("Complete operation failed") from None

    @_sanitize_storage
    def fail(
        self,
        message_id: str,
        claim_token: str,
        error_category: str = "processor_failed",
        retry_seconds: float = 30,
    ) -> bool:
        """Return job to pending state with retry delay and record error category."""
        if (
            not isinstance(error_category, str)
            or error_category not in VALID_ERROR_CATEGORIES
        ):
            raise ValueError("Invalid error_category")

        if isinstance(retry_seconds, bool) or not isinstance(
            retry_seconds, (int, float)
        ):
            raise ValueError(
                "retry_seconds must be a finite non-negative number <= 86400"
            )
        if (
            math.isnan(retry_seconds)
            or math.isinf(retry_seconds)
            or retry_seconds < 0
            or retry_seconds > 86400
        ):
            raise ValueError(
                "retry_seconds must be a finite non-negative number <= 86400"
            )
        retry_seconds = float(retry_seconds)

        message_id = _validate_id(message_id, "message_id")
        claim_token = _validate_id(claim_token, "claim_token")

        with self._lock:
            now = self._now()
            next_attempt_at = now + retry_seconds

            with self.store._connect() as conn:
                if conn.in_transaction:
                    raise DeferredQueueError("Caller transaction already active")
                self.store._begin_immediate(conn)
                try:
                    row = conn.execute(
                        """
                        SELECT state, claim_token
                        FROM deferred_memory_jobs
                        WHERE owner_id = ? AND character_id = ? AND message_id = ?
                        """,
                        (self.owner_id, self.character_id, message_id),
                    ).fetchone()

                    if (
                        row is None
                        or row["state"] != "processing"
                        or row["claim_token"] != claim_token
                    ):
                        conn.execute("COMMIT")
                        return False

                    conn.execute(
                        """
                        UPDATE deferred_memory_jobs
                        SET state = 'pending',
                            claim_token = NULL,
                            lease_until = NULL,
                            next_attempt_at = ?
                        WHERE owner_id = ? AND character_id = ? AND message_id = ? AND claim_token = ?
                        """,
                        (
                            next_attempt_at,
                            self.owner_id,
                            self.character_id,
                            message_id,
                            claim_token,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE deferred_memory_scopes
                        SET last_error_category = ?
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (error_category, self.owner_id, self.character_id),
                    )
                    conn.execute("COMMIT")
                    return True
                except Exception as exc:
                    self.store._safe_rollback(conn)
                    if isinstance(exc, (DeferredQueueError, ValueError)):
                        raise
                    raise DeferredQueueError("Fail operation failed") from None

    @_sanitize_storage
    def status(self) -> dict[str, Any]:
        """Return non-sensitive queue metrics and scope status."""
        with self._lock:
            with self.store._connect() as conn:
                if conn.in_transaction:
                    raise DeferredQueueError("Caller transaction already active")
                try:
                    scope = conn.execute(
                        """
                        SELECT last_activity, last_success, last_error_category
                        FROM deferred_memory_scopes
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (self.owner_id, self.character_id),
                    ).fetchone()

                    if scope is None:
                        raise DeferredQueueError("Scope not registered")

                    counts = conn.execute(
                        """
                        SELECT
                            SUM(CASE WHEN state IN ('pending', 'processing') THEN 1 ELSE 0 END) AS pending_count,
                            SUM(CASE WHEN state = 'processing' THEN 1 ELSE 0 END) AS processing_count,
                            SUM(CASE WHEN state = 'excluded' THEN 1 ELSE 0 END) AS excluded_count,
                            SUM(CASE WHEN state = 'done' THEN 1 ELSE 0 END) AS done_count
                        FROM deferred_memory_jobs
                        WHERE owner_id = ? AND character_id = ?
                        """,
                        (self.owner_id, self.character_id),
                    ).fetchone()

                    return {
                        "pending_count": int(counts["pending_count"] or 0),
                        "processing_count": int(counts["processing_count"] or 0),
                        "excluded_count": int(counts["excluded_count"] or 0),
                        "done_count": int(counts["done_count"] or 0),
                        "last_activity": (
                            float(scope["last_activity"])
                            if scope["last_activity"] is not None
                            else None
                        ),
                        "last_success": (
                            float(scope["last_success"])
                            if scope["last_success"] is not None
                            else None
                        ),
                        "last_error_category": (
                            str(scope["last_error_category"])
                            if scope["last_error_category"] is not None
                            else None
                        ),
                    }
                except Exception as exc:
                    if isinstance(exc, (DeferredQueueError, ValueError)):
                        raise
                    raise DeferredQueueError("Status operation failed") from None
