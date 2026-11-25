"""Focused tests for DeferredMemoryQueue constructor transaction and initialization.

Verifies:
1. Fresh constructor works in :memory: and tempfile databases.
2. Forcing initial scope INSERT failure (via SQLite authorizer injection) causes
   full atomic rollback with no partial new schema and original messages unchanged.
3. Existing caller transaction is rejected before BEGIN, leaving caller transaction
   and data intact.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch

from memory_core import MemoryStore
from memory_deferred_queue import DeferredMemoryQueue, DeferredQueueError


class TestDeferredMemoryQueueInit(unittest.TestCase):
    def test_fresh_constructor_in_memory(self) -> None:
        """Fresh constructor works in :memory: database and atomically creates schema + scope."""
        store = MemoryStore(":memory:")
        try:
            store.append_message(
                "user-init-1",
                "char-init-1",
                "ext-init-1",
                "user",
                "Initial message prior to queue",
            )
            queue = DeferredMemoryQueue(store, "user-init-1", "char-init-1")

            # Verify queue tables and index exist
            with store._connect() as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'deferred_%'"
                    ).fetchall()
                }
                self.assertIn("deferred_memory_scopes", tables)
                self.assertIn("deferred_memory_jobs", tables)

                indices = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_deferred_%'"
                    ).fetchall()
                }
                self.assertIn("idx_deferred_jobs_scope_state", indices)

                # Verify scope registration row
                scope_row = conn.execute(
                    "SELECT owner_id, character_id, scan_cursor FROM deferred_memory_scopes WHERE owner_id = ? AND character_id = ?",
                    ("user-init-1", "char-init-1"),
                ).fetchone()
                self.assertIsNotNone(scope_row)
                self.assertEqual(scope_row[0], "user-init-1")
                self.assertEqual(scope_row[1], "char-init-1")
                self.assertEqual(scope_row[2], 1)

            st = queue.status()
            self.assertEqual(st["pending_count"], 0)
            self.assertEqual(st["processing_count"], 0)
        finally:
            store.close()

    def test_fresh_constructor_tempfile(self) -> None:
        """Fresh constructor works in a file-backed database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_fresh_init.db")
            store = MemoryStore(db_path)
            try:
                store.append_message(
                    "user-init-2",
                    "char-init-2",
                    "ext-init-2",
                    "user",
                    "Initial file-backed message",
                )
                queue = DeferredMemoryQueue(store, "user-init-2", "char-init-2")

                with store._connect() as conn:
                    scope_row = conn.execute(
                        "SELECT scan_cursor FROM deferred_memory_scopes WHERE owner_id = ? AND character_id = ?",
                        ("user-init-2", "char-init-2"),
                    ).fetchone()
                    self.assertIsNotNone(scope_row)
                    self.assertEqual(scope_row[0], 1)

                st = queue.status()
                self.assertEqual(st["pending_count"], 0)
            finally:
                store.close()

    def test_force_initial_scope_insert_failure_rolls_back_in_memory(self) -> None:
        """Forcing initial scope INSERT failure via SQLite authorizer rolls back DDL atomically.

        Ensures no partial new schema (no queue tables or indices) remains and original
        messages are completely unchanged.
        """
        store = MemoryStore(":memory:")
        try:
            store.append_message(
                "user-auth-1",
                "char-auth-1",
                "ext-auth-1",
                "user",
                "Original untouched message in memory",
            )

            # SQLite authorizer callback that denies INSERT into deferred_memory_scopes
            def deny_scope_insert(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger_name: str | None) -> int:
                if action == sqlite3.SQLITE_INSERT and arg1 == "deferred_memory_scopes":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            store._memory_conn.set_authorizer(deny_scope_insert)
            try:
                with self.assertRaises(DeferredQueueError):
                    DeferredMemoryQueue(store, "user-auth-1", "char-auth-1")
            finally:
                store._memory_conn.set_authorizer(None)

            # Verify complete rollback: no deferred_* tables or indices exist
            with store._connect() as conn:
                deferred_objects = conn.execute(
                    "SELECT type, name FROM sqlite_master WHERE name LIKE 'deferred_%' OR name LIKE 'idx_deferred_%'"
                ).fetchall()
                self.assertEqual(
                    deferred_objects,
                    [],
                    f"Expected no partial schema after rollback, found: {deferred_objects}",
                )

                # Original messages table and content must remain completely unchanged
                msg_row = conn.execute(
                    "SELECT message_id, content FROM messages WHERE owner_id = ? AND character_id = ?",
                    ("user-auth-1", "char-auth-1"),
                ).fetchone()
                self.assertIsNotNone(msg_row)
                self.assertEqual(msg_row["content"], "Original untouched message in memory")
        finally:
            store.close()

    def test_force_initial_scope_insert_failure_rolls_back_tempfile(self) -> None:
        """Forcing initial scope INSERT failure in tempfile database rolls back DDL atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_rollback_file.db")
            store = MemoryStore(db_path)
            try:
                store.append_message(
                    "user-auth-2",
                    "char-auth-2",
                    "ext-auth-2",
                    "user",
                    "Original untouched message on disk",
                )

                orig_create = store._create_connection

                def create_with_authorizer() -> sqlite3.Connection:
                    conn = orig_create()

                    def deny_scope_insert(action: int, arg1: str | None, arg2: str | None, db_name: str | None, trigger_name: str | None) -> int:
                        if action == sqlite3.SQLITE_INSERT and arg1 == "deferred_memory_scopes":
                            return sqlite3.SQLITE_DENY
                        return sqlite3.SQLITE_OK

                    conn.set_authorizer(deny_scope_insert)
                    return conn

                store._create_connection = create_with_authorizer
                try:
                    with self.assertRaises(DeferredQueueError):
                        DeferredMemoryQueue(store, "user-auth-2", "char-auth-2")
                finally:
                    store._create_connection = orig_create

                # Verify with normal connection: no deferred schema created, original message intact
                with store._connect() as conn:
                    deferred_objects = conn.execute(
                        "SELECT type, name FROM sqlite_master WHERE name LIKE 'deferred_%' OR name LIKE 'idx_deferred_%'"
                    ).fetchall()
                    self.assertEqual(
                        deferred_objects,
                        [],
                        f"Expected no partial schema after rollback on disk, found: {deferred_objects}",
                    )

                    msg_row = conn.execute(
                        "SELECT content FROM messages WHERE owner_id = ? AND character_id = ?",
                        ("user-auth-2", "char-auth-2"),
                    ).fetchone()
                    self.assertIsNotNone(msg_row)
                    self.assertEqual(msg_row["content"], "Original untouched message on disk")
            finally:
                store.close()

    def test_caller_transaction_rejected_unchanged_in_memory(self) -> None:
        """Active caller transaction is rejected before BEGIN without modifying caller transaction state or data."""
        store = MemoryStore(":memory:")
        try:
            store.append_message(
                "user-caller-1",
                "char-caller-1",
                "ext-pre-1",
                "user",
                "Pre-existing committed message",
            )

            with store._connect() as conn:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    INSERT INTO messages (
                        message_id, owner_id, character_id, external_id,
                        role, content, occurred_at, created_at, payload_hash
                    ) VALUES ('caller-uncommitted-msg-1', 'user-caller-1', 'char-caller-1', 'ext-uncommitted-1',
                              'user', 'Uncommitted caller message', '2026-01-01T00:00:00.000000Z',
                              '2026-01-01T00:00:00.000000Z', 'hash-caller-1')
                    """
                )

                # Attempting to construct queue must raise DeferredQueueError immediately
                with self.assertRaises(DeferredQueueError) as ctx:
                    DeferredMemoryQueue(store, "user-caller-1", "char-caller-1")
                self.assertIn("Caller transaction already active", str(ctx.exception))

                # Caller transaction must remain actively open and uncommitted
                self.assertTrue(conn.in_transaction)

                # Caller can now successfully commit its own transaction
                conn.execute("COMMIT")

            # Verify caller transaction committed cleanly and data is intact
            with store._connect() as conn:
                rows = conn.execute(
                    "SELECT message_id, content FROM messages WHERE owner_id = ? AND character_id = ? ORDER BY seq",
                    ("user-caller-1", "char-caller-1"),
                ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["content"], "Pre-existing committed message")
                self.assertEqual(rows[1]["content"], "Uncommitted caller message")

                # Verify queue schema was NOT created because rejection happened before BEGIN
                deferred_objects = conn.execute(
                    "SELECT name FROM sqlite_master WHERE name LIKE 'deferred_%' OR name LIKE 'idx_deferred_%'"
                ).fetchall()
                self.assertEqual(deferred_objects, [])
        finally:
            store.close()

    def test_caller_transaction_rejected_unchanged_tempfile(self) -> None:
        """Active caller transaction rejection behaves identically on disk database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_caller_file.db")
            store = MemoryStore(db_path)
            try:
                store.append_message(
                    "user-caller-2",
                    "char-caller-2",
                    "ext-pre-2",
                    "user",
                    "Pre-existing disk message",
                )

                with store._connect() as conn:
                    conn.execute("BEGIN")
                    conn.execute(
                        """
                        INSERT INTO messages (
                            message_id, owner_id, character_id, external_id,
                            role, content, occurred_at, created_at, payload_hash
                        ) VALUES ('caller-uncommitted-msg-2', 'user-caller-2', 'char-caller-2', 'ext-uncommitted-2',
                                  'user', 'Uncommitted disk caller message', '2026-01-01T00:00:00.000000Z',
                                  '2026-01-01T00:00:00.000000Z', 'hash-caller-2')
                        """
                    )

                    # Reuse the actual caller connection; file stores normally open
                    # a different connection, which tests cross-connection locking instead.
                    with patch.object(store, "_connect", return_value=nullcontext(conn)):
                        with self.assertRaises(DeferredQueueError):
                            DeferredMemoryQueue(store, "user-caller-2", "char-caller-2")

                    self.assertTrue(conn.in_transaction)
                    conn.execute("COMMIT")

                with store._connect() as conn:
                    rows = conn.execute(
                        "SELECT message_id, content FROM messages WHERE owner_id = ? AND character_id = ? ORDER BY seq",
                        ("user-caller-2", "char-caller-2"),
                    ).fetchall()
                    self.assertEqual(len(rows), 2)
                    self.assertEqual(rows[1]["content"], "Uncommitted disk caller message")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
