"""Tests for storage error boundary and sanitization (TASK A-5a)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from memory_core import MemoryStore
from memory_deferred_queue import DeferredMemoryQueue, DeferredQueueError


class FakeClock:
    def __init__(self, initial_time: float = 1000.0) -> None:
        self.current_time = float(initial_time)

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += float(seconds)


class TestStorageErrorSanitization(unittest.TestCase):
    def setUp(self) -> None:
        self.owner_id = "user-sec-1"
        self.character_id = "char-sec-1"
        self.clock = FakeClock(1000.0)

    def test_native_trigger_payload_leak_prevented(self) -> None:
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store, self.owner_id, self.character_id, clock=self.clock
            )
            payload = "FICTIONAL_PRIVATE_PAYLOAD_927"
            with store._connect() as conn:
                conn.execute(
                    f"CREATE TRIGGER reject_messages BEFORE INSERT ON messages "
                    f"BEGIN SELECT RAISE(ABORT, '{payload}'); END;"
                )

            with self.assertRaises(DeferredQueueError) as ctx:
                queue.capture("sensitive message content")

            err_str, err_repr = str(ctx.exception), repr(ctx.exception)
            self.assertEqual(err_str, "Queue storage unavailable")
            self.assertNotIn(payload, err_str)
            self.assertNotIn(payload, err_repr)

            with store._connect() as conn:
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                job_count = conn.execute(
                    "SELECT COUNT(*) FROM deferred_memory_jobs"
                ).fetchone()[0]
                self.assertEqual(msg_count, 0)
                self.assertEqual(job_count, 0)
        finally:
            store.close()

    def test_authorizer_denies_begin_across_writing_entry_points(self) -> None:
        store = MemoryStore(":memory:")
        try:
            def deny_begin(action: int, arg1: str | None, *_) -> int:
                if action == sqlite3.SQLITE_TRANSACTION and arg1 == "BEGIN":
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            with store._connect() as conn:
                conn.set_authorizer(deny_begin)

            # __init__ denied BEGIN
            with self.assertRaises(DeferredQueueError) as ctx:
                DeferredMemoryQueue(
                    store, self.owner_id, self.character_id, clock=self.clock
                )
            self.assertEqual(str(ctx.exception), "Queue storage unavailable")
            with store._connect() as conn:
                self.assertFalse(conn.in_transaction)
                conn.set_authorizer(None)

            # Normal setup to create state
            queue = DeferredMemoryQueue(
                store, self.owner_id, self.character_id, idle_seconds=100, clock=self.clock
            )
            msg_id = queue.capture("Initial message")
            st_before = queue.status()
            self.assertEqual(st_before["pending_count"], 1)

            # Deny BEGIN across remaining writing entry points
            with store._connect() as conn:
                conn.set_authorizer(deny_begin)

            for call, desc in [
                (lambda: queue.capture("New msg"), "capture"),
                (lambda: queue.recover(), "recover"),
                (lambda: queue.claim_due(), "claim_due"),
                (lambda: queue.complete(msg_id, "fake_token"), "complete"),
                (lambda: queue.fail(msg_id, "fake_token"), "fail"),
            ]:
                with self.assertRaises(DeferredQueueError, msg=desc) as ctx:
                    call()
                self.assertEqual(str(ctx.exception), "Queue storage unavailable")
                with store._connect() as conn:
                    self.assertFalse(conn.in_transaction, f"Dangling transaction after {desc}")

            # Restore authorizer and verify queue/source state preserved
            with store._connect() as conn:
                conn.set_authorizer(None)

            st_after = queue.status()
            self.assertEqual(st_after["pending_count"], 1)
            self.clock.advance(150)
            claim = queue.claim_due()
            self.assertIsNotNone(claim)
            self.assertEqual(claim["message_id"], msg_id)
            self.assertTrue(queue.complete(msg_id, claim["claim_token"]))
        finally:
            with store._connect() as conn:
                conn.set_authorizer(None)
            store.close()

    def test_connection_acquisition_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "queue_store.db")
            store = MemoryStore(db_path)
            try:
                queue = DeferredMemoryQueue(
                    store, self.owner_id, self.character_id, clock=self.clock
                )
                msg_id = queue.capture("Persisted message")

                # Point store path at directory to trigger acquisition error
                store.db_path = tmp_dir

                for call in [
                    lambda: queue.capture("Another message"),
                    lambda: queue.recover(),
                    lambda: queue.claim_due(),
                    lambda: queue.complete(msg_id, "fake_token"),
                    lambda: queue.fail(msg_id, "fake_token"),
                    lambda: queue.status(),
                    lambda: DeferredMemoryQueue(
                        store, "u2", "c2", clock=self.clock
                    ),
                ]:
                    with self.assertRaises(DeferredQueueError) as ctx:
                        call()
                    self.assertEqual(str(ctx.exception), "Queue storage unavailable")

                # Restore path and verify status works
                store.db_path = db_path
                st = queue.status()
                self.assertEqual(st["pending_count"], 1)
            finally:
                store.db_path = db_path
                store.close()

    def test_validation_and_caller_transaction_preserved(self) -> None:
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store, self.owner_id, self.character_id, clock=self.clock
            )
            # Validation ValueError not wrapped in DeferredQueueError
            with self.assertRaises(ValueError):
                queue.capture("")
            with self.assertRaises(ValueError):
                queue.claim_due(lease_seconds=-10)
            with self.assertRaises(ValueError):
                queue.fail("msg", "tok", error_category="invalid_category")
            with self.assertRaises(ValueError):
                DeferredMemoryQueue(store, "", self.character_id)

            # Active caller transaction raises specific error and remains unmodified
            with store._connect() as conn:
                store._begin_immediate(conn)
                self.assertTrue(conn.in_transaction)
                with self.assertRaises(DeferredQueueError) as ctx:
                    queue.capture("Should be rejected")
                self.assertEqual(str(ctx.exception), "Caller transaction already active")
                self.assertTrue(conn.in_transaction)
                conn.execute("ROLLBACK")
                self.assertFalse(conn.in_transaction)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
