"""Tests for nondecreasing queue activity timestamp under races and clock skew."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest

from memory_core import MemoryStore
from memory_deferred_queue import DeferredMemoryQueue
from memory_trigger import message_external_id


class TestDeferredQueueActivity(unittest.TestCase):
    def test_two_store_race_preserves_latest_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "race.db")
            store_a = MemoryStore(db_path)
            store_b = MemoryStore(db_path)
            entered = threading.Event()
            release = threading.Event()
            pause_a = [False]
            time_b = [1000.0]
            errors: list[Exception] = []

            def clock_a() -> float:
                if pause_a[0]:
                    entered.set()
                    release.wait(timeout=5.0)
                    return 1100.0
                return 1000.0

            def run_a() -> None:
                try:
                    q_a.capture("msg A")
                except Exception as exc:
                    errors.append(exc)

            q_a = DeferredMemoryQueue(
                store_a, "u1", "c1", idle_seconds=180, clock=clock_a
            )
            q_b = DeferredMemoryQueue(
                store_b, "u1", "c1", idle_seconds=180, clock=lambda: time_b[0]
            )

            t = None
            try:
                pause_a[0] = True
                t = threading.Thread(target=run_a)
                t.start()

                self.assertTrue(entered.wait(timeout=5.0))
                time_b[0] = 1200.0
                q_b.capture("msg B")

                release.set()
                t.join(timeout=5.0)
                self.assertFalse(t.is_alive())
                self.assertEqual(errors, [])

                st = q_b.status()
                self.assertEqual(st["last_activity"], 1200.0)

                time_b[0] = 1280.0
                self.assertIsNone(q_b.claim_due(lease_seconds=300))

                time_b[0] = 1380.0
                claimed = q_b.claim_due(lease_seconds=300)
                self.assertIsNotNone(claimed)
            finally:
                release.set()
                if t is not None and t.is_alive():
                    t.join(timeout=5.0)
                store_a.close()
                store_b.close()

    def test_recover_and_status_nondecreasing_on_clock_rollback(self) -> None:
        store = MemoryStore(":memory:")
        try:
            curr_time = [1000.0]
            q = DeferredMemoryQueue(
                store, "u1", "c1", idle_seconds=180, clock=lambda: curr_time[0]
            )
            curr_time[0] = 1200.0
            q.capture("Initial message")
            self.assertEqual(q.status()["last_activity"], 1200.0)

            ext_id = message_external_id("Delayed message")
            store.append_message("u1", "c1", ext_id, "user", "Delayed message")

            curr_time[0] = 1100.0
            recovered = q.recover()
            self.assertEqual(recovered, 1)
            self.assertEqual(q.status()["last_activity"], 1200.0)

            curr_time[0] = 1050.0
            self.assertEqual(q.recover(), 0)
            self.assertEqual(q.status()["last_activity"], 1200.0)

            curr_time[0] = 900.0
            st = q.status()
            self.assertEqual(st["last_activity"], 1200.0)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
