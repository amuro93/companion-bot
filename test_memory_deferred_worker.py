"""Tests for DeferredMemoryWorker (TASK A-5b)."""

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from typing import Any

from memory_core import MemoryStore
from memory_deferred_queue import DeferredMemoryQueue, DeferredQueueError
from memory_deferred_worker import DeferredMemoryWorker


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class TestDeferredMemoryWorker(unittest.TestCase):
    def setUpself(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.store = MemoryStore(self.db_path)
        self.clock = FakeClock()
        self.owner = "usr-worker-1"
        self.char = "chr-worker-1"
        self.queue = DeferredMemoryQueue(
            self.store, self.owner, self.char, idle_seconds=180, clock=self.clock
        )
        self.active_workers: list[DeferredMemoryWorker] = []

    def make_worker(
        self,
        processor: Any = None,
        queue: Any = None,
        poll: float = 5,
        lease: float = 300,
        retry: float = 30,
    ) -> DeferredMemoryWorker:
        proc = processor if processor is not None else (lambda c: {"status": "ok"})
        q = queue if queue is not None else self.queue
        w = DeferredMemoryWorker(q, proc, poll_seconds=poll, lease_seconds=lease, retry_seconds=retry)
        self.active_workers.append(w)
        return w

    def tearDown(self) -> None:
        for w in self.active_workers:
            self.assertTrue(w.stop(timeout=3.0), "test worker must stop before store cleanup")
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def setUp(self) -> None:
        self.setUpself()

    def test_constructor_and_validation(self) -> None:
        w = self.make_worker()
        self.assertFalse(w.running)
        st = w.status()
        self.assertFalse(st["running"])
        self.assertIsNone(st["last_run_status"])
        self.assertEqual(st["pending_count"], 0)

        with self.assertRaises(ValueError):
            DeferredMemoryWorker(None, lambda c: {"status": "ok"})
        with self.assertRaises(ValueError):
            DeferredMemoryWorker(object(), lambda c: {"status": "ok"})
        with self.assertRaises(ValueError):
            DeferredMemoryWorker(self.queue, None)

        for invalid in (-1, 0, 86401, True, False, float("nan"), float("inf"), "5"):
            with self.assertRaises(ValueError):
                DeferredMemoryWorker(self.queue, lambda c: {"status": "ok"}, poll_seconds=invalid)
            with self.assertRaises(ValueError):
                DeferredMemoryWorker(self.queue, lambda c: {"status": "ok"}, lease_seconds=invalid)
        for invalid in (-1, 86401, True, False, float("nan"), float("inf"), "0"):
            with self.assertRaises(ValueError):
                DeferredMemoryWorker(self.queue, lambda c: {"status": "ok"}, retry_seconds=invalid)

        with self.assertRaises(ValueError):
            w.stop(timeout=-1)
        with self.assertRaises(ValueError):
            w.stop(timeout=True)

    def test_capture_and_idle_postponement(self) -> None:
        called: list[str] = []
        w = self.make_worker(processor=lambda c: called.append(c) or {"status": "ok"})

        msg1 = w.capture("我喜歡看星星。")
        self.assertTrue(msg1.startswith("msg-") or len(msg1) > 0)
        self.assertEqual(len(called), 0)
        self.assertEqual(w.status()["pending_count"], 1)

        self.clock.advance(100)
        res = w.process_once()
        self.assertEqual(res, {"status": "idle"})
        self.assertEqual(len(called), 0)

        w.capture("我每天都運動。")
        self.clock.advance(100)
        self.assertEqual(w.process_once(), {"status": "idle"})
        self.assertEqual(len(called), 0)

        self.clock.advance(181)
        res = w.process_once()
        self.assertEqual(res, {"status": "ok"})
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0], "我喜歡看星星。")

    def test_processor_outcomes_and_error_categories(self) -> None:
        def test_outcome(proc_res, expected_run_status, expected_error_cat=None):
            case_queue = DeferredMemoryQueue(
                self.store, f"outcome-{len(self.active_workers)}", self.char,
                idle_seconds=180, clock=self.clock,
            )
            w = self.make_worker(processor=proc_res, queue=case_queue)
            w.capture("我最喜歡吃蘋果。")
            self.clock.advance(200)
            res = w.process_once()
            self.assertEqual(res, expected_run_status)
            st = w.status()
            self.assertEqual(st["last_run_status"], expected_run_status)
            if expected_error_cat:
                self.assertEqual(st["last_error_category"], expected_error_cat)

        test_outcome(lambda c: {"status": "ok"}, {"status": "ok"})
        test_outcome(
            lambda c: {"status": "failed", "error_category": "index_failed"},
            {"status": "retry", "error_category": "index_failed"},
            "index_failed",
        )
        test_outcome(
            lambda c: {"status": "failed"},
            {"status": "retry", "error_category": "processor_failed"},
            "processor_failed",
        )
        test_outcome(
            lambda c: {"status": "failed", "error_category": "other"},
            {"status": "retry", "error_category": "processor_failed"},
            "processor_failed",
        )

        def raise_err(c):
            raise RuntimeError("secret-leak-text")

        test_outcome(
            raise_err,
            {"status": "retry", "error_category": "processor_failed"},
            "processor_failed",
        )

        for malformed in (None, False, {}, {"status": "unknown"}, "wrong_type"):
            test_outcome(
                lambda c, m=malformed: m,
                {"status": "retry", "error_category": "invalid_result"},
                "invalid_result",
            )

    def test_stale_ack_and_fail_never_ok(self) -> None:
        def stale_ok_proc(content: str) -> dict[str, Any]:
            self.clock.advance(301)
            other_claim = self.queue.claim_due(lease_seconds=300)
            self.assertIsNotNone(other_claim)
            return {"status": "ok"}

        w1 = self.make_worker(processor=stale_ok_proc, lease=300)
        w1.capture("我喜歡喝茶。")
        self.clock.advance(200)

        res1 = w1.process_once()
        self.assertEqual(res1, {"status": "stale"})

        fail_queue = DeferredMemoryQueue(
            self.store, "stale-fail-owner", self.char, idle_seconds=180, clock=self.clock
        )

        def stale_fail_proc(content: str) -> dict[str, Any]:
            self.clock.advance(301)
            other_claim = fail_queue.claim_due(lease_seconds=300)
            self.assertIsNotNone(other_claim)
            return {"status": "failed"}

        w2 = self.make_worker(processor=stale_fail_proc, lease=300, queue=fail_queue)
        w2.capture("我喜歡喝咖啡。")
        self.clock.advance(200)

        res2 = w2.process_once()
        self.assertEqual(res2, {"status": "stale"})

    def test_restart_persists_pending_no_replay_and_excluded(self) -> None:
        w1 = self.make_worker()
        msg_id = w1.capture("我以前學過鋼琴。")
        self.clock.advance(200)

        w2 = self.make_worker(processor=lambda c: {"status": "ok"})
        self.assertFalse(w2.running)
        res = w2.process_once()
        self.assertEqual(res, {"status": "ok"})
        self.assertEqual(w2.status()["done_count"], 1)

        self.assertEqual(w2.process_once(), {"status": "idle"})
        self.assertEqual(w2.status()["done_count"], 1)

        with self.store._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, owner_id, character_id, external_id, role,
                    content, occurred_at, created_at, payload_hash, excluded_from_memory
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "msg-ex-1", self.owner, self.char, "tg-ex-1", "user",
                    "我不想被記住。", "2026-11-12T00:00:00Z", "2026-11-12T00:00:00Z", "hash-1"
                ),
            )
            conn.commit()

        self.clock.advance(200)
        called: list[str] = []
        w3 = self.make_worker(processor=lambda c: called.append(c) or {"status": "ok"})
        self.assertEqual(w3.process_once(), {"status": "idle"})
        self.assertEqual(len(called), 0)
        self.assertEqual(w3.status()["excluded_count"], 1)

    def test_start_idempotent_concurrent_busy_daemon_and_stop(self) -> None:
        proc_entered = threading.Event()
        proc_release = threading.Event()

        def slow_proc(content: str) -> dict[str, Any]:
            proc_entered.set()
            proc_release.wait(timeout=2.0)
            return {"status": "ok"}

        w = self.make_worker(processor=slow_proc, poll=0.05)
        self.assertTrue(w.start())
        self.assertFalse(w.start())
        self.assertTrue(w.running)
        self.assertTrue(w._thread.daemon)

        w.capture("我通常七點起床。")
        self.clock.advance(200)

        self.assertTrue(proc_entered.wait(timeout=2.0))
        busy_res = w.process_once()
        self.assertEqual(busy_res, {"status": "busy"})

        proc_release.set()
        self.assertTrue(w.stop(timeout=2.0))
        self.assertFalse(w.running)
        self.assertEqual(w.status()["done_count"], 1)

    def test_blocked_processor_bounded_stop_and_capture_nonblocking(self) -> None:
        proc_gate = threading.Event()
        proc_started = threading.Event()

        def blocking_proc(content: str) -> dict[str, Any]:
            proc_started.set()
            proc_gate.wait(timeout=2.0)
            return {"status": "ok"}

        w = self.make_worker(processor=blocking_proc, poll=0.05)
        w.capture("我平常都吃素。")
        self.clock.advance(200)

        self.assertTrue(w.start())
        self.assertTrue(proc_started.wait(timeout=2.0))

        t0 = time.time()
        msg2 = w.capture("我也有去跑步。")
        self.assertLess(time.time() - t0, 0.5)
        self.assertTrue(msg2.startswith("msg-") or len(msg2) > 0)

        stopped = w.stop(timeout=0.05)
        self.assertFalse(stopped)
        self.assertTrue(w.running)

        proc_gate.set()
        self.assertTrue(w.stop(timeout=2.0))
        self.assertFalse(w.running)

    def test_queue_storage_error_sanitized(self) -> None:
        w = self.make_worker()
        w.capture("我有三個小孩。")
        self.clock.advance(200)

        with self.store._connect() as conn:
            conn.execute("DROP TABLE deferred_memory_jobs")
            conn.commit()

        res = w.process_once()
        self.assertEqual(res, {"status": "error", "error_category": "storage_failed"})
        self.assertNotIn("no such table", str(res))
        with self.assertRaises(DeferredQueueError):
            w.status()

    def test_stop_before_start_and_restart_lifecycle(self) -> None:
        w = self.make_worker()
        self.assertTrue(w.stop())
        self.assertFalse(w.running)

        w.capture("我喜歡夏天。")
        self.assertEqual(w.status()["pending_count"], 1)

        self.assertTrue(w.start())
        self.assertTrue(w.running)
        self.assertTrue(w.stop(timeout=1.0))
        self.assertFalse(w.running)

        self.assertEqual(w.status()["pending_count"], 1)
        self.assertEqual(w.status()["done_count"], 0)

        self.clock.advance(200)
        self.assertTrue(w.start())
        stop_wait = threading.Event()

        def wait_done():
            for _ in range(50):
                if w.status()["done_count"] == 1:
                    stop_wait.set()
                    break
                time.sleep(0.02)

        t = threading.Thread(target=wait_done)
        t.start()
        self.assertTrue(stop_wait.wait(timeout=2.0))
        t.join(timeout=1.0)

        self.assertTrue(w.stop(timeout=1.0))
        self.assertFalse(w.running)
        self.assertEqual(w.status()["done_count"], 1)


if __name__ == "__main__":
    unittest.main()
