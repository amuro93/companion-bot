"""Tests for stop/start replacement race condition in DeferredMemoryWorker."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from memory_core import MemoryStore
from memory_deferred_queue import DeferredMemoryQueue
from memory_deferred_worker import DeferredMemoryWorker


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class TestDeferredMemoryWorkerStopRace(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_stop_race.db")
        self.store = MemoryStore(self.db_path)
        self.clock = FakeClock()
        self.queue = DeferredMemoryQueue(
            self.store, "usr-stop-race", "chr-stop-race", idle_seconds=180, clock=self.clock
        )
        self.active_workers: list[DeferredMemoryWorker] = []

    def make_worker(self) -> DeferredMemoryWorker:
        w = DeferredMemoryWorker(
            self.queue,
            lambda content: {"status": "ok"},
            poll_seconds=0.05,
            lease_seconds=300,
            retry_seconds=30,
        )
        self.active_workers.append(w)
        return w

    def tearDown(self) -> None:
        for w in self.active_workers:
            w.stop(timeout=2.0)
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stop_returns_false_when_replacement_started_during_stop(self) -> None:
        worker = self.make_worker()
        self.assertTrue(worker.start())
        self.assertTrue(worker.running)

        old_thread = worker._thread
        self.assertIsNotNone(old_thread)

        real_join = old_thread.join
        replacement_started: list[bool] = []

        def wrapped_join(timeout: float | None = None) -> None:
            real_join(timeout=timeout)
            replacement_started.append(worker.start())

        old_thread.join = wrapped_join  # type: ignore[assignment]

        stopped = worker.stop(timeout=2.0)
        self.assertEqual(replacement_started, [True])
        self.assertFalse(stopped)
        self.assertTrue(worker.running)

        # Cleanly stop live replacement worker before teardown
        self.assertTrue(worker.stop(timeout=2.0))
        self.assertFalse(worker.running)

    def test_ordinary_no_replacement_stop_and_repeated_stop(self) -> None:
        worker = self.make_worker()
        self.assertTrue(worker.start())
        self.assertTrue(worker.running)

        self.assertTrue(worker.stop(timeout=2.0))
        self.assertFalse(worker.running)

        # Repeated stop regression check
        self.assertTrue(worker.stop(timeout=2.0))
        self.assertFalse(worker.running)


if __name__ == "__main__":
    unittest.main()
