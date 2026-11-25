"""Parent-added regressions: stop must include public synchronous work."""
from __future__ import annotations

import os
import tempfile
import threading
import unittest

from memory_core import MemoryStore
from memory_deferred_queue import DeferredMemoryQueue
from memory_deferred_worker import DeferredMemoryWorker


class TestPublicProcessStopSafety(unittest.TestCase):
    def _check_inflight(self, start_background: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(os.path.join(directory, "fictional.sqlite3"))
            now = [1000.0]
            queue = DeferredMemoryQueue(
                store, "fictional-owner", "fictional-character", clock=lambda: now[0]
            )
            entered = threading.Event()
            release = threading.Event()
            results = []
            errors = []

            def processor(content):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("fictional test gate timed out")
                return {"status": "ok"}

            worker = DeferredMemoryWorker(queue, processor, poll_seconds=0.02)

            def synchronous_call():
                try:
                    results.append(worker.process_once())
                except Exception as exc:
                    errors.append(type(exc).__name__)

            hook = threading.Thread(target=synchronous_call)
            try:
                worker.capture("Fictional pending source")
                now[0] += 181
                hook.start()
                self.assertTrue(entered.wait(2))
                if start_background:
                    self.assertTrue(worker.start())
                self.assertFalse(worker.stop(timeout=0.01))
                self.assertTrue(hook.is_alive())
                self.assertEqual(queue.status()["processing_count"], 1)
                release.set()
                hook.join(2)
                self.assertFalse(hook.is_alive())
                self.assertTrue(worker.stop(timeout=2))
                self.assertEqual(errors, [])
                self.assertEqual(results, [{"status": "ok"}])
                self.assertEqual(queue.status()["done_count"], 1)
                self.assertTrue(worker.stop(timeout=0))
            finally:
                release.set()
                if hook.ident is not None:
                    hook.join(3)
                self.assertFalse(hook.is_alive())
                self.assertTrue(worker.stop(timeout=3))
                store.close()

    def test_manual_process_blocks_safe_stop_without_background(self):
        self._check_inflight(False)

    def test_manual_process_blocks_safe_stop_after_background_stops(self):
        self._check_inflight(True)


if __name__ == "__main__":
    unittest.main()
