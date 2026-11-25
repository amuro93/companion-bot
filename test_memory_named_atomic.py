"""Regression tests for named-key SQLite transaction boundaries (synthetic only)."""
import pathlib
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from memory_core import MemoryStore


class TestNamedKeyAtomicity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(pathlib.Path(self.temp.name) / "fictional.sqlite3")
        self.store = MemoryStore(self.db)
        self.owner = "fiction-owner"
        self.character = "fiction-companion"
        self.key = "約定與計畫/家務分工"

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def write(self, store, label, text, origin):
        mid = store.append_message(self.owner, self.character, "msg-" + label, "user", text)
        return store.save_keyed_memory(
            self.owner, self.character, "mem-" + label, "promise", text,
            [mid], self.key, source=origin,
        )

    def test_one_connection_and_same_fact_correction(self):
        original = self.store._create_connection
        opened = []

        def counted():
            opened.append(1)
            return original()

        with patch.object(self.store, "_create_connection", side_effect=counted):
            self.write(self.store, "a", "虛構課程時間一", "auto")
        self.assertEqual(len(opened), 2)  # append + one named-key transaction
        opened.clear()
        with patch.object(self.store, "_create_connection", side_effect=counted):
            out = self.write(self.store, "b", "虛構課程時間二", "auto")
        self.assertEqual(len(opened), 2)
        self.assertEqual(out["action"], "correct")
        rows = self.store.list_memories(self.owner, self.character)["items"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "虛構課程時間二")

    def test_manual_and_worker_compete(self):
        manual, auto = MemoryStore(self.db), MemoryStore(self.db)
        reached, release = threading.Event(), threading.Event()
        outcomes, errors = [], []
        original = manual._queue_job

        def pause(conn, *args, **kwargs):
            result = original(conn, *args, **kwargs)
            reached.set()
            if not release.wait(3):
                raise RuntimeError("synthetic lock hold timed out")
            return result

        def manual_writer():
            try:
                self.write(manual, "man", "虛構手動安排", "manual")
            except Exception as exc:
                errors.append(exc)

        def auto_writer():
            try:
                outcomes.append(self.write(auto, "aut", "虛構自動安排", "auto"))
            except Exception as exc:
                errors.append(exc)

        try:
            with patch.object(manual, "_queue_job", side_effect=pause):
                t1 = threading.Thread(target=manual_writer)
                t1.start()
                self.assertTrue(reached.wait(2), "manual write not reached")
                t2 = threading.Thread(target=auto_writer)
                t2.start()
                time.sleep(0.08)
                self.assertTrue(t2.is_alive(), "second writer did not wait on SQLite lock")
                release.set()
                t1.join(4)
                t2.join(4)
                self.assertFalse(t1.is_alive() or t2.is_alive())
            self.assertFalse(errors, errors)
            self.assertEqual([r["action"] for r in outcomes], ["skipped"])
            rows = self.store.list_memories(self.owner, self.character)["items"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "manual")
        finally:
            release.set()
            manual.close()
            auto.close()

    def test_queue_error_rolls_back(self):
        with patch.object(self.store, "_queue_job", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                self.write(self.store, "fail", "虛構新課程", "auto")
        self.assertEqual(self.store.list_memories(self.owner, self.character)["total"], 0)
        first = self.write(self.store, "first", "虛構課程一", "auto")
        with patch.object(self.store, "_queue_job", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                self.write(self.store, "second", "虛構課程二", "auto")
        rows = self.store.list_memories(self.owner, self.character)["items"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["memory_id"], first["memory_id"])

    def test_invalid_internal_connection_rejected(self):
        conn = self.store._create_connection()
        try:
            mid = self.store.append_message(self.owner, self.character, "msg-invalid", "user", "虛構內容")
            with self.assertRaises(ValueError):
                self.store.save_memory(
                    self.owner, self.character, "mem-invalid", "promise", "虛構內容",
                    [mid], meta_key=self.key, _transaction_conn=conn,
                )
            with self.assertRaises(ValueError):
                self.store.correct_memory(
                    self.owner, self.character, "missing", "mem-fix", "虛構新內容",
                    [mid], _transaction_conn=conn,
                )
        finally:
            conn.close()
        self.assertEqual(self.store.list_memories(self.owner, self.character)["total"], 0)

    def test_keyboard_interrupt_rolls_back_memory_db(self):
        store = MemoryStore(":memory:")
        try:
            mid = store.append_message(self.owner, self.character, "msg-int", "user", "虛構內容")
            with patch.object(store, "_queue_job", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    store.save_keyed_memory(
                        self.owner, self.character, "mem-int", "promise", "虛構內容",
                        [mid], self.key,
                    )
            self.assertEqual(store.list_memories(self.owner, self.character)["total"], 0)
            self.assertFalse(store._memory_conn.in_transaction)
        finally:
            store.close()

    def test_early_skip_preserves_legacy_source_behavior(self):
        self.write(self.store, "manual", "虛構手動安排", "manual")
        mid = self.store.append_message(
            self.owner, self.character, "msg-excluded", "user", "虛構自動安排"
        )
        with self.store._connect() as conn:
            conn.execute("UPDATE messages SET excluded_from_memory = 1 WHERE message_id = ?", (mid,))
        out = self.store.save_keyed_memory(
            self.owner, self.character, "mem-excluded", "promise", "虛構自動安排",
            [mid], self.key, source="auto",
        )
        self.assertEqual(out["action"], "skipped")
        self.assertEqual(self.store.list_memories(self.owner, self.character)["total"], 1)

    def test_retry_is_idempotent(self):
        self.write(self.store, "a", "虛構課程一", "auto")
        second = self.write(self.store, "b", "虛構課程二", "auto")
        replay = self.write(self.store, "b", "虛構課程二", "auto")
        self.assertEqual(replay["memory_id"], second["memory_id"])
        self.assertEqual(self.store.list_memories(self.owner, self.character)["total"], 1)


if __name__ == "__main__":
    unittest.main()
