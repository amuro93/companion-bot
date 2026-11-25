"""Unit tests for A-2b source-message exclusion after explicit memory deletion.

Standard-library-only test cases with temporary fictional databases.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from memory_core import MemoryStore
from memory_auto import record_user_message_keyed
import memory_trigger as mt


class TestMemorySourceExclusion(unittest.TestCase):
    """Test suite covering source-message exclusion policy and edge cases."""

    def test_all_revision_sources_flagged_and_reopen_flag1(self) -> None:
        """All revisions in a deleted lineage have sources excluded=1; persists on reopen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "lineage_reopen.db")
            store = MemoryStore(db_path)
            try:
                owner = "owner-1"
                char = "char-1"
                msg1 = store.append_message(owner, char, "ext-msg-1", "user", "I like green tea.")
                mem1 = store.save_memory(
                    owner, char, "ext-mem-1", "preference", "User likes green tea.", [msg1], meta_key="喜好/茶"
                )
                msg2 = store.append_message(owner, char, "ext-msg-2", "user", "Actually I prefer black tea.")
                mem2 = store.correct_memory(
                    owner, char, mem1, "ext-mem-2", "User prefers black tea.", [msg2], meta_key="喜好/茶"
                )

                self.assertFalse(store.is_message_excluded(owner, char, msg1))
                self.assertFalse(store.is_message_excluded(owner, char, msg2))

                deleted = store.delete_memory(owner, char, mem2)
                self.assertTrue(deleted)

                self.assertTrue(store.is_message_excluded(owner, char, msg1))
                self.assertTrue(store.is_message_excluded(owner, char, msg2))
            finally:
                store.close()

            # Reopen file database and verify flag persistence
            store2 = MemoryStore(db_path)
            try:
                self.assertTrue(store2.is_message_excluded(owner, char, msg1))
                self.assertTrue(store2.is_message_excluded(owner, char, msg2))
            finally:
                store2.close()

    def test_raw_content_and_rows_preserved(self) -> None:
        """Physical message rows and raw text remain intact after memory deletion."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            raw_text = "Verbatim raw message content"
            msg_id = store.append_message(owner, char, "ext-raw-1", "user", raw_text)
            mem_id = store.save_memory(owner, char, "ext-mem-raw", "fact", "Derived fact", [msg_id])

            store.delete_memory(owner, char, mem_id)

            with store._connect() as conn:
                row = conn.execute(
                    "SELECT content, excluded_from_memory FROM messages WHERE message_id = ?",
                    (msg_id,),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(str(row["content"]), raw_text)
                self.assertEqual(int(row["excluded_from_memory"]), 1)
        finally:
            store.close()

    def test_unrelated_existing_memory_same_source_remains(self) -> None:
        """Unrelated memories referencing the same source message remain active."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            msg_id = store.append_message(owner, char, "ext-multi", "user", "I live in Taipei and work in tech.")
            mem_a = store.save_memory(owner, char, "ext-mem-a", "fact", "User lives in Taipei", [msg_id])
            mem_b = store.save_memory(owner, char, "ext-mem-b", "fact", "User works in tech", [msg_id])

            deleted_a = store.delete_memory(owner, char, mem_a)
            self.assertTrue(deleted_a)

            self.assertIsNone(store.get_memory(owner, char, mem_a))
            active_b = store.get_memory(owner, char, mem_b)
            self.assertIsNotNone(active_b)
            self.assertEqual(active_b["text"], "User works in tech")
            self.assertTrue(store.is_message_excluded(owner, char, msg_id))
        finally:
            store.close()

    def test_newhash_paraphrase_cannot_reuse_source(self) -> None:
        """New save/re-key with new external hash or paraphrase fails on excluded source."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            msg_id = store.append_message(owner, char, "ext-orig", "user", "I love spicy food.")
            mem_id = store.save_memory(owner, char, "ext-mem-orig", "preference", "User loves spicy food.", [msg_id])
            store.delete_memory(owner, char, mem_id)

            with self.assertRaises(ValueError) as ctx:
                store.save_memory(
                    owner, char, "ext-mem-paraphrase", "preference", "User likes spicy hot meals.", [msg_id]
                )
            err_msg = str(ctx.exception)
            self.assertIn("excluded", err_msg.lower())
            self.assertNotIn("I love spicy food", err_msg)
            self.assertNotIn("User likes spicy hot meals", err_msg)
        finally:
            store.close()

    def test_direct_correct_rejects_excluded_source(self) -> None:
        """Direct correction targeting active memory cannot link an excluded source."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            msg_active = store.append_message(owner, char, "ext-active", "user", "Active message source.")
            mem_active = store.save_memory(owner, char, "ext-mem-act", "fact", "Active memory", [msg_active])

            msg_deleted = store.append_message(owner, char, "ext-del", "user", "Deleted message source.")
            mem_deleted = store.save_memory(owner, char, "ext-mem-del", "fact", "To delete", [msg_deleted])
            store.delete_memory(owner, char, mem_deleted)

            with self.assertRaises(ValueError) as ctx:
                store.correct_memory(
                    owner, char, mem_active, "ext-corr-fail", "Corrected memory", [msg_deleted]
                )
            self.assertIn("excluded", str(ctx.exception).lower())
        finally:
            store.close()

    def test_recorder_spy_keyer_call_count_exactly_zero(self) -> None:
        """Deleted source replay triggers immediate tombstone without model/keyer calls."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            content = "我最喜歡喝綠茶"

            class SpyKeyer:
                def __init__(self) -> None:
                    self.call_count = 0

                def decide(self, sentence: str) -> dict:
                    self.call_count += 1
                    return {
                        "key": "喜好/茶",
                        "how": "model",
                        "needs_review": False,
                        "trace": [{"stage": "decision", "how": "model"}],
                    }

            spy = SpyKeyer()
            res1 = record_user_message_keyed(store, owner, char, content, keyer=spy)
            self.assertGreater(spy.call_count, 0)
            self.assertTrue(res1["memory_ids"])
            mem_id = res1["memory_ids"][0]

            store.delete_memory(owner, char, mem_id)
            self.assertTrue(store.is_message_excluded(owner, char, res1["message_id"]))

            spy.call_count = 0
            res2 = record_user_message_keyed(store, owner, char, content, keyer=spy)
            self.assertEqual(spy.call_count, 0)
            self.assertEqual(res2["memory_ids"], [])
            self.assertEqual(
                res2["results"],
                [{"key": None, "action": "tombstoned", "reason": "source_excluded"}],
            )
            self.assertEqual(
                res2["trace"],
                [{"stage": "source_exclusion", "action": "tombstoned", "reason": "source_excluded"}],
            )
        finally:
            store.close()

    def test_gate_case_also_skipped(self) -> None:
        """Gate decision path is also skipped immediately on excluded source replay."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            content = "我最喜歡喝綠茶"

            class SpyGate:
                def __init__(self) -> None:
                    self.call_count = 0

                def remember(self, sentence: str) -> bool:
                    self.call_count += 1
                    return True

            spy = SpyGate()
            res1 = record_user_message_keyed(store, owner, char, content, gate=spy)
            self.assertGreater(spy.call_count, 0)
            self.assertTrue(res1["memory_ids"])

            store.delete_memory(owner, char, res1["memory_ids"][0])

            spy.call_count = 0
            res2 = record_user_message_keyed(store, owner, char, content, gate=spy)
            self.assertEqual(spy.call_count, 0)
            self.assertEqual(res2["results"][0]["action"], "tombstoned")
            self.assertEqual(res2["results"][0]["reason"], "source_excluded")
        finally:
            store.close()

    def test_wrong_owner_and_reader_delete_denied_without_flags(self) -> None:
        """Unauthorized delete attempts raise PermissionError without mutating excluded flag."""
        store = MemoryStore(":memory:")
        try:
            owner1 = "owner-1"
            owner2 = "owner-2"
            char1 = "char-1"
            char2 = "char-2"
            msg1 = store.append_message(owner1, char1, "ext-sec-1", "user", "Shared secret")
            mem1 = store.save_memory(
                owner1, char1, "ext-mem-sec", "fact", "Shared fact", [msg1], visible_to=[char1, char2]
            )

            with self.assertRaises(PermissionError):
                store.delete_memory(owner2, char1, mem1)

            with self.assertRaises(PermissionError):
                store.delete_memory(owner1, char2, mem1)

            self.assertFalse(store.is_message_excluded(owner1, char1, msg1))
        finally:
            store.close()

    def test_missing_and_wrongscope_check_false(self) -> None:
        """is_message_excluded returns False for missing ID or out-of-scope owner/character."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            msg_id = store.append_message(owner, char, "ext-scope-1", "user", "Scope message")

            self.assertFalse(store.is_message_excluded(owner, char, "missing-msg-id"))
            self.assertFalse(store.is_message_excluded("other-owner", char, msg_id))
            self.assertFalse(store.is_message_excluded(owner, "other-char", msg_id))

            with self.assertRaises(ValueError):
                store.is_message_excluded("", char, msg_id)
        finally:
            store.close()

    def test_queue_failure_atomic_rollback(self) -> None:
        """Failure during index job queueing rolls back the entire deletion and flag update."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            msg_id = store.append_message(owner, char, "ext-atomic-1", "user", "Rollback message")
            mem_id = store.save_memory(owner, char, "ext-mem-atomic", "fact", "Atomic memory", [msg_id])

            orig_queue = store._queue_job

            def failing_queue(*args, **kwargs):
                raise RuntimeError("simulated index queue failure")

            store._queue_job = failing_queue

            with self.assertRaises(RuntimeError):
                store.delete_memory(owner, char, mem_id)

            store._queue_job = orig_queue

            self.assertFalse(store.is_message_excluded(owner, char, msg_id))
            active = store.get_memory(owner, char, mem_id)
            self.assertIsNotNone(active)
            self.assertEqual(active["text"], "Atomic memory")
        finally:
            store.close()

    def test_old_db_migrates_default_zero(self) -> None:
        """Older schema databases migrate idempotently with excluded_from_memory defaulting to 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "old_schema.db")
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE messages (
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
                """
            )
            conn.execute(
                """
                INSERT INTO messages VALUES (
                    1, 'legacy-msg-1', 'o1', 'c1', 'legacy-ext-1', 'user', 'Legacy content',
                    '2026-01-01T00:00:00.000000Z', '2026-01-01T00:00:00.000000Z', 'legacy_hash'
                );
                """
            )
            conn.commit()
            conn.close()

            store = MemoryStore(db_path)
            try:
                self.assertFalse(store.is_message_excluded("o1", "c1", "legacy-msg-1"))
                with store._connect() as c:
                    row = c.execute(
                        "SELECT excluded_from_memory FROM messages WHERE message_id = 'legacy-msg-1'"
                    ).fetchone()
                    self.assertIsNotNone(row)
                    self.assertEqual(int(row[0]), 0)
            finally:
                store.close()

    def test_separate_newly_appended_source_with_same_words_not_blocked(self) -> None:
        """A genuinely different source message identity with same text is not blanket blocked."""
        store = MemoryStore(":memory:")
        try:
            owner = "owner-1"
            char = "char-1"
            msg1 = store.append_message(owner, char, "ext-user-1", "user", "I like green tea.")
            mem1 = store.save_memory(owner, char, "ext-mem-1", "preference", "User likes green tea.", [msg1])
            store.delete_memory(owner, char, mem1)
            self.assertTrue(store.is_message_excluded(owner, char, msg1))

            # A distinct message identity (different external_id)
            msg2 = store.append_message(owner, char, "ext-user-2-distinct", "user", "I like green tea.")
            self.assertNotEqual(msg1, msg2)
            self.assertFalse(store.is_message_excluded(owner, char, msg2))

            # Saving memory from the new distinct source message succeeds
            mem2 = store.save_memory(
                owner, char, "ext-mem-2-distinct", "preference", "User likes green tea.", [msg2]
            )
            self.assertIsNotNone(mem2)
            active2 = store.get_memory(owner, char, mem2)
            self.assertIsNotNone(active2)
            self.assertEqual(active2["source_ids"], [msg2])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
