"""Tests for MemoryStore.rollback_memory (P1a revision restore semantics)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from memory_core import MemoryStore


class TestMemoryRollback(unittest.TestCase):
    """Unit tests for MemoryStore.rollback_memory."""

    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.addCleanup(self.store.close)

        self.owner = "test-owner"
        self.char = "test-char"
        self.reader = "reader-char"

        # Seed initial source messages
        self.msg1 = self.store.append_message(
            self.owner, self.char, "msg-ext-1", "user", "Alice likes apples"
        )
        self.msg2 = self.store.append_message(
            self.owner, self.char, "msg-ext-2", "user", "Alice likes oranges"
        )
        self.msg3 = self.store.append_message(
            self.owner, self.char, "msg-ext-3", "user", "Alice likes bananas"
        )

    def test_restore_2version_chain_creates_third_row(self) -> None:
        """2-version chain rollback creates 3rd row with right root/parent/superseded links."""
        v1 = self.store.save_memory(
            self.owner,
            self.char,
            "mem-ext-1",
            "preference",
            "Alice prefers apples",
            [self.msg1],
            meta_key="pref/food",
            source="manual",
        )
        v2 = self.store.correct_memory(
            self.owner,
            self.char,
            v1,
            "mem-ext-2",
            "Alice prefers oranges",
            [self.msg2],
            meta_key="pref/food",
            source="auto",
        )

        res = self.store.rollback_memory(self.owner, self.char, v2)

        self.assertEqual(res["action"], "rollback")
        v3 = res["memory_id"]
        self.assertEqual(res["restored_from"], v1)
        self.assertEqual(res["superseded"], v2)
        self.assertNotEqual(v3, v2)
        self.assertNotEqual(v3, v1)

        with self.store._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, root_id, parent_id, superseded_by, text, meta_key, source, status
                FROM memories
                WHERE root_id = ?
                ORDER BY seq ASC
                """,
                (v1,),
            ).fetchall()

            self.assertEqual(len(rows), 3)

            r1, r2, r3 = rows[0], rows[1], rows[2]
            # Revision 1
            self.assertEqual(r1["memory_id"], v1)
            self.assertEqual(r1["root_id"], v1)
            self.assertIsNone(r1["parent_id"])
            self.assertEqual(r1["superseded_by"], v2)
            self.assertEqual(r1["status"], "superseded")

            # Revision 2 (target)
            self.assertEqual(r2["memory_id"], v2)
            self.assertEqual(r2["root_id"], v1)
            self.assertEqual(r2["parent_id"], v1)
            self.assertEqual(r2["superseded_by"], v3)
            self.assertEqual(r2["status"], "superseded")

            # Revision 3 (restored head)
            self.assertEqual(r3["memory_id"], v3)
            self.assertEqual(r3["root_id"], v1)
            self.assertEqual(r3["parent_id"], v2)
            self.assertIsNone(r3["superseded_by"])
            self.assertEqual(r3["status"], "active")
            self.assertEqual(r3["text"], "Alice prefers apples")
            self.assertEqual(r3["meta_key"], "pref/food")
            self.assertEqual(r3["source"], "manual")

    def test_sources_and_source_preserved(self) -> None:
        """Restores parent's text, source_ids, and source origin."""
        v1 = self.store.save_memory(
            self.owner,
            self.char,
            "ext-p1",
            "fact",
            "Parent text content",
            [self.msg1],
            source="manual",
        )
        v2 = self.store.correct_memory(
            self.owner,
            self.char,
            v1,
            "ext-p2",
            "Target modified text",
            [self.msg2],
            source="auto",
        )

        res = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res["memory_id"]

        mem = self.store.get_memory(self.owner, self.char, v3)
        self.assertIsNotNone(mem)
        self.assertEqual(mem["text"], "Parent text content")
        self.assertEqual(mem["source"], "manual")
        self.assertEqual(mem["source_ids"], [self.msg1])

    def test_current_acl_retained_not_old_acl(self) -> None:
        """New revision retains current target ACL without restoring historical revoked readers."""
        # v1 had visible_to = [char, reader_old]
        v1 = self.store.save_memory(
            self.owner,
            self.char,
            "acl-ext-1",
            "preference",
            "V1 text",
            [self.msg1],
            visible_to=["reader_old"],
        )
        # v2 created; suppose reader_old is revoked and reader_cur is added on v2
        v2 = self.store.correct_memory(
            self.owner,
            self.char,
            v1,
            "acl-ext-2",
            "V2 text",
            [self.msg2],
        )

        # Mutate v2 ACL to simulate revoked reader_old and added reader_cur
        with self.store._connect() as conn:
            conn.execute("DELETE FROM memory_acl WHERE memory_id = ?", (v2,))
            conn.execute(
                "INSERT INTO memory_acl (memory_id, owner_id, character_id) VALUES (?, ?, ?)",
                (v2, self.owner, "reader_cur"),
            )
            conn.execute(
                "INSERT INTO memory_acl (memory_id, owner_id, character_id) VALUES (?, ?, ?)",
                (v2, self.owner, self.char),
            )

        res = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res["memory_id"]

        with self.store._connect() as conn:
            acl_rows = conn.execute(
                "SELECT character_id FROM memory_acl WHERE memory_id = ? ORDER BY character_id ASC",
                (v3,),
            ).fetchall()
            v3_acl = [str(r["character_id"]) for r in acl_rows]

        self.assertIn("reader_cur", v3_acl)
        self.assertNotIn("reader_old", v3_acl)

    def test_one_active_head(self) -> None:
        """Exactly one active revision exists in the lineage at any time."""
        v1 = self.store.save_memory(
            self.owner, self.char, "act-1", "preference", "T1", [self.msg1]
        )
        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "act-2", "T2", [self.msg2]
        )
        res = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res["memory_id"]

        with self.store._connect() as conn:
            active_rows = conn.execute(
                "SELECT memory_id FROM memories WHERE root_id = ? AND status = 'active'",
                (v1,),
            ).fetchall()
            self.assertEqual(len(active_rows), 1)
            self.assertEqual(str(active_rows[0]["memory_id"]), v3)

    def test_queue_old_delete_new_upsert(self) -> None:
        """Rollback queues delete job for target and upsert job for new revision."""
        v1 = self.store.save_memory(
            self.owner, self.char, "q-1", "preference", "T1", [self.msg1]
        )
        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "q-2", "T2", [self.msg2]
        )

        # Clear prior jobs so we only inspect rollback jobs
        with self.store._connect() as conn:
            conn.execute("DELETE FROM index_jobs")

        res = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res["memory_id"]

        jobs = self.store.pending_jobs(self.owner, self.char)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["memory_id"], v2)
        self.assertEqual(jobs[0]["action"], "delete")
        self.assertEqual(jobs[1]["memory_id"], v3)
        self.assertEqual(jobs[1]["action"], "upsert")

    def test_wrong_owner_and_reader_denied(self) -> None:
        """Wrong owner and reader-only character are rejected with PermissionError."""
        v1 = self.store.save_memory(
            self.owner,
            self.char,
            "perm-1",
            "preference",
            "T1",
            [self.msg1],
            visible_to=[self.reader],
        )
        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "perm-2", "T2", [self.msg2]
        )

        # Wrong owner on active target
        with self.assertRaises(PermissionError):
            self.store.rollback_memory("wrong-owner", self.char, v2)

        # Reader character on active target
        with self.assertRaises(PermissionError):
            self.store.rollback_memory(self.owner, self.reader, v2)

        # Perform valid rollback so v2 becomes inactive
        self.store.rollback_memory(self.owner, self.char, v2)

        # Wrong owner on inactive target
        with self.assertRaises(PermissionError):
            self.store.rollback_memory("wrong-owner", self.char, v2)

        # Reader character on inactive target
        with self.assertRaises(PermissionError):
            self.store.rollback_memory(self.owner, self.reader, v2)

        # Delete lineage and test wrong owner on tombstone
        self.store.delete_memory(self.owner, self.char, v1)
        with self.assertRaises(PermissionError):
            self.store.rollback_memory("wrong-owner", self.char, v2)

    def test_no_parent_deleted_inactive_rejected(self) -> None:
        """Root revision, deleted lineage, and already inactive target are rejected with ValueError."""
        v1 = self.store.save_memory(
            self.owner, self.char, "val-1", "preference", "Root V1", [self.msg1]
        )

        # Root revision has no parent_id -> ValueError
        with self.assertRaises(ValueError):
            self.store.rollback_memory(self.owner, self.char, v1)

        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "val-2", "Corrected V2", [self.msg2]
        )
        res = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res["memory_id"]

        # v2 is now inactive -> calling rollback again rejects (no stale rollback)
        with self.assertRaises(ValueError):
            self.store.rollback_memory(self.owner, self.char, v2)

        # Deleted lineage rejects with ValueError
        self.store.delete_memory(self.owner, self.char, v3)
        with self.assertRaises(ValueError):
            self.store.rollback_memory(self.owner, self.char, v3)

        # Non-existent target rejects with ValueError
        with self.assertRaises(ValueError):
            self.store.rollback_memory(self.owner, self.char, "non-existent-uuid")

    def test_3version_rollback_restores_immediate_previous(self) -> None:
        """3-version chain rollback restores immediate previous revision."""
        v1 = self.store.save_memory(
            self.owner, self.char, "three-1", "preference", "V1 text", [self.msg1]
        )
        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "three-2", "V2 text", [self.msg2]
        )
        v3 = self.store.correct_memory(
            self.owner, self.char, v2, "three-3", "V3 text", [self.msg3]
        )

        res = self.store.rollback_memory(self.owner, self.char, v3)
        v4 = res["memory_id"]

        self.assertEqual(res["restored_from"], v2)
        self.assertEqual(res["superseded"], v3)

        mem = self.store.get_memory(self.owner, self.char, v4)
        self.assertIsNotNone(mem)
        self.assertEqual(mem["text"], "V2 text")
        self.assertEqual(mem["source_ids"], [self.msg2])

    def test_subsequent_rollback_on_restored_head(self) -> None:
        """Calling rollback on current restored head rolls back to its immediate parent."""
        v1 = self.store.save_memory(
            self.owner, self.char, "sub-1", "preference", "V1 text", [self.msg1]
        )
        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "sub-2", "V2 text", [self.msg2]
        )

        res1 = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res1["memory_id"]

        # v3 is active, its parent is v2. Rolling back v3 restores v2 content into v4!
        res2 = self.store.rollback_memory(self.owner, self.char, v3)
        v4 = res2["memory_id"]

        self.assertEqual(res2["restored_from"], v2)
        self.assertEqual(res2["superseded"], v3)

        mem = self.store.get_memory(self.owner, self.char, v4)
        self.assertIsNotNone(mem)
        self.assertEqual(mem["text"], "V2 text")

    def test_same_old_external_replay_remains_inactive(self) -> None:
        """Idempotent replay of historical correction does not reactivate anything."""
        v1 = self.store.save_memory(
            self.owner, self.char, "replay-ext-1", "preference", "V1", [self.msg1]
        )
        v2 = self.store.correct_memory(
            self.owner,
            self.char,
            v1,
            "replay-ext-2",
            "V2",
            [self.msg2],
        )

        res = self.store.rollback_memory(self.owner, self.char, v2)
        v3 = res["memory_id"]

        # Replay the correction for v2
        replay_id = self.store.correct_memory(
            self.owner,
            self.char,
            v1,
            "replay-ext-2",
            "V2",
            [self.msg2],
        )
        self.assertEqual(replay_id, v2)

        # Verify in DB: v2 remains superseded, v3 remains active
        with self.store._connect() as conn:
            v2_row = conn.execute(
                "SELECT status FROM memories WHERE memory_id = ?", (v2,)
            ).fetchone()
            v3_row = conn.execute(
                "SELECT status FROM memories WHERE memory_id = ?", (v3,)
            ).fetchone()

            self.assertEqual(v2_row["status"], "superseded")
            self.assertEqual(v3_row["status"], "active")

    def test_inject_correct_failure_rolls_back_no_half_row(self) -> None:
        """Injected failure inside correction rolls back transaction leaving no half row."""
        v1 = self.store.save_memory(
            self.owner, self.char, "fail-1", "preference", "V1", [self.msg1]
        )
        v2 = self.store.correct_memory(
            self.owner, self.char, v1, "fail-2", "V2", [self.msg2]
        )

        # Mock queue_job inside correct_memory to simulate unexpected failure
        with patch.object(self.store, "_queue_job", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                self.store.rollback_memory(self.owner, self.char, v2)

        # Verify DB: v2 is still active, no 3rd row created, no dangling changes
        with self.store._connect() as conn:
            v2_row = conn.execute(
                "SELECT status, superseded_by FROM memories WHERE memory_id = ?", (v2,)
            ).fetchone()
            self.assertEqual(v2_row["status"], "active")
            self.assertIsNone(v2_row["superseded_by"])

            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE root_id = ?", (v1,)
            ).fetchone()[0]
            self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
