"""Unit tests for isolated SQLite companion memory core (P1a).

Uses standard-library unittest, synthetic fixtures, and temporary directories.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
import uuid

from memory_core import MemoryStore


class TestMemoryCore(unittest.TestCase):
    """Comprehensive verification suite for MemoryStore."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "companion_memory.db"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()

    def _assert_valid_uuid(self, value: str) -> None:
        parsed = uuid.UUID(value)
        self.assertEqual(str(parsed), value)
        self.assertEqual(parsed.version, 4)

    def test_schema_and_foreign_keys_enabled(self) -> None:
        msg_id = self.store.append_message(
            owner_id="owner_a",
            character_id="char_1",
            external_id="msg_ext_1",
            role="user",
            content="Hello world",
        )
        mem_id = self.store.save_memory(
            owner_id="owner_a",
            character_id="char_1",
            external_id="mem_ext_1",
            kind="fact",
            text="User greeted the character",
            source_ids=[msg_id],
        )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            for expected in (
                "messages",
                "memories",
                "memory_sources",
                "memory_acl",
                "memory_idempotency",
                "index_jobs",
            ):
                self.assertIn(expected, tables)

            # Attempting to insert a source link with a nonexistent message_id must fail FK
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO memory_sources (
                        memory_id, message_id, owner_id, character_id, position
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (mem_id, "nonexistent_msg_id", "owner_a", "char_1", 99),
                )
        finally:
            conn.close()

    def test_schema_initialization_transaction_and_rollback(self) -> None:
        mem_store = MemoryStore(":memory:")
        try:
            assert mem_store._memory_conn is not None
            self.assertFalse(mem_store._memory_conn.in_transaction)
            fk_val = mem_store._memory_conn.execute("PRAGMA foreign_keys").fetchone()[0]
            self.assertEqual(fk_val, 1)

            bad_script = """
            BEGIN IMMEDIATE;
            CREATE TABLE rollback_probe (id INTEGER PRIMARY KEY);
            INSERT INTO nonexistent_table_during_init VALUES (1);
            COMMIT;
            """
            with self.assertRaises(sqlite3.OperationalError):
                mem_store._initialize_schema(mem_store._memory_conn, bad_script)

            self.assertFalse(mem_store._memory_conn.in_transaction)
            fk_after = mem_store._memory_conn.execute("PRAGMA foreign_keys").fetchone()[0]
            self.assertEqual(fk_after, 1)
            probe_row = mem_store._memory_conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'rollback_probe'"
            ).fetchone()
            self.assertIsNone(probe_row)
        finally:
            mem_store.close()

    def test_input_validation_and_malformed_inputs(self) -> None:
        with self.assertRaises(ValueError):
            MemoryStore("")
        with self.assertRaises(ValueError):
            MemoryStore("   ")
        with self.assertRaises(ValueError):
            MemoryStore(None)  # type: ignore[arg-type]

        # Identifier validation
        for bad_id in ("", "   ", " leading", "trailing ", "bad\nid", "x" * 200, None, 123):
            with self.assertRaises(ValueError):
                self.store.append_message(
                    owner_id=bad_id,  # type: ignore[arg-type]
                    character_id="char_1",
                    external_id="ext_1",
                    role="user",
                    content="valid text",
                )

        # Role validation
        for bad_role in ("system", "admin", "", None, 123):
            with self.assertRaises(ValueError):
                self.store.append_message(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_role",
                    role=bad_role,  # type: ignore[arg-type]
                    content="valid text",
                )

        # Content validation
        for bad_text in ("", "   \t\n ", "null\x00byte", None, 42):
            with self.assertRaises(ValueError):
                self.store.append_message(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_content",
                    role="user",
                    content=bad_text,  # type: ignore[arg-type]
                )

        # occurred_at validation
        for bad_ts in ("not-a-timestamp", "", 12345):
            with self.assertRaises(ValueError):
                self.store.append_message(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_ts",
                    role="user",
                    content="valid",
                    occurred_at=bad_ts,  # type: ignore[arg-type]
                )

        msg_id = self.store.append_message(
            owner_id="owner_a",
            character_id="char_1",
            external_id="valid_msg",
            role="user",
            content="Valid source message",
        )

        # Kind and domain validation
        for bad_kind in ("invalid", "note", "", None):
            with self.assertRaises(ValueError):
                self.store.save_memory(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_k",
                    kind=bad_kind,  # type: ignore[arg-type]
                    text="valid",
                    source_ids=[msg_id],
                )

        for bad_domain in ("fantasy", "global", "", None, 123):
            with self.assertRaises(ValueError):
                self.store.save_memory(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_d",
                    kind="fact",
                    text="valid",
                    source_ids=[msg_id],
                    domain=bad_domain,  # type: ignore[arg-type]
                )

        # source_ids validation (including empty or raw string instead of sequence)
        for bad_sources in ([], None, msg_id, [""], [123]):
            with self.assertRaises(ValueError):
                self.store.save_memory(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_s",
                    kind="fact",
                    text="valid",
                    source_ids=bad_sources,  # type: ignore[arg-type]
                )

        # visible_to validation
        for bad_visible in ("char_2", [""], [123], {"bad": "dict"}):
            with self.assertRaises(ValueError):
                self.store.save_memory(
                    owner_id="owner_a",
                    character_id="char_1",
                    external_id="ext_v",
                    kind="fact",
                    text="valid",
                    source_ids=[msg_id],
                    visible_to=bad_visible,  # type: ignore[arg-type]
                )

        # limit validation (must reject bools, <= 0, non-int, huge ints)
        for bad_limit in (True, False, 0, -5, "5", 1.5, None, 10_000):
            with self.assertRaises(ValueError):
                self.store.search(
                    owner_id="owner_a",
                    character_id="char_1",
                    query="valid",
                    limit=bad_limit,  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                self.store.pending_jobs(
                    owner_id="owner_a",
                    character_id="char_1",
                    limit=bad_limit,  # type: ignore[arg-type]
                )

    def test_message_idempotency_and_utc_normalization(self) -> None:
        m1 = self.store.append_message(
            owner_id="owner_1",
            character_id="char_a",
            external_id="msg_1",
            role="user",
            content="I love drinking oolong tea.",
            occurred_at="2026-10-18T16:30:00+08:00",
        )
        self._assert_valid_uuid(m1)

        # Exact replay (including equivalent UTC timestamp) returns same UUID
        m1_replay = self.store.append_message(
            owner_id="owner_1",
            character_id="char_a",
            external_id="msg_1",
            role="user",
            content="I love drinking oolong tea.",
            occurred_at=datetime(2026, 10, 18, 8, 30, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(m1, m1_replay)

        # Changed payload with same (owner_id, character_id, external_id) raises ValueError
        with self.assertRaises(ValueError):
            self.store.append_message(
                owner_id="owner_1",
                character_id="char_a",
                external_id="msg_1",
                role="user",
                content="I love drinking black coffee.",
                occurred_at="2026-10-18T16:30:00+08:00",
            )
        with self.assertRaises(ValueError):
            self.store.append_message(
                owner_id="owner_1",
                character_id="char_a",
                external_id="msg_1",
                role="assistant",
                content="I love drinking oolong tea.",
                occurred_at="2026-10-18T16:30:00+08:00",
            )

        # Different owner or different character can use same external_id independently
        m_other_char = self.store.append_message(
            owner_id="owner_1",
            character_id="char_b",
            external_id="msg_1",
            role="user",
            content="Different character message",
        )
        m_other_owner = self.store.append_message(
            owner_id="owner_2",
            character_id="char_a",
            external_id="msg_1",
            role="user",
            content="Different owner message",
        )
        self.assertNotEqual(m1, m_other_char)
        self.assertNotEqual(m1, m_other_owner)

        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT occurred_at FROM messages WHERE message_id = ?", (m1,)
            ).fetchone()
            self.assertEqual(row[0], "2026-10-18T08:30:00.000000Z")
        finally:
            conn.close()

    def test_source_integrity_and_scope_isolation(self) -> None:
        msg_o1_c1 = self.store.append_message(
            "owner_1", "char_1", "m1", "user", "Owner 1 Char 1 message"
        )
        msg_o1_c2 = self.store.append_message(
            "owner_1", "char_2", "m1", "user", "Owner 1 Char 2 message"
        )
        msg_o2_c1 = self.store.append_message(
            "owner_2", "char_1", "m1", "user", "Owner 2 Char 1 message"
        )

        # Missing source_id
        with self.assertRaises(ValueError):
            self.store.save_memory(
                "owner_1",
                "char_1",
                "mem_bad_1",
                "fact",
                "Fact with missing source",
                [str(uuid.uuid4())],
            )

        # Cross-character source_id (same owner, different character)
        with self.assertRaises(ValueError):
            self.store.save_memory(
                "owner_1",
                "char_1",
                "mem_bad_2",
                "fact",
                "Fact with cross-character source",
                [msg_o1_c2],
            )

        # Cross-owner source_id
        with self.assertRaises(ValueError):
            self.store.save_memory(
                "owner_1",
                "char_1",
                "mem_bad_3",
                "fact",
                "Fact with cross-owner source",
                [msg_o2_c1],
            )

        # Mixed valid and invalid source_ids rolls back completely
        with self.assertRaises(ValueError):
            self.store.save_memory(
                "owner_1",
                "char_1",
                "mem_bad_4",
                "fact",
                "Fact with mixed sources",
                [msg_o1_c1, msg_o1_c2],
            )

        self.assertEqual(self.store.pending_jobs("owner_1", "char_1"), [])

    def test_two_owners_two_characters_visibility_and_mutation_rights(self) -> None:
        msg_1a = self.store.append_message(
            "owner_1", "char_a", "m1", "user", "My favorite color is indigo."
        )
        msg_1a_2 = self.store.append_message(
            "owner_1", "char_a", "m2", "user", "We promised to visit Kyoto."
        )
        msg_1b = self.store.append_message(
            "owner_1", "char_b", "m1", "user", "Char B message."
        )
        msg_2a = self.store.append_message(
            "owner_2", "char_a", "m1", "user", "Owner 2 message."
        )

        priv_id = self.store.save_memory(
            owner_id="owner_1",
            character_id="char_a",
            external_id="mem_priv",
            kind="preference",
            text="Favorite color is indigo",
            source_ids=[msg_1a],
            domain="real",
            visible_to=None,
        )

        shared_id = self.store.save_memory(
            owner_id="owner_1",
            character_id="char_a",
            external_id="mem_shared",
            kind="promise",
            text="Promised to visit Kyoto together",
            source_ids=[msg_1a_2],
            domain="real",
            visible_to=["char_b"],
        )

        # Creator can read both
        self.assertIsNotNone(self.store.get_memory("owner_1", "char_a", priv_id))
        shared_doc = self.store.get_memory("owner_1", "char_a", shared_id)
        self.assertIsNotNone(shared_doc)
        assert shared_doc is not None
        self.assertEqual(shared_doc["id"], shared_id)
        self.assertEqual(shared_doc["creator_character"], "char_a")
        self.assertEqual(shared_doc["source_ids"], [msg_1a_2])

        # Same-owner char_b cannot read priv_id, but CAN read shared_id
        self.assertIsNone(self.store.get_memory("owner_1", "char_b", priv_id))
        shared_by_b = self.store.get_memory("owner_1", "char_b", shared_id)
        self.assertIsNotNone(shared_by_b)
        assert shared_by_b is not None
        self.assertEqual(shared_by_b["text"], "Promised to visit Kyoto together")
        self.assertEqual(shared_by_b["creator_character"], "char_a")

        # Search visibility for char_b
        self.assertEqual(self.store.search("owner_1", "char_b", "indigo"), [])
        kyoto_hits = self.store.search("owner_1", "char_b", "Kyoto")
        self.assertEqual(len(kyoto_hits), 1)
        self.assertEqual(kyoto_hits[0]["id"], shared_id)

        # Cross-owner isolation (owner_2 cannot read or search either, even with char_a/char_b)
        self.assertIsNone(self.store.get_memory("owner_2", "char_a", priv_id))
        self.assertIsNone(self.store.get_memory("owner_2", "char_a", shared_id))
        self.assertIsNone(self.store.get_memory("owner_2", "char_b", shared_id))
        self.assertEqual(self.store.search("owner_2", "char_a", "indigo"), [])
        self.assertEqual(self.store.search("owner_2", "char_b", "Kyoto"), [])

        # Shared reader (owner_1, char_b) CANNOT correct or delete shared_id
        with self.assertRaises(PermissionError):
            self.store.correct_memory(
                "owner_1", "char_b", shared_id, "unauthorized_corr", "Hacked", [msg_1b]
            )
        with self.assertRaises(PermissionError):
            self.store.delete_memory("owner_1", "char_b", shared_id)

        # Cross-owner (owner_2, char_a) CANNOT correct or delete shared_id or priv_id
        with self.assertRaises(PermissionError):
            self.store.correct_memory(
                "owner_2", "char_a", shared_id, "unauthorized_corr2", "Hacked", [msg_2a]
            )
        with self.assertRaises(PermissionError):
            self.store.delete_memory("owner_2", "char_a", shared_id)

        # Verify memory remains intact after unauthorized attempts
        still_there = self.store.get_memory("owner_1", "char_a", shared_id)
        self.assertIsNotNone(still_there)
        assert still_there is not None
        self.assertEqual(still_there["text"], "Promised to visit Kyoto together")

        # pending_jobs only visible to exact creator (owner_1, char_a)
        jobs_1a = self.store.pending_jobs("owner_1", "char_a")
        self.assertEqual(len(jobs_1a), 2)
        self.assertEqual(self.store.pending_jobs("owner_1", "char_b"), [])
        self.assertEqual(self.store.pending_jobs("owner_2", "char_a"), [])

    def test_search_literal_wildcards_chinese_domain_and_ordering(self) -> None:
        m1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src1")
        m2 = self.store.append_message("owner_1", "char_a", "m2", "user", "src2")

        id_pct = self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "專案進度已達 100% 完成", [m1]
        )
        id_und = self.store.save_memory(
            "owner_1", "char_a", "e2", "fact", "變數名稱為 user_name 欄位", [m1]
        )
        id_bs = self.store.save_memory(
            "owner_1", "char_a", "e3", "fact", "路徑位於 C:\\Users\\Alice", [m1]
        )
        id_tea = self.store.save_memory(
            "owner_1", "char_a", "e4", "preference", "最喜歡喝台灣凍頂烏龍茶與靜岡綠茶", [m1]
        )
        id_cat = self.store.save_memory(
            "owner_1", "char_a", "e5", "preference", "家裡養了一隻橘貓", [m2]
        )
        id_rp = self.store.save_memory(
            "owner_1",
            "char_a",
            "e6",
            "event",
            "在奇幻王國喝了魔法綠茶並許下約定",
            [m2],
            domain="roleplay",
        )

        # Literal wildcard searches must NOT match everything
        pct_hits = self.store.search("owner_1", "char_a", "100%")
        self.assertEqual([h["id"] for h in pct_hits], [id_pct])

        raw_pct_hits = self.store.search("owner_1", "char_a", "%")
        self.assertEqual([h["id"] for h in raw_pct_hits], [id_pct])

        und_hits = self.store.search("owner_1", "char_a", "_")
        self.assertEqual([h["id"] for h in und_hits], [id_und])

        bs_hits = self.store.search("owner_1", "char_a", "C:\\Users")
        self.assertEqual([h["id"] for h in bs_hits], [id_bs])

        # 1-character and 2-character Traditional Chinese search
        cat_hits = self.store.search("owner_1", "char_a", "貓")
        self.assertEqual([h["id"] for h in cat_hits], [id_cat])

        green_tea_all = self.store.search("owner_1", "char_a", "綠茶")
        # Newest-first deterministic ordering: id_rp was created after id_tea
        self.assertEqual([h["id"] for h in green_tea_all], [id_rp, id_tea])
        self.assertEqual(green_tea_all[0]["domain"], "roleplay")
        self.assertEqual(green_tea_all[1]["domain"], "real")

        # Domain filtering
        green_tea_real = self.store.search(
            "owner_1", "char_a", "綠茶", domain="real"
        )
        self.assertEqual([h["id"] for h in green_tea_real], [id_tea])

        green_tea_rp = self.store.search(
            "owner_1", "char_a", "綠茶", domain="roleplay"
        )
        self.assertEqual([h["id"] for h in green_tea_rp], [id_rp])

        # Limit enforcement
        limited = self.store.search("owner_1", "char_a", "茶", limit=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["id"], id_rp)

    def test_correction_lineage_and_idempotency(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "s1", "user", "I like coffee")
        s2 = self.store.append_message(
            "owner_1", "char_a", "s2", "user", "Actually I switched to matcha"
        )

        m1 = self.store.save_memory(
            owner_id="owner_1",
            character_id="char_a",
            external_id="save_drink",
            kind="preference",
            text="Likes drinking coffee",
            source_ids=[s1],
            domain="real",
            visible_to=["char_b"],
        )

        # Replay save_memory
        self.assertEqual(
            self.store.save_memory(
                owner_id="owner_1",
                character_id="char_a",
                external_id="save_drink",
                kind="preference",
                text="Likes drinking coffee",
                source_ids=[s1],
                domain="real",
                visible_to=["char_b"],
            ),
            m1,
        )

        m2 = self.store.correct_memory(
            owner_id="owner_1",
            character_id="char_a",
            memory_id=m1,
            external_id="corr_drink_1",
            text="Prefers drinking matcha instead of coffee",
            source_ids=[s1, s2],
        )
        self._assert_valid_uuid(m2)
        self.assertNotEqual(m1, m2)

        # Old memory is superseded and hidden from get/search
        self.assertIsNone(self.store.get_memory("owner_1", "char_a", m1))
        self.assertIsNone(self.store.get_memory("owner_1", "char_b", m1))
        self.assertEqual(self.store.search("owner_1", "char_a", "Likes drinking coffee"), [])

        # New memory preserves kind, domain, and ACL (char_b can read it)
        doc2 = self.store.get_memory("owner_1", "char_b", m2)
        self.assertIsNotNone(doc2)
        assert doc2 is not None
        self.assertEqual(doc2["kind"], "preference")
        self.assertEqual(doc2["domain"], "real")
        self.assertEqual(doc2["source_ids"], [s1, s2])
        self.assertEqual(doc2["creator_character"], "char_a")

        # Correction replay with same external_id is idempotent even though m1 is now superseded
        m2_replay = self.store.correct_memory(
            owner_id="owner_1",
            character_id="char_a",
            memory_id=m1,
            external_id="corr_drink_1",
            text="Prefers drinking matcha instead of coffee",
            source_ids=[s1, s2],
        )
        self.assertEqual(m2, m2_replay)

        # Changed replay for same external_id raises ValueError
        with self.assertRaises(ValueError):
            self.store.correct_memory(
                owner_id="owner_1",
                character_id="char_a",
                memory_id=m1,
                external_id="corr_drink_1",
                text="Different text on replay",
                source_ids=[s1, s2],
            )

        # New correction targeting already-superseded m1 with a new external_id fails
        with self.assertRaises(ValueError):
            self.store.correct_memory(
                owner_id="owner_1",
                character_id="char_a",
                memory_id=m1,
                external_id="corr_drink_2",
                text="Trying to branch from superseded m1",
                source_ids=[s2],
            )

        # Verify jobs: upsert(m1), delete(m1), upsert(m2)
        jobs = self.store.pending_jobs("owner_1", "char_a")
        self.assertEqual(
            [(j["memory_id"], j["action"]) for j in jobs],
            [(m1, "upsert"), (m1, "delete"), (m2, "upsert")],
        )

    def test_failed_write_rollback(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "s1", "user", "Valid source")
        m1 = self.store.save_memory(
            "owner_1", "char_a", "save_1", "fact", "Initial active memory", [s1]
        )

        # Attempt to correct m1 with a nonexistent source ID must roll back completely
        with self.assertRaises(ValueError):
            self.store.correct_memory(
                owner_id="owner_1",
                character_id="char_a",
                memory_id=m1,
                external_id="failed_corr",
                text="Should not persist",
                source_ids=[s1, "missing_source_id"],
            )

        # m1 must still be active and retrievable
        active_doc = self.store.get_memory("owner_1", "char_a", m1)
        self.assertIsNotNone(active_doc)
        assert active_doc is not None
        self.assertEqual(active_doc["text"], "Initial active memory")

        # Only the initial upsert job should exist
        jobs = self.store.pending_jobs("owner_1", "char_a")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["memory_id"], m1)
        self.assertEqual(jobs[0]["action"], "upsert")

    def test_delete_lineage_non_resurrection_jobs_and_raw_sqlite_inspection(self) -> None:
        orig_text = "ORIGINAL_CONVERSATION_MESSAGE_KEEP_ME"
        secret_v1 = "DERIVED_PRIVATE_MEMORY_V1_MUST_VANISH"
        secret_v2 = "DERIVED_PRIVATE_MEMORY_V2_MUST_VANISH"

        s1 = self.store.append_message(
            "owner_1", "char_a", "msg_keep_1", "user", orig_text
        )
        m1 = self.store.save_memory(
            owner_id="owner_1",
            character_id="char_a",
            external_id="ext_lineage_save",
            kind="fact",
            text=secret_v1,
            source_ids=[s1],
            visible_to=["char_b"],
        )
        m2 = self.store.correct_memory(
            owner_id="owner_1",
            character_id="char_a",
            memory_id=m1,
            external_id="ext_lineage_corr",
            text=secret_v2,
            source_ids=[s1],
        )

        # Delete the lineage via m2
        deleted = self.store.delete_memory("owner_1", "char_a", m2)
        self.assertTrue(deleted)

        # Repeating delete returns False
        self.assertFalse(self.store.delete_memory("owner_1", "char_a", m2))
        self.assertFalse(self.store.delete_memory("owner_1", "char_a", m1))
        self.assertFalse(self.store.delete_memory("owner_1", "char_a", str(uuid.uuid4())))

        # Unauthorized caller probing deleted memory still gets PermissionError
        with self.assertRaises(PermissionError):
            self.store.delete_memory("owner_1", "char_b", m2)
        with self.assertRaises(PermissionError):
            self.store.delete_memory("owner_2", "char_a", m2)

        # Neither m1 nor m2 is readable or searchable
        self.assertIsNone(self.store.get_memory("owner_1", "char_a", m1))
        self.assertIsNone(self.store.get_memory("owner_1", "char_a", m2))
        self.assertEqual(self.store.search("owner_1", "char_a", "DERIVED_PRIVATE"), [])

        # Replay of deleted external_ids must raise ValueError (no resurrection)
        with self.assertRaises(ValueError):
            self.store.save_memory(
                owner_id="owner_1",
                character_id="char_a",
                external_id="ext_lineage_save",
                kind="fact",
                text=secret_v1,
                source_ids=[s1],
                visible_to=["char_b"],
            )
        with self.assertRaises(ValueError):
            self.store.correct_memory(
                owner_id="owner_1",
                character_id="char_a",
                memory_id=m1,
                external_id="ext_lineage_corr",
                text=secret_v2,
                source_ids=[s1],
            )

        # Pending jobs: pending upserts removed, content-free delete jobs queued for lineage
        jobs = self.store.pending_jobs("owner_1", "char_a")
        self.assertEqual(
            [(j["memory_id"], j["action"]) for j in jobs],
            [(m1, "delete"), (m2, "delete")],
        )
        for job in jobs:
            self.assertNotIn("text", job)
            self.assertNotIn("content", job)

        # Raw SQLite inspection: derived texts must be completely absent across ALL tables,
        # while original message in `messages` is retained.
        conn = sqlite3.connect(self.db_path)
        try:
            mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            src_count = conn.execute("SELECT COUNT(*) FROM memory_sources").fetchone()[0]
            acl_count = conn.execute("SELECT COUNT(*) FROM memory_acl").fetchone()[0]
            self.assertEqual(mem_count, 0)
            self.assertEqual(src_count, 0)
            self.assertEqual(acl_count, 0)

            tombstones = conn.execute(
                "SELECT external_id, is_deleted, payload_hash FROM memory_idempotency"
            ).fetchall()
            self.assertEqual(len(tombstones), 2)
            for ext_id, is_del, phash in tombstones:
                self.assertIn(ext_id, ("ext_lineage_save", "ext_lineage_corr"))
                self.assertEqual(is_del, 1)
                self.assertIsNone(phash)

            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            ]
            all_db_strings: list[str] = []
            for tbl in tables:
                for row in conn.execute(f'SELECT * FROM "{tbl}"').fetchall():
                    for col in row:
                        if isinstance(col, str):
                            all_db_strings.append(col)

            combined_dump = "\n".join(all_db_strings)
            self.assertNotIn(secret_v1, combined_dump)
            self.assertNotIn(secret_v2, combined_dump)
            self.assertIn(orig_text, combined_dump)
        finally:
            conn.close()

    def test_db_reopen_persistence(self) -> None:
        s1 = self.store.append_message(
            "owner_1", "char_a", "persist_msg", "user", "Persistent source message"
        )
        m1 = self.store.save_memory(
            "owner_1",
            "char_a",
            "persist_mem",
            "event",
            "Graduated from university in Taipei",
            [s1],
            domain="real",
            visible_to=["char_b"],
        )
        self.store.close()

        reopened = MemoryStore(self.db_path)
        try:
            doc = reopened.get_memory("owner_1", "char_b", m1)
            self.assertIsNotNone(doc)
            assert doc is not None
            self.assertEqual(doc["text"], "Graduated from university in Taipei")
            self.assertEqual(doc["source_ids"], [s1])

            hits = reopened.search("owner_1", "char_a", "Taipei")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["id"], m1)

            # Idempotent replay works after reopen
            m1_again = reopened.save_memory(
                "owner_1",
                "char_a",
                "persist_mem",
                "event",
                "Graduated from university in Taipei",
                [s1],
                domain="real",
                visible_to=["char_b"],
            )
            self.assertEqual(m1, m1_again)
        finally:
            reopened.close()

    def test_concurrent_idempotency(self) -> None:
        s1 = self.store.append_message(
            "owner_1", "char_a", "conc_msg", "user", "Concurrent source message"
        )
        num_threads = 8
        barrier = threading.Barrier(num_threads)
        results: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            local_store = MemoryStore(self.db_path)
            try:
                barrier.wait(timeout=10.0)
                mem_id = local_store.save_memory(
                    owner_id="owner_1",
                    character_id="char_a",
                    external_id="concurrent_ext_1",
                    kind="preference",
                    text="Concurrent preference memory",
                    source_ids=[s1],
                )
                with lock:
                    results.append(mem_id)
            except Exception as exc:
                with lock:
                    errors.append(exc)
            finally:
                local_store.close()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), num_threads)
        self.assertEqual(len(set(results)), 1)

        jobs = self.store.pending_jobs("owner_1", "char_a")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["memory_id"], results[0])


if __name__ == "__main__":
    unittest.main()
