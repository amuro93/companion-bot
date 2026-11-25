"""Tests for category-only memory coexistence, legacy index migration, and idempotency guards.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest

from memory_core import MemoryStore


class TestMemoryCategoryCoexist(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.stores: list[MemoryStore] = []

    def tearDown(self) -> None:
        for store in self.stores:
            try:
                store.close()
            except Exception:
                pass
        self.stores.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_store(self, path: str | None = None) -> MemoryStore:
        db_path = ":memory:" if path is None else path
        store = MemoryStore(db_path)
        self.stores.append(store)
        return store

    def _seed_message(
        self,
        store: MemoryStore,
        owner_id: str = "user-1",
        character_id: str = "assistant-1",
        external_id: str = "msg-ext-1",
        content: str = "user message content",
    ) -> str:
        return store.append_message(
            owner_id=owner_id,
            character_id=character_id,
            external_id=external_id,
            role="user",
            content=content,
        )

    def test_01_two_category_facts_coexist(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-cat-1",
            kind="preference",
            text="user prefers dark roast coffee",
            source_ids=[m1],
            meta_key="preference",
        )
        r2 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-cat-2",
            kind="preference",
            text="user enjoys green tea in morning",
            source_ids=[m1],
            meta_key="preference",
        )

        self.assertEqual(r1["action"], "insert")
        self.assertEqual(r2["action"], "insert")
        self.assertNotEqual(r1["memory_id"], r2["memory_id"])

        mem1 = store.get_memory("user-1", "assistant-1", r1["memory_id"])
        mem2 = store.get_memory("user-1", "assistant-1", r2["memory_id"])
        self.assertIsNotNone(mem1)
        self.assertIsNotNone(mem2)

        listing = store.list_memories("user-1", "assistant-1")
        self.assertEqual(listing["total"], 2)

    def test_02_repeat_older_unchanged(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-cat-1",
            kind="preference",
            text="reads sci-fi novels",
            source_ids=[m1],
            meta_key="preference",
        )
        r2 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-cat-2",
            kind="preference",
            text="reads fantasy books",
            source_ids=[m1],
            meta_key="preference",
        )
        self.assertEqual(r1["action"], "insert")
        self.assertEqual(r2["action"], "insert")

        # Repeat fact 1 with varied whitespace and new external_id
        r3 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-cat-3",
            kind="preference",
            text="  reads   sci-fi   novels  ",
            source_ids=[m1],
            meta_key="preference",
        )
        self.assertEqual(r3["action"], "unchanged")
        self.assertEqual(r3["memory_id"], r1["memory_id"])

        listing = store.list_memories("user-1", "assistant-1")
        self.assertEqual(listing["total"], 2)

    def test_03_manual_category_coexistence_with_auto(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r_manual = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-manual",
            kind="preference",
            text="manual entry: prefers vegetarian",
            source_ids=[m1],
            meta_key="preference",
            source="manual",
        )
        self.assertEqual(r_manual["action"], "insert")

        # Unrelated auto fact must not be suppressed
        r_auto = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-auto",
            kind="preference",
            text="auto entry: enjoys seasonal tofu",
            source_ids=[m1],
            meta_key="preference",
            source="auto",
        )
        self.assertEqual(r_auto["action"], "insert")
        self.assertNotEqual(r_manual["memory_id"], r_auto["memory_id"])

    def test_04_domain_distinction_identical_text(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r_real = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-real",
            kind="fact",
            text="lives in a coastal town",
            source_ids=[m1],
            meta_key="fact",
            domain="real",
        )
        r_rp = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-rp",
            kind="fact",
            text="lives in a coastal town",
            source_ids=[m1],
            meta_key="fact",
            domain="roleplay",
        )
        self.assertEqual(r_real["action"], "insert")
        self.assertEqual(r_rp["action"], "insert")
        self.assertNotEqual(r_real["memory_id"], r_rp["memory_id"])

    def test_05_kind_distinction_identical_text(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r_pref = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-kind-pref",
            kind="preference",
            text="daily morning jogging",
            source_ids=[m1],
            meta_key="habit",
        )
        r_event = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-kind-event",
            kind="event",
            text="daily morning jogging",
            source_ids=[m1],
            meta_key="habit",
        )
        self.assertEqual(r_pref["action"], "insert")
        self.assertEqual(r_event["action"], "insert")
        self.assertNotEqual(r_pref["memory_id"], r_event["memory_id"])

    def test_06_owner_and_creator_distinction(self) -> None:
        store = self._create_store()
        m_u1_c1 = self._seed_message(
            store, owner_id="user-1", character_id="assistant-1", external_id="m1"
        )
        m_u1_c2 = self._seed_message(
            store, owner_id="user-1", character_id="assistant-2", external_id="m2"
        )
        m_u2_c1 = self._seed_message(
            store, owner_id="user-2", character_id="assistant-1", external_id="m3"
        )

        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-1",
            kind="fact",
            text="speaks multiple languages",
            source_ids=[m_u1_c1],
            meta_key="fact",
        )
        r2 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-2",
            external_id="ext-1",
            kind="fact",
            text="speaks multiple languages",
            source_ids=[m_u1_c2],
            meta_key="fact",
        )
        r3 = store.save_keyed_memory(
            owner_id="user-2",
            character_id="assistant-1",
            external_id="ext-1",
            kind="fact",
            text="speaks multiple languages",
            source_ids=[m_u2_c1],
            meta_key="fact",
        )

        self.assertEqual(r1["action"], "insert")
        self.assertEqual(r2["action"], "insert")
        self.assertEqual(r3["action"], "insert")
        self.assertNotEqual(r1["memory_id"], r2["memory_id"])
        self.assertNotEqual(r1["memory_id"], r3["memory_id"])

    def test_07_direct_save_memory_category_allowed(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        id1 = store.save_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="direct-ext-1",
            kind="preference",
            text="first direct category memory",
            source_ids=[m1],
            meta_key="food",
        )
        id2 = store.save_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="direct-ext-2",
            kind="preference",
            text="second direct category memory",
            source_ids=[m1],
            meta_key="food",
        )

        self.assertNotEqual(id1, id2)
        mem1 = store.get_memory("user-1", "assistant-1", id1)
        mem2 = store.get_memory("user-1", "assistant-1", id2)
        self.assertIsNotNone(mem1)
        self.assertIsNotNone(mem2)

    def test_08_direct_save_memory_named_db_uniqueness(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        id1 = store.save_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="named-ext-1",
            kind="preference",
            text="likes green tea",
            source_ids=[m1],
            meta_key="pref/drink",
        )
        self.assertTrue(id1)

        # Second active memory with the same named key must fail unique index
        with self.assertRaises(sqlite3.IntegrityError):
            store.save_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="named-ext-2",
                kind="preference",
                text="likes black coffee",
                source_ids=[m1],
                meta_key="pref/drink",
            )

    def test_09_named_key_supersedes_via_save_keyed(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="named-key-ext-1",
            kind="preference",
            text="favorite color is blue",
            source_ids=[m1],
            meta_key="pref/color",
            source="manual",
        )
        self.assertEqual(r1["action"], "insert")

        # Auto write for same named key is skipped due to manual precedence
        r_skip = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="named-key-ext-2",
            kind="preference",
            text="favorite color is green",
            source_ids=[m1],
            meta_key="pref/color",
            source="auto",
        )
        self.assertEqual(r_skip["action"], "skipped")
        self.assertEqual(r_skip["reason"], "manual takes precedence")

        # Manual correction succeeds
        r_corr = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="named-key-ext-3",
            kind="preference",
            text="favorite color is indigo",
            source_ids=[m1],
            meta_key="pref/color",
            source="manual",
        )
        self.assertEqual(r_corr["action"], "correct")
        self.assertEqual(r_corr["superseded"], r1["memory_id"])

    def test_10_external_id_conflict_changed_sources(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")
        m2 = self._seed_message(store, external_id="m2")

        store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-idem-src",
            kind="fact",
            text="travels by train",
            source_ids=[m1],
            meta_key="fact",
        )

        with self.assertRaises(ValueError) as ctx:
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-idem-src",
                kind="fact",
                text="travels by train",
                source_ids=[m2],
                meta_key="fact",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

    def test_11_external_id_conflict_changed_acl(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-idem-acl",
            kind="fact",
            text="enjoys gardening",
            source_ids=[m1],
            meta_key="fact",
            visible_to=["assistant-1"],
        )

        with self.assertRaises(ValueError) as ctx:
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-idem-acl",
                kind="fact",
                text="enjoys gardening",
                source_ids=[m1],
                meta_key="fact",
                visible_to=["assistant-1", "assistant-2"],
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

    def test_12_external_id_conflict_not_hidden_by_normalized_text(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")
        m2 = self._seed_message(store, external_id="m2")

        store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-shared-1",
            kind="fact",
            text="reads historical fiction",
            source_ids=[m1],
            meta_key="fact",
        )

        # Attempt to reuse ext-shared-1 with altered payload but matching normalized text
        with self.assertRaises(ValueError) as ctx:
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-shared-1",
                kind="fact",
                text="  reads   historical   fiction  ",
                source_ids=[m2],  # changed source_ids
                meta_key="fact",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

    def test_13_input_validation_despite_unchanged_text(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-orig",
            kind="preference",
            text="enjoys stargazing",
            source_ids=[m1],
            meta_key="preference",
        )

        # Invalid kind
        with self.assertRaises(ValueError):
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-invalid-kind",
                kind="invalid_kind",
                text="enjoys stargazing",
                source_ids=[m1],
                meta_key="preference",
            )

        # Invalid domain
        with self.assertRaises(ValueError):
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-invalid-domain",
                kind="preference",
                text="enjoys stargazing",
                source_ids=[m1],
                meta_key="preference",
                domain="fantasy",
            )

        # Out-of-scope source_id
        with self.assertRaises(ValueError):
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-invalid-src",
                kind="preference",
                text="enjoys stargazing",
                source_ids=["nonexistent-msg-id"],
                meta_key="preference",
            )

    def test_14_tombstone_prevents_resurrection(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-tomb",
            kind="fact",
            text="temporary hobby note",
            source_ids=[m1],
            meta_key="hobby",
        )
        self.assertEqual(r["action"], "insert")

        deleted = store.delete_memory("user-1", "assistant-1", r["memory_id"])
        self.assertTrue(deleted)

        with self.assertRaises(ValueError) as ctx:
            store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-tomb",
                kind="fact",
                text="temporary hobby note",
                source_ids=[m1],
                meta_key="hobby",
            )
        self.assertIn("deleted memory external_id", str(ctx.exception))

    def test_15_exact_replay_same_ext_no_jobs(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-replay",
            kind="fact",
            text="visited mountain trail",
            source_ids=[m1],
            meta_key="event",
        )
        self.assertEqual(r1["action"], "insert")

        jobs = store.pending_jobs("user-1", "assistant-1")
        self.assertEqual(len(jobs), 1)
        acked = store.ack_index_job("user-1", "assistant-1", jobs[0]["job_id"])
        self.assertTrue(acked)
        self.assertEqual(len(store.pending_jobs("user-1", "assistant-1")), 0)

        # Exact replay with same external_id
        r2 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-replay",
            kind="fact",
            text="visited mountain trail",
            source_ids=[m1],
            meta_key="event",
        )
        self.assertEqual(r2["action"], "insert")
        self.assertEqual(r2["memory_id"], r1["memory_id"])

        # No new job created
        jobs_after = store.pending_jobs("user-1", "assistant-1")
        self.assertEqual(len(jobs_after), 0)

    def test_16_legacy_index_migration_and_reopen(self) -> None:
        db_file = os.path.join(self.temp_dir, "legacy_test.db")

        # Create raw sqlite database with legacy index definition
        raw_conn = sqlite3.connect(db_file)
        raw_conn.execute(
            """
            CREATE TABLE memories (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL UNIQUE,
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                external_id TEXT NOT NULL,
                root_id TEXT NOT NULL,
                parent_id TEXT,
                superseded_by TEXT,
                kind TEXT NOT NULL,
                domain TEXT NOT NULL,
                text TEXT NOT NULL,
                meta_key TEXT,
                source TEXT NOT NULL DEFAULT 'auto',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL UNIQUE,
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                external_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL
            )
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE memory_sources (
                memory_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (memory_id, message_id)
            )
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE memory_acl (
                memory_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, character_id)
            )
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE memory_idempotency (
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                external_id TEXT NOT NULL,
                op_type TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                target_memory_id TEXT,
                root_id TEXT NOT NULL,
                payload_hash TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (owner_id, character_id, external_id)
            )
            """
        )
        raw_conn.execute(
            """
            CREATE TABLE index_jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL UNIQUE,
                memory_id TEXT NOT NULL,
                action TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Create legacy index without instr predicate
        raw_conn.execute(
            """
            CREATE UNIQUE INDEX idx_memories_active_meta_key
                ON memories(owner_id, character_id, meta_key)
                WHERE status = 'active' AND meta_key IS NOT NULL
            """
        )
        raw_conn.commit()
        raw_conn.close()

        # Step 1: Open store -> should detect old index and migrate it
        store = self._create_store(db_file)
        with store._connect() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_memories_active_meta_key'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertIn("instr(meta_key", "".join(str(row[0]).split()).lower())

        m1 = self._seed_message(store, external_id="m1")
        # Step 2: Verify category coexistence works
        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-mig-1",
            kind="preference",
            text="likes hiking",
            source_ids=[m1],
            meta_key="hobby",
        )
        r2 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-mig-2",
            kind="preference",
            text="likes cycling",
            source_ids=[m1],
            meta_key="hobby",
        )
        self.assertEqual(r1["action"], "insert")
        self.assertEqual(r2["action"], "insert")

        # Step 3: Close and reopen -> index must not be broken or needlessly rebuilt
        store.close()
        store2 = self._create_store(db_file)
        listing = store2.list_memories("user-1", "assistant-1")
        self.assertEqual(listing["total"], 2)

        # Step 4: Named keys must still enforce uniqueness
        store2.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-named-1",
            kind="preference",
            text="dark mode",
            source_ids=[m1],
            meta_key="ui/theme",
        )
        with self.assertRaises(sqlite3.IntegrityError):
            store2.save_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="ext-named-2",
                kind="preference",
                text="light mode",
                source_ids=[m1],
                meta_key="ui/theme",
            )
        store2.close()

    def test_17_migration_atomic_failure_keeps_old_index(self) -> None:
        db_file = os.path.join(self.temp_dir, "migration_fail.db")
        store = self._create_store(db_file)

        legacy_index_sql = (
            "CREATE UNIQUE INDEX idx_memories_active_meta_key\n"
            "    ON memories(owner_id, character_id, meta_key)\n"
            "    WHERE status = 'active' AND meta_key IS NOT NULL"
        )
        with store._connect() as conn:
            conn.execute("DROP INDEX idx_memories_active_meta_key")
            conn.execute(legacy_index_sql)
            row_before = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_memories_active_meta_key'"
            ).fetchone()
            self.assertIsNotNone(row_before)
            old_sql = row_before[0]

            def authorizer(
                action_code: int,
                arg1: str | None,
                arg2: str | None,
                dbname: str | None,
                source: str | None,
            ) -> int:
                if (
                    action_code == sqlite3.SQLITE_CREATE_INDEX
                    and arg1 == "idx_memories_active_meta_key"
                ):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            conn.set_authorizer(authorizer)
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    store._migrate_columns(conn)
            finally:
                conn.set_authorizer(None)

            row_after = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_memories_active_meta_key'"
            ).fetchone()
            self.assertIsNotNone(row_after)
            self.assertEqual(row_after[0], old_sql)
            self.assertFalse(conn.in_transaction)

    def test_18_find_active_by_key_deterministic_order(self) -> None:
        store = self._create_store()
        m1 = self._seed_message(store, external_id="m1")

        r1 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-det-1",
            kind="preference",
            text="version 1 preference",
            source_ids=[m1],
            meta_key="music",
        )
        r2 = store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="ext-det-2",
            kind="preference",
            text="version 2 preference",
            source_ids=[m1],
            meta_key="music",
        )

        found = store.find_active_by_key("user-1", "assistant-1", "music")
        self.assertIsNotNone(found)
        # Deterministically returns the latest active row
        self.assertEqual(found["memory_id"], r2["memory_id"])

    def test_19_legacy_correction_exact_replay_unchanged(self) -> None:
        db_file = os.path.join(self.temp_dir, "legacy_corr_replay.db")
        store = self._create_store(db_file)
        m1 = self._seed_message(store, external_id="m1")

        # Swap index to legacy definition to simulate pre-migration database state
        with store._connect() as conn:
            conn.execute("DROP INDEX idx_memories_active_meta_key")
            conn.execute(
                """
                CREATE UNIQUE INDEX idx_memories_active_meta_key
                    ON memories(owner_id, character_id, meta_key)
                    WHERE status = 'active' AND meta_key IS NOT NULL
                """
            )

        # Baseline workflow: insert memory then correct it under category meta_key
        id1 = store.save_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="save-ext",
            kind="fact",
            text="old synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )
        id2 = store.correct_memory(
            owner_id="user-1",
            character_id="assistant-1",
            memory_id=id1,
            external_id="correct-ext",
            text="updated synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )

        # Ack all existing jobs so pending queue is clean
        for job in store.pending_jobs("user-1", "assistant-1"):
            store.ack_index_job("user-1", "assistant-1", job["job_id"])
        self.assertEqual(len(store.pending_jobs("user-1", "assistant-1")), 0)

        # Close and reopen store to trigger product migration
        store.close()
        migrated_store = self._create_store(db_file)

        # Exact retry of the historical correction through PUBLIC save_keyed_memory
        r = migrated_store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="correct-ext",
            kind="fact",
            text="updated synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )
        self.assertEqual(r["action"], "unchanged")
        self.assertEqual(r["memory_id"], id2)

        # Rows and jobs remain unchanged
        with migrated_store._connect() as conn:
            total_mems = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            self.assertEqual(total_mems, 2)
        self.assertEqual(len(migrated_store.pending_jobs("user-1", "assistant-1")), 0)

    def test_20_legacy_correction_altered_payload_rejected(self) -> None:
        db_file = os.path.join(self.temp_dir, "legacy_corr_altered.db")
        store = self._create_store(db_file)
        m1 = self._seed_message(store, external_id="m1")
        m2 = self._seed_message(store, external_id="m2")

        with store._connect() as conn:
            conn.execute("DROP INDEX idx_memories_active_meta_key")
            conn.execute(
                """
                CREATE UNIQUE INDEX idx_memories_active_meta_key
                    ON memories(owner_id, character_id, meta_key)
                    WHERE status = 'active' AND meta_key IS NOT NULL
                """
            )

        id1 = store.save_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="save-ext",
            kind="fact",
            text="old synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )
        store.correct_memory(
            owner_id="user-1",
            character_id="assistant-1",
            memory_id=id1,
            external_id="correct-ext",
            text="updated synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )

        store.close()
        migrated_store = self._create_store(db_file)

        # Altered source_ids
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="updated synthetic fact",
                source_ids=[m2],
                meta_key="fact",
                source="auto",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

        # Altered source (auto vs manual)
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="updated synthetic fact",
                source_ids=[m1],
                meta_key="fact",
                source="manual",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

        # Altered ACL (visible_to)
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="updated synthetic fact",
                source_ids=[m1],
                meta_key="fact",
                source="auto",
                visible_to=["assistant-1", "assistant-2"],
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

        # Altered text
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="completely different text",
                source_ids=[m1],
                meta_key="fact",
                source="auto",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

        # Altered kind
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="preference",
                text="updated synthetic fact",
                source_ids=[m1],
                meta_key="fact",
                source="auto",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

        # Altered domain
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="updated synthetic fact",
                source_ids=[m1],
                meta_key="fact",
                domain="roleplay",
                source="auto",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

        # Altered meta_key
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="updated synthetic fact",
                source_ids=[m1],
                meta_key="other_category",
                source="auto",
            )
        self.assertIn("Idempotency conflict", str(ctx.exception))

    def test_21_legacy_correction_superseded_and_tombstone(self) -> None:
        db_file = os.path.join(self.temp_dir, "legacy_corr_superseded.db")
        store = self._create_store(db_file)
        m1 = self._seed_message(store, external_id="m1")

        with store._connect() as conn:
            conn.execute("DROP INDEX idx_memories_active_meta_key")
            conn.execute(
                """
                CREATE UNIQUE INDEX idx_memories_active_meta_key
                    ON memories(owner_id, character_id, meta_key)
                    WHERE status = 'active' AND meta_key IS NOT NULL
                """
            )

        id1 = store.save_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="save-ext",
            kind="fact",
            text="old synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )
        id2 = store.correct_memory(
            owner_id="user-1",
            character_id="assistant-1",
            memory_id=id1,
            external_id="correct-ext",
            text="updated synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )

        store.close()
        migrated_store = self._create_store(db_file)

        # Ack jobs
        for job in migrated_store.pending_jobs("user-1", "assistant-1"):
            migrated_store.ack_index_job("user-1", "assistant-1", job["job_id"])

        # Supersede id2 with a new correction (id3)
        id3 = migrated_store.correct_memory(
            owner_id="user-1",
            character_id="assistant-1",
            memory_id=id2,
            external_id="correct-ext-3",
            text="third generation synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )
        self.assertNotEqual(id2, id3)

        # Clear jobs again to track if replay adds any
        for job in migrated_store.pending_jobs("user-1", "assistant-1"):
            migrated_store.ack_index_job("user-1", "assistant-1", job["job_id"])
        self.assertEqual(len(migrated_store.pending_jobs("user-1", "assistant-1")), 0)

        # Replay the superseded historical correction (correct-ext)
        r = migrated_store.save_keyed_memory(
            owner_id="user-1",
            character_id="assistant-1",
            external_id="correct-ext",
            kind="fact",
            text="updated synthetic fact",
            source_ids=[m1],
            meta_key="fact",
            source="auto",
        )
        self.assertEqual(r["action"], "unchanged")
        self.assertEqual(r["memory_id"], id2)

        # Assert superseded legacy memory does NOT reactivate
        with migrated_store._connect() as conn:
            row = conn.execute(
                "SELECT status, superseded_by FROM memories WHERE memory_id = ?",
                (id2,),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "superseded")
            self.assertEqual(row["superseded_by"], id3)

        # Assert no jobs queued
        self.assertEqual(len(migrated_store.pending_jobs("user-1", "assistant-1")), 0)

        # Hard-delete the lineage to create tombstones
        deleted = migrated_store.delete_memory("user-1", "assistant-1", id3)
        self.assertTrue(deleted)

        # Replay of tombstoned historical external_id must be rejected
        with self.assertRaises(ValueError) as ctx:
            migrated_store.save_keyed_memory(
                owner_id="user-1",
                character_id="assistant-1",
                external_id="correct-ext",
                kind="fact",
                text="updated synthetic fact",
                source_ids=[m1],
                meta_key="fact",
                source="auto",
            )
        self.assertIn("deleted memory external_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
