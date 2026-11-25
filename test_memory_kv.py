"""Unit tests for P2o-1: fixed key taxonomy and key/value memory writes.

Covers memory_keys.normalize_key / validate_key, MemoryStore.save_keyed_memory
(insert / correct / unchanged / manual-wins), the active-key unique index, and the
in-place column migration for databases created before P2o.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import memory_keys as mk
from memory_core import MemoryStore

OWNER = "owner_a"
CHAR = "char_1"


class TestKeyTaxonomy(unittest.TestCase):
    def test_alias_maps_into_canonical_key(self) -> None:
        self.assertEqual(mk.normalize_key("茶"), ("喜好/茶", False))
        self.assertEqual(mk.normalize_key("老婆"), ("家人/配偶", False))
        self.assertEqual(mk.normalize_key("血壓"), ("健康/血壓", False))

    def test_free_text_contains_alias(self) -> None:
        key, review = mk.normalize_key("他喝高山烏龍的喜好")
        self.assertEqual(key, "喜好")
        self.assertFalse(review)

    def test_unknown_subject_is_provisional_and_flagged(self) -> None:
        key, review = mk.normalize_key("釣魚裝備堆在倉庫")
        self.assertTrue(review)
        self.assertTrue(key.startswith("經歷/"))
        self.assertLessEqual(len(key.split("/")[1]), mk.MAX_TOPIC_LEN)

    def test_category_only_key_is_allowed(self) -> None:
        self.assertEqual(mk.validate_key("飲食"), "飲食")

    def test_unknown_category_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mk.validate_key("政治/立場")

    def test_family_topics_are_closed(self) -> None:
        with self.assertRaises(ValueError):
            mk.validate_key("家人/朋友")
        self.assertEqual(mk.validate_key("家人/長輩"), "家人/長輩")

    def test_intimate_topics_are_closed_and_flagged(self) -> None:
        self.assertEqual(mk.validate_key("親密/性偏好"), "親密/性偏好")
        with self.assertRaises(ValueError):
            mk.validate_key("親密/其他")
        self.assertTrue(mk.is_intimate("親密/性偏好"))
        self.assertFalse(mk.is_intimate("喜好/茶"))

    def test_topic_length_capped_for_free_categories(self) -> None:
        key, review = mk.normalize_key("喜好", "一二三四五六七八")
        self.assertFalse(review)
        self.assertEqual(key, "喜好/一二三四五六")

    def test_category_topic_form(self) -> None:
        self.assertEqual(mk.normalize_key("家人", "配偶"), ("家人/配偶", False))
        self.assertEqual(mk.split_key("家人/配偶"), ("家人", "配偶"))


class TestKeyedMemory(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "kv.db"
        self.store = MemoryStore(self.db_path)
        self.msg = self.store.append_message(
            owner_id=OWNER,
            character_id=CHAR,
            external_id="msg_1",
            role="user",
            content="我最近很愛喝高山烏龍，越濃越好",
        )

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()

    def _save(self, ext: str, text: str, key: str, source: str = "auto"):
        return self.store.save_keyed_memory(
            OWNER, CHAR, ext, "preference", text, [self.msg], key, source=source
        )

    def test_insert(self) -> None:
        out = self._save("mem_1", "很愛喝高山烏龍", "喜好/茶")
        self.assertEqual(out["action"], "insert")
        row = self.store.find_active_by_key(OWNER, CHAR, "喜好/茶")
        self.assertEqual(row["meta_key"], "喜好/茶")
        self.assertEqual(row["source"], "auto")

    def test_summary_text_is_accepted(self) -> None:
        """P2o drops the P2e verbatim rule: a summary need not quote the message."""
        out = self._save("mem_1", "愛喝濃的高山烏龍", "喜好/茶")
        row = self.store.get_memory(OWNER, CHAR, out["memory_id"])
        self.assertEqual(row["text"], "愛喝濃的高山烏龍")
        self.assertNotIn(row["text"], "我最近很愛喝高山烏龍，越濃越好")

    def test_same_key_corrects_and_keeps_key(self) -> None:
        first = self._save("mem_1", "愛喝高山烏龍", "喜好/茶")
        second = self._save("mem_2", "改喝麥茶，喝濃的會睡不著", "喜好/茶")
        self.assertEqual(second["action"], "correct")
        self.assertEqual(second["superseded"], first["memory_id"])

        active = self.store.find_active_by_key(OWNER, CHAR, "喜好/茶")
        self.assertEqual(active["memory_id"], second["memory_id"])
        self.assertEqual(active["meta_key"], "喜好/茶")
        self.assertEqual(active["source"], "auto")
        self.assertEqual(
            self.store.get_memory(OWNER, CHAR, first["memory_id"]), None
        )

    def test_identical_text_is_unchanged(self) -> None:
        self._save("mem_1", "愛喝高山烏龍", "喜好/茶")
        out = self._save("mem_2", "愛喝高山烏龍", "喜好/茶")
        self.assertEqual(out["action"], "unchanged")
        self.assertEqual(len(self.store.list_memories(OWNER, CHAR)["items"]), 1)

    def test_manual_is_not_overwritten_by_auto(self) -> None:
        self.store.save_keyed_memory(
            OWNER, CHAR, "mem_1", "preference", "只喝無糖", [self.msg],
            "飲食/口味", source="manual",
        )
        out = self._save("mem_2", "喜歡甜食", "飲食/口味", source="auto")
        self.assertEqual(out["action"], "skipped")
        row = self.store.find_active_by_key(OWNER, CHAR, "飲食/口味")
        self.assertEqual(row["text"], "只喝無糖")
        self.assertEqual(row["source"], "manual")

    def test_manual_overwrites_auto(self) -> None:
        self._save("mem_1", "喜歡甜食", "飲食/口味", source="auto")
        out = self.store.save_keyed_memory(
            OWNER, CHAR, "mem_2", "preference", "已經戒糖了", [self.msg],
            "飲食/口味", source="manual",
        )
        self.assertEqual(out["action"], "correct")
        row = self.store.find_active_by_key(OWNER, CHAR, "飲食/口味")
        self.assertEqual(row["text"], "已經戒糖了")
        self.assertEqual(row["source"], "manual")

    def test_different_keys_do_not_collide(self) -> None:
        self._save("mem_1", "愛喝茶", "喜好/茶")
        self._save("mem_2", "愛散步", "喜好/散步")
        self.assertEqual(len(self.store.list_memories(OWNER, CHAR)["items"]), 2)

    def test_intimate_key_records_domain_and_source(self) -> None:
        out = self.store.save_keyed_memory(
            OWNER, CHAR, "mem_1", "preference", "喜歡慢一點", [self.msg],
            "親密/性偏好", source="auto",
        )
        row = self.store.get_memory(OWNER, CHAR, out["memory_id"])
        self.assertEqual(row["meta_key"], "親密/性偏好")
        self.assertEqual(row["domain"], "real")

    def test_database_enforces_one_active_per_key(self) -> None:
        self.store.save_memory(
            OWNER, CHAR, "mem_1", "preference", "一", [self.msg],
            meta_key="喜好/茶", source="auto",
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_memory(
                OWNER, CHAR, "mem_2", "preference", "二", [self.msg],
                meta_key="喜好/茶", source="auto",
            )

    def test_list_exposes_key_and_source(self) -> None:
        self._save("mem_1", "愛喝茶", "喜好/茶")
        items = self.store.list_memories(OWNER, CHAR)["items"]
        self.assertEqual(items[0]["meta_key"], "喜好/茶")
        self.assertEqual(items[0]["source"], "auto")


class TestColumnMigration(unittest.TestCase):
    """A database created before P2o must gain the columns in place."""

    OLD_SCHEMA = """
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
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """

    def test_old_database_is_migrated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "old.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(self.OLD_SCHEMA)
            conn.execute(
                """
                INSERT INTO memories (
                    memory_id, owner_id, character_id, external_id, root_id,
                    kind, domain, text, status, created_at, updated_at
                ) VALUES ('m1', ?, ?, 'ext1', 'm1', 'fact', 'real', '舊資料',
                          'active', '2026-01-01T00:00:00.000000Z',
                          '2026-01-01T00:00:00.000000Z')
                """,
                (OWNER, CHAR),
            )
            conn.commit()
            conn.close()

            store = MemoryStore(db_path)
            try:
                probe = sqlite3.connect(db_path)
                try:
                    cols = {str(r[1]) for r in probe.execute("PRAGMA table_info(memories)")}
                finally:
                    probe.close()
                self.assertIn("meta_key", cols)
                self.assertIn("source", cols)
                row = store.get_memory(OWNER, CHAR, "m1")
                self.assertEqual(row["text"], "舊資料")
                self.assertIsNone(row["meta_key"])
                self.assertEqual(row["source"], "auto")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
