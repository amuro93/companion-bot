"""Unit tests for P2o-3: keyed automatic writes, manual precedence, delete blocks."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import memory_auto as ma
import memory_keys as mk
from memory_admin import MemoryAdmin
from memory_core import MemoryStore

OWNER = "owner_a"
CHAR = "char_1"


class YesGate:
    """Stand-in for the local model: every prefilted sentence is remembered."""

    def remember(self, sentence: str) -> bool:
        return True


class NoGate:
    def remember(self, sentence: str) -> bool:
        return False


class FakeIndexer:
    def __init__(self) -> None:
        self.calls = 0

    def run_once(self, owner_id: str, character_id: str) -> None:
        self.calls += 1


class TestDeriveKey(unittest.TestCase):
    def test_alias_in_sentence_wins(self) -> None:
        self.assertEqual(ma.derive_key("我太太在醫院輪班"), ("家人/配偶", False))
        self.assertEqual(ma.derive_key("我血壓有點高，醫生叫我少鹽"), ("健康/血壓", False))

    def test_family_without_relative_is_parked(self) -> None:
        key, review = ma.derive_key("我家人最近都很忙")
        self.assertTrue(review)
        self.assertTrue(key.startswith("經歷/"))

    def test_residence(self) -> None:
        self.assertEqual(ma.derive_key("我們住這個小鎮"), ("住居與地緣/住處", False))
        # "住院" is not a residence: it falls through to the review bucket.
        key, review = ma.derive_key("我上個月住院三天")
        self.assertTrue(review)
        self.assertTrue(key.startswith("經歷/"))

    def test_eating_preference(self) -> None:
        self.assertEqual(ma.derive_key("我最近很愛喝高山烏龍，越濃越好"), ("飲食/口味", False))

    def test_generic_preference(self) -> None:
        self.assertEqual(ma.derive_key("我比較喜歡看電影"), ("喜好/影劇", False))

    def test_deterministic(self) -> None:
        sentence = "我最近很愛喝高山烏龍"
        self.assertEqual(ma.derive_key(sentence), ma.derive_key(sentence))

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ma.derive_key("   ")


class TestAutoWrites(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "auto.db"
        self.store = MemoryStore(self.db_path)
        self.indexer = FakeIndexer()
        self.admin = MemoryAdmin(self.store, self.indexer, OWNER, CHAR)

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()

    def _turn(self, text: str, gate=None):
        return ma.record_user_message_keyed(self.store, OWNER, CHAR, text, gate=gate)

    def test_insert_with_source_auto(self) -> None:
        out = self._turn("我最近很愛喝高山烏龍", gate=YesGate())
        self.assertEqual(out["results"][0]["action"], "insert")
        row = self.store.find_active_by_key(OWNER, CHAR, out["results"][0]["key"])
        self.assertEqual(row["source"], "auto")
        self.assertIsNotNone(out["message_id"])

    def test_second_statement_on_same_key_corrects(self) -> None:
        first = self._turn("我喜歡很濃的高山烏龍茶", gate=YesGate())
        second = self._turn("我現在改喝麥茶，喝濃的會睡不著", gate=YesGate())
        key = first["results"][0]["key"]
        self.assertEqual(key, "喜好/茶")
        self.assertEqual(second["results"][0]["key"], key)
        self.assertEqual(second["results"][0]["action"], "correct")
        active = self.store.find_active_by_key(OWNER, CHAR, key)
        self.assertEqual(active["text"], "我現在改喝麥茶，喝濃的會睡不著")
        c = sqlite3.connect(self.db_path)
        try:
            total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            superseded = c.execute(
                "SELECT COUNT(*) FROM memories WHERE status='superseded'"
            ).fetchone()[0]
        finally:
            c.close()
        self.assertEqual(total, 2)          # history kept, not deleted
        self.assertEqual(superseded, 1)

    def test_no_gate_verdict_means_no_memory(self) -> None:
        out = self._turn("我最近很愛喝高山烏龍", gate=NoGate())
        self.assertEqual(out["results"], [])
        self.assertEqual(out["memory_ids"], [])
        self.assertEqual(self.store.list_memories(OWNER, CHAR)["total"], 0)

    def test_manual_memory_is_never_overwritten(self) -> None:
        self.admin.create_manual("喜好", "茶", "只喝無糖烏龍")
        out = self._turn("我最近很愛喝麥茶", gate=YesGate())
        self.assertEqual(out["results"][0]["key"], "喜好/茶")
        self.assertEqual(out["results"][0]["action"], "skipped")
        row = self.store.find_active_by_key(OWNER, CHAR, "喜好/茶")
        self.assertEqual(row["text"], "只喝無糖烏龍")
        self.assertEqual(row["source"], "manual")

    def test_deleted_sentence_does_not_come_back(self) -> None:
        first = self._turn("我最近很愛喝高山烏龍", gate=YesGate())
        memory_id = first["memory_ids"][0]
        self.assertTrue(self.store.delete_memory(OWNER, CHAR, memory_id))
        again = self._turn("我最近很愛喝高山烏龍", gate=YesGate())
        self.assertEqual(again["results"][0]["action"], "tombstoned")
        self.assertEqual(self.store.list_memories(OWNER, CHAR)["total"], 0)

    def test_manual_delete_blocks_the_key(self) -> None:
        self.admin.create_manual("喜好", "茶", "只喝無糖烏龍")
        memory_id = self.store.find_active_by_key(OWNER, CHAR, "喜好/茶")["memory_id"]
        self.assertTrue(self.admin.delete(memory_id))
        self.assertTrue(self.store.is_key_blocked(OWNER, CHAR, "喜好/茶"))
        out = self._turn("我最近很愛喝麥茶", gate=YesGate())
        self.assertEqual(out["results"][0]["action"], "blocked")
        self.assertEqual(self.store.list_memories(OWNER, CHAR)["total"], 0)

    def test_manual_entry_lifts_the_block(self) -> None:
        self.store.block_key(OWNER, CHAR, "喜好/茶")
        out = self.admin.create_manual("喜好", "茶", "只喝無糖烏龍")
        self.assertEqual(out["action"], "insert")
        self.assertFalse(self.store.is_key_blocked(OWNER, CHAR, "喜好/茶"))
        again = self._turn("我最近很愛喝麥茶", gate=YesGate())
        self.assertEqual(again["results"][0]["action"], "skipped")

    def test_auto_delete_does_not_block(self) -> None:
        out = self._turn("我喜歡很濃的烏龍茶", gate=YesGate())
        key = out["results"][0]["key"]
        self.assertTrue(self.admin.delete(out["memory_ids"][0]))
        self.assertFalse(self.store.is_key_blocked(OWNER, CHAR, key))
        after = self._turn("我最近改喝麥茶", gate=YesGate())
        self.assertEqual(after["results"][0]["action"], "insert")

    def test_needs_review_is_reported(self) -> None:
        out = self._turn("我家人都很忙，最近事情很多", gate=YesGate())
        self.assertTrue(out["results"])
        self.assertTrue(out["results"][0]["needs_review"])


if __name__ == "__main__":
    unittest.main()
