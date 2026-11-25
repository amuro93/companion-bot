"""Unit tests for P2o-2: manual memory entry (buttons, validation, writer)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

import memory_manual as mm
from memory_admin import MemoryAdmin, parse_callback
from memory_core import MemoryStore

OWNER = "owner_a"
CHAR = "char_1"


class FakeIndexer:
    def __init__(self) -> None:
        self.calls = 0

    def run_once(self, owner_id: str, character_id: str) -> None:
        self.calls += 1


class TestManualButtons(unittest.TestCase):
    def test_category_rows_cover_nine_categories(self) -> None:
        rows = mm.category_rows()
        flat = [label for row in rows for label, _ in row]
        self.assertEqual(len(flat), 9)
        self.assertIn("親密", flat)
        self.assertTrue(all(len(row) <= 3 for row in rows))

    def test_fixed_topic_categories_expose_topics(self) -> None:
        family = mm.topic_rows("家人")
        intimate = mm.topic_rows("親密")
        self.assertIsNotNone(family)
        self.assertEqual(len([1 for row in family for _ in row]), 3)
        self.assertEqual(len([1 for row in intimate for _ in row]), 6)
        self.assertIsNone(mm.topic_rows("喜好"))

    def test_callback_data_is_short(self) -> None:
        for row in mm.category_rows():
            for _, data in row:
                self.assertLessEqual(len(data.encode("utf-8")), 64)

    def test_parse_manual_callbacks(self) -> None:
        self.assertEqual(parse_callback("mem:add"), ("add", None))
        self.assertEqual(parse_callback("mem:skip"), ("skip", None))
        self.assertEqual(parse_callback("mem:cancel"), ("cancel", None))
        self.assertEqual(parse_callback("mem:cat:8"), ("category", 8))
        self.assertEqual(parse_callback("mem:top:8:5"), ("topic", (8, 5)))
        self.assertIsNone(parse_callback("mem:cat:x"))
        self.assertIsNone(parse_callback("mem:top:8"))


class TestManualValidation(unittest.TestCase):
    def test_content_bounds(self) -> None:
        self.assertEqual(mm.clean_content("  很愛喝高山烏龍  "), "很愛喝高山烏龍")
        with self.assertRaises(ValueError):
            mm.clean_content("   ")
        with self.assertRaises(ValueError):
            mm.clean_content("字" * (mm.MAX_CONTENT_LEN + 1))
        with self.assertRaises(ValueError):
            mm.clean_content("壞\x00字")

    def test_topic_bounds(self) -> None:
        with self.assertRaises(ValueError):
            mm.clean_topic("一二三四五六七")

    def test_build_key(self) -> None:
        self.assertEqual(mm.build_key("家人", "配偶"), "家人/配偶")
        self.assertEqual(mm.build_key("喜好", "茶"), "喜好/茶")
        self.assertEqual(mm.build_key("喜好", None), "喜好")
        with self.assertRaises(ValueError):
            mm.build_key("家人", None)
        with self.assertRaises(ValueError):
            mm.build_key("家人", "朋友")

    def test_kind_mapping(self) -> None:
        self.assertEqual(mm.kind_for("喜好"), "preference")
        self.assertEqual(mm.kind_for("經歷"), "event")
        self.assertEqual(mm.kind_for("約定與計畫"), "promise")
        self.assertEqual(mm.kind_for("親密"), "preference")

    def test_fingerprint_is_content_addressed(self) -> None:
        one = mm.content_fingerprint(OWNER, CHAR, "喜好/茶", "愛喝茶")
        two = mm.content_fingerprint(OWNER, CHAR, "喜好/茶", "愛喝茶")
        three = mm.content_fingerprint(OWNER, CHAR, "喜好/茶", "愛喝咖啡")
        self.assertEqual(one, two)
        self.assertNotEqual(one, three)

    def test_preview_line_marks_manual_and_key(self) -> None:
        line = mm.preview_line({"text": "很愛喝高山烏龍", "meta_key": "喜好/茶", "source": "manual"}, 1)
        self.assertIn("🔒", line)
        self.assertIn("喜好/茶", line)
        legacy = mm.preview_line({"text": "舊資料", "meta_key": None, "source": "auto"}, 2)
        self.assertIn("（未分類）", legacy)
        self.assertNotIn("🔒", legacy)


class TestManualFlow(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "manual.db"
        self.store = MemoryStore(self.db_path)
        self.indexer = FakeIndexer()
        self.admin = MemoryAdmin(self.store, self.indexer, OWNER, CHAR)
        self.msg = self.store.append_message(
            owner_id=OWNER, character_id=CHAR, external_id="msg_1",
            role="user", content="我最近很愛喝高山烏龍",
        )

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()

    def _walk_free_category(self, category: str, topic: str, content: str):
        idx = mm.category_index(category)
        self.admin.start_add()
        self.admin.choose_category(idx)
        body, _ = self.admin.submit_text(topic)
        self.assertIn("內容", body)
        return self.admin.submit_text(content)

    def test_full_flow_writes_manual_memory(self) -> None:
        body, rows = self._walk_free_category("喜好", "茶", "很愛喝高山烏龍，越濃越好")
        self.assertIn("✅ 記住了", body)
        self.assertTrue(rows)
        row = self.store.find_active_by_key(OWNER, CHAR, "喜好/茶")
        self.assertEqual(row["source"], "manual")
        self.assertEqual(row["text"], "很愛喝高山烏龍，越濃越好")
        self.assertEqual(self.indexer.calls, 1)

    def test_fixed_topic_flow_uses_buttons(self) -> None:
        self.admin.start_add()
        body, rows = self.admin.choose_category(mm.category_index("家人"))
        self.assertIn("配偶", [label for row in rows for label, _ in row])
        body, _ = self.admin.choose_topic_index(mm.category_index("家人"), 0)
        self.assertIn("內容", body)
        body, _ = self.admin.submit_text("她在醫院輪班，常常要加班")
        self.assertIn("✅ 記住了", body)
        row = self.store.find_active_by_key(OWNER, CHAR, "家人/配偶")
        self.assertEqual(row["text"], "她在醫院輪班，常常要加班")

    def test_intimate_category_records_domain_real(self) -> None:
        self.admin.start_add()
        self.admin.choose_category(mm.category_index("親密"))
        self.admin.choose_topic_index(mm.category_index("親密"), 0)
        self.admin.submit_text("喜歡慢一點")
        row = self.store.find_active_by_key(OWNER, CHAR, "親密/性偏好")
        self.assertEqual(row["domain"], "real")
        self.assertEqual(row["source"], "manual")

    def test_retry_same_content_is_unchanged(self) -> None:
        self._walk_free_category("喜好", "茶", "愛喝茶")
        self._walk_free_category("喜好", "茶", "愛喝茶")
        items = self.store.list_memories(OWNER, CHAR)["items"]
        self.assertEqual(len(items), 1)

    def test_manual_overwrites_existing_auto(self) -> None:
        self.store.save_keyed_memory(
            OWNER, CHAR, "auto_1", "preference", "喜歡甜食", [self.msg],
            "飲食/口味", source="auto",
        )
        self._walk_free_category("飲食", "口味", "已經戒糖了")
        row = self.store.find_active_by_key(OWNER, CHAR, "飲食/口味")
        self.assertEqual(row["text"], "已經戒糖了")
        self.assertEqual(row["source"], "manual")

    def test_skip_topic_uses_category_as_key(self) -> None:
        self.admin.start_add()
        self.admin.choose_category(mm.category_index("喜好"))
        body, _ = self.admin.skip_topic()
        self.assertIn("內容", body)
        self.admin.submit_text("愛散步")
        self.assertIsNotNone(self.store.find_active_by_key(OWNER, CHAR, "喜好"))

    def test_too_long_content_keeps_flow_alive(self) -> None:
        self.admin.start_add()
        self.admin.choose_category(mm.category_index("喜好"))
        self.admin.submit_text("茶")
        body, _ = self.admin.submit_text("字" * 41)
        self.assertIn("40 字", body)
        self.assertTrue(self.admin.has_pending())
        body, _ = self.admin.submit_text("愛喝茶")
        self.assertIn("✅ 記住了", body)

    def test_pending_expires_after_timeout(self) -> None:
        self.admin.start_add()
        self.admin._pending["at"] = time.monotonic() - (mm.PENDING_TIMEOUT_SECONDS + 1)
        self.assertFalse(self.admin.has_pending())
        body, _ = self.admin.submit_text("茶")
        self.assertIn("沒有進行中的新增流程", body)

    def test_cancel_clears_pending(self) -> None:
        self.admin.start_add()
        self.admin.cancel_pending()
        self.assertFalse(self.admin.has_pending())

    def test_listing_shows_lock_and_key(self) -> None:
        self._walk_free_category("喜好", "茶", "愛喝茶")
        body, rows = self.admin.page(0)
        self.assertIn("🔒", body)
        self.assertIn("喜好/茶", body)
        self.assertIn("➕ 新增記憶", [label for row in rows for label, _ in row])


if __name__ == "__main__":
    unittest.main()
