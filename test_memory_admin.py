import unittest

from memory_admin import (EMPTY_PAGE, MemoryAdmin, build_keyboard, callback_delete,
                          callback_list, is_list_command, local_label, parse_callback,
                          render_page)
from memory_core import MemoryStore

OWNER, CHARACTER = "owner-admin", "role-admin"


class FakeIndexer:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def run_once(self, owner_id, character_id):
        if self.fail:
            raise RuntimeError("qdrant down")
        self.calls.append((owner_id, character_id))
        return {"status": "ok"}


def make_store(memories):
    store = MemoryStore(":memory:")
    indexer = FakeIndexer()
    ids = []
    for label, text in memories:
        from memory_write import remember_statement
        message_id = store.append_message(OWNER, CHARACTER, f"msg-{label}", "user", text)
        memory_id, _body = remember_statement(
            store, OWNER, CHARACTER, f"ext-{label}", "fact", text, [message_id],
            visible_to=[CHARACTER])
        ids.append(memory_id)
    return store, indexer, ids


class CommandTests(unittest.TestCase):
    def test_list_triggers(self):
        for text in ("記了什麼", " 記了什麼 ", "記憶清單", "記得什麼", "/memories"):
            self.assertTrue(is_list_command(text), text)

    def test_non_triggers(self):
        for text in ("我記了什麼嗎", "記得什麼時候", "", None, 123):
            self.assertFalse(is_list_command(text), text)

    def test_callback_roundtrip(self):
        self.assertEqual(parse_callback(callback_list(2)), ("list", 2))
        self.assertEqual(parse_callback(callback_delete("abc-123")), ("delete", "abc-123"))

    def test_bad_callbacks_rejected(self):
        for data in (None, "", "mem", "mem:list", "mem:list:x", "mem:del:", "mem:zzz:1",
                     "other:list:0", "mem:list:0:extra", "x" * 80):
            self.assertIsNone(parse_callback(data), data)

    def test_local_label(self):
        self.assertEqual(local_label("2026-10-20T08:00:43.292995Z"), "10/20 16:00")
        self.assertEqual(local_label(None), "--/--")
        self.assertEqual(local_label("not-a-date"), "--/--")


class PageTests(unittest.TestCase):
    def test_empty_page(self):
        store, indexer, _ids = make_store([])
        admin = MemoryAdmin(store, indexer, OWNER, CHARACTER)
        text, rows = admin.page()
        self.assertEqual(text, EMPTY_PAGE)
        # P2o-2: the empty list still offers the ➕ button so the first memory can be added.
        self.assertEqual(rows, [[("➕ 新增記憶", "mem:add")]])

    def test_render_numbers_and_hides_internal_ids(self):
        store, indexer, ids = make_store([("a", "我比較愛茉莉花茶"), ("b", "我們住這個小鎮")])
        admin = MemoryAdmin(store, indexer, OWNER, CHARACTER)
        text, rows = admin.page()
        self.assertIn("共 2 筆", text)
        self.assertIn("我比較愛茉莉花茶", text)
        self.assertIn("我們住這個小鎮", text)
        for memory_id in ids:
            self.assertNotIn(memory_id, text)
        self.assertEqual(len(rows[0]), 2)

    def test_pagination_second_page_and_nav(self):
        store, indexer, _ids = make_store([(f"m{i}", f"我喜歡第{i}樣東西") for i in range(10)])
        admin = MemoryAdmin(store, indexer, OWNER, CHARACTER)
        first, rows = admin.page(0)
        self.assertIn("第 1/2 頁", first)
        self.assertEqual(len(rows[0]), 8)
        self.assertEqual(rows[1], [("下一頁 ▶", callback_list(1))])
        second, rows2 = admin.page(1)
        self.assertIn("第 2/2 頁", second)
        self.assertEqual(len(rows2[0]), 2)
        self.assertEqual(rows2[1], [("◀ 上一頁", callback_list(0))])

    def test_out_of_range_page_falls_back_to_first(self):
        store, indexer, _ids = make_store([("a", "我喜歡烏龍")])
        admin = MemoryAdmin(store, indexer, OWNER, CHARACTER)
        text, _rows = admin.page(9)
        self.assertIn("共 1 筆", text)

    def test_delete_removes_and_syncs_index(self):
        store, indexer, ids = make_store([("a", "我比較愛茉莉花茶"), ("b", "我們住這個小鎮")])
        admin = MemoryAdmin(store, indexer, OWNER, CHARACTER)
        self.assertTrue(admin.delete(ids[0]))
        self.assertEqual(indexer.calls, [(OWNER, CHARACTER)])
        self.assertIsNone(store.get_memory(OWNER, CHARACTER, ids[0]))
        text, _rows = admin.page()
        self.assertIn("共 1 筆", text)
        self.assertNotIn("我比較愛茉莉花茶", text)

    def test_delete_unknown_id_is_false(self):
        store, indexer, _ids = make_store([("a", "我喜歡烏龍")])
        admin = MemoryAdmin(store, indexer, OWNER, CHARACTER)
        self.assertFalse(admin.delete("00000000-0000-4000-8000-000000000000"))

    def test_index_failure_does_not_undo_delete(self):
        store, _ignored, ids = make_store([("a", "我喜歡烏龍")])
        admin = MemoryAdmin(store, FakeIndexer(fail=True), OWNER, CHARACTER)
        self.assertTrue(admin.delete(ids[0]))
        self.assertEqual(admin.last_index_error, "RuntimeError")
        self.assertIsNone(store.get_memory(OWNER, CHARACTER, ids[0]))

    def test_admin_requires_dependencies(self):
        store, indexer, _ids = make_store([])
        with self.assertRaises(ValueError):
            MemoryAdmin(None, indexer, OWNER, CHARACTER)
        with self.assertRaises(ValueError):
            MemoryAdmin(store, None, OWNER, CHARACTER)


if __name__ == "__main__":
    unittest.main()
