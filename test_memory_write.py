import unittest

from memory_core import MemoryStore
from memory_write import MemoryWriteError, build_memory_text, is_verbatim, remember_statement

SOURCE = "我最近想喝點茶。我比較愛茉莉花茶，清香的那種，不要太濃。"


class MemoryWriteTests(unittest.TestCase):
    def test_body_is_a_verbatim_slice_of_the_user_message(self):
        single = "我比較愛茉莉花茶，清香的那種，不要太濃。"
        self.assertEqual(build_memory_text(single), single)
        self.assertEqual(build_memory_text(SOURCE), SOURCE)
        for text in (build_memory_text(single), build_memory_text(SOURCE)):
            self.assertTrue(is_verbatim(text, SOURCE))

    def test_whitespace_is_trimmed_but_wording_is_untouched(self):
        padded = "  我比較愛茉莉花茶，清香的那種。  "
        self.assertEqual(build_memory_text(padded), "我比較愛茉莉花茶，清香的那種。")

    def test_third_person_paraphrase_is_rejected(self):
        for bad in ("使用者偏好茉莉花茶。", "用戶喜歡這款茶。",
                    "Synthetic test user prefers fictional jasmine tea.",
                    "the user prefers jasmine tea"):
            with self.assertRaises(MemoryWriteError):
                build_memory_text(bad)

    def test_invalid_sources_are_rejected(self):
        for bad in ("", "   ", "\x00茶", None, 42, ["茶"]):
            with self.assertRaises(MemoryWriteError):
                build_memory_text(bad)
        with self.assertRaises(MemoryWriteError):
            build_memory_text("茶", max_length=0)
        with self.assertRaises(MemoryWriteError):
            build_memory_text("茶", max_length=10_000)
        with self.assertRaises(MemoryWriteError):
            build_memory_text("茶", max_length=True)

    def test_truncation_stays_verbatim_and_bounded(self):
        long_source = "我比較愛茉莉花茶，" * 60
        text = build_memory_text(long_source, max_length=50)
        self.assertLessEqual(len(text), 50)
        self.assertTrue(is_verbatim(text, long_source))

    def test_is_verbatim_rejects_rewrites(self):
        self.assertFalse(is_verbatim("使用者偏好茉莉花茶。", SOURCE))
        self.assertFalse(is_verbatim("", SOURCE))
        self.assertFalse(is_verbatim(SOURCE, SOURCE[:-1]))

    def test_remember_statement_stores_the_verbatim_body(self):
        store = MemoryStore(":memory:")
        try:
            source_id = store.append_message("owner-x", "role-x", "msg-1", "user", SOURCE)
            memory_id, text = remember_statement(store, "owner-x", "role-x", "mem-1", "preference",
                                                 SOURCE, [source_id])
            record = store.get_memory("owner-x", "role-x", memory_id)
            self.assertEqual(record["text"], text)
            self.assertTrue(is_verbatim(record["text"], SOURCE))
            self.assertEqual(record["kind"], "preference")
            self.assertEqual(record["source_ids"], [source_id])
        finally:
            store.close()

    def test_remember_statement_rejects_bad_kind_before_writing(self):
        store = MemoryStore(":memory:")
        try:
            source_id = store.append_message("owner-x", "role-x", "msg-1", "user", SOURCE)
            with self.assertRaises(MemoryWriteError):
                remember_statement(store, "owner-x", "role-x", "mem-1", "summary", SOURCE, [source_id])
            self.assertEqual(store.search("owner-x", "role-x", "茉莉"), [])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
