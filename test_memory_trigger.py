import unittest

from memory_core import MemoryStore
from memory_trigger import (candidates, is_candidate, kind_of, memory_external_id,
                            message_external_id, record_user_message)


class TriggerRuleTests(unittest.TestCase):
    def test_preferences_are_candidates(self):
        for sentence in ("我比較愛茉莉花茶，清香的那種", "我不太能吃辣", "我每天都要喝一杯咖啡"):
            self.assertTrue(is_candidate(sentence), sentence)

    def test_questions_and_chitchat_are_not(self):
        for sentence in ("你喜歡喝茶嗎？", "你覺得呢？", "今天有點累", "好", "哈哈",
                         "原來如此", "那台茶18號妳有喝過嗎？"):
            self.assertFalse(is_candidate(sentence), sentence)

    def test_third_person_sentence_is_not_a_candidate(self):
        self.assertFalse(is_candidate("他比較愛茉莉花茶"))
        self.assertFalse(is_candidate("使用者喜歡茉莉花茶"))

    def test_first_person_must_come_first(self):
        self.assertFalse(is_candidate("茉莉花茶這種東西我比較喜歡"))  # "我" too late in the sentence

    def test_at_most_one_candidate_per_message(self):
        message = "我比較愛茉莉花茶。我每天都會喝一杯。我兒子昨天去比賽。"
        self.assertEqual(len(candidates(message)), 1)
        self.assertEqual(candidates(message)[0], "我比較愛茉莉花茶")

    def test_invalid_message_is_not_a_candidate(self):
        for bad in ("", "   ", None, 42, "我喜歡" + "茶" * 2000):
            self.assertEqual(candidates(bad), [])

    def test_kind_inference(self):
        self.assertEqual(kind_of("我比較愛茉莉花茶"), "preference")
        self.assertEqual(kind_of("我兒子昨天去比賽"), "fact")
        self.assertEqual(kind_of("我上個月去過這個小鎮"), "event")


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_message_is_stored_even_without_a_memory(self):
        message_id, memory_ids = record_user_message(self.store, "owner-x", "role-x", "你喜歡喝茶嗎？")
        self.assertTrue(message_id)
        self.assertEqual(memory_ids, [])
        again, memory_ids_again = record_user_message(self.store, "owner-x", "role-x", "你喜歡喝茶嗎？")
        self.assertEqual(again, message_id)  # content-addressed, idempotent
        self.assertEqual(memory_ids_again, [])

    def test_candidate_becomes_a_verbatim_memory(self):
        content = "我比較愛茉莉花茶，清香的那種，不要太濃。"
        message_id, memory_ids = record_user_message(self.store, "owner-x", "role-x", content)
        self.assertEqual(len(memory_ids), 1)
        record = self.store.get_memory("owner-x", "role-x", memory_ids[0])
        self.assertEqual(record["text"], content.rstrip("。"))
        self.assertIn(record["text"], content)
        self.assertEqual(record["kind"], "preference")
        self.assertEqual(record["source_ids"], [message_id])

    def test_repeating_a_sentence_creates_exactly_one_memory(self):
        sentence = "我比較愛茉莉花茶。"
        for _ in range(3):
            record_user_message(self.store, "owner-x", "role-x", sentence)
        memories = self.store.search("owner-x", "role-x", "茉莉")
        self.assertEqual(len(memories), 1)

    def test_identifiers_are_content_addressed_and_scoped(self):
        self.assertEqual(message_external_id("我喜歡茶"), message_external_id("  我喜歡茶  "))
        a = memory_external_id("owner-x", "role-a", "我喜歡茉莉花茶")
        b = memory_external_id("owner-x", "role-b", "我喜歡茉莉花茶")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("auto-") and len(a) == 21)
