import unittest

from memory_core import MemoryStore
from memory_gate import (GATE_SYSTEM, LocalGate, parse_verdict, prefilter,
                         split_sentences)
from memory_trigger import gated_candidates, record_user_message

OWNER, CHARACTER = "owner-gate", "role-gate"


class FakeGate:
    def __init__(self, verdicts, fail=False):
        self.verdicts = list(verdicts)
        self.seen = []
        self.fail = fail

    def remember(self, sentence):
        self.seen.append(sentence)
        if self.fail:
            raise RuntimeError("model down")
        return self.verdicts.pop(0) if self.verdicts else False


class ParseVerdictTests(unittest.TestCase):
    def test_yes_forms(self):
        for reply in ("是", "是。", "是的", " 是\n", "Yes", "yes.", "要記"):
            self.assertTrue(parse_verdict(reply), reply)

    def test_no_forms(self):
        for reply in ("否", "否。", "不是", "No", "no", "", None, 123, "嗯…", "這要看情況"):
            self.assertFalse(parse_verdict(reply), reply)

    def test_not_a_statement_is_no(self):
        self.assertFalse(parse_verdict("不是長期事實"))


class SplitTests(unittest.TestCase):
    def test_split_drops_question_clauses(self):
        text = "說到看河景，我比較喜歡看老婆，我們已經有幾年沒去汽車旅館了?"
        self.assertEqual(split_sentences(text), ["說到看河景，我比較喜歡看老婆，我們已經有幾年沒去汽車旅館了"])

    def test_split_keeps_fact_before_trailing_question_clause(self):
        self.assertEqual(split_sentences("我血壓有點高，醫生叫我少鹽，這道菜不要太鹹好不好"),
                         ["我血壓有點高，醫生叫我少鹽"])
        self.assertEqual(split_sentences("我不太能吃辣，這家麻辣鍋妳確定要去嗎"),
                         ["我不太能吃辣"])

    def test_hard_stops(self):
        self.assertEqual(split_sentences("我喜歡烏龍。那時我還沒退伍！"),
                         ["我喜歡烏龍", "那時我還沒退伍"])

    def test_prefilter(self):
        self.assertFalse(prefilter("嗯"))
        self.assertFalse(prefilter("對阿，生完小孩都變了"))
        self.assertFalse(prefilter("記了什麼"))
        self.assertFalse(prefilter("妳確定妳不用加班嗎？"))
        self.assertTrue(prefilter("那時我還沒退伍啦"))
        self.assertFalse(prefilter(None))


class GateTests(unittest.TestCase):
    def test_prompt_scope_covers_family_life(self):
        # Requirement: family facts must be recordable — guard the prompt against silent narrowing.
        for marker in ("家庭生活", "配偶", "孩子", "長輩"):
            self.assertIn(marker, GATE_SYSTEM)

    def test_local_gate_asks_and_parses(self):
        calls = []

        def fake_model(messages):
            calls.append(messages)
            return "是"

        gate = LocalGate(fake_model)
        self.assertTrue(gate.remember("我們住這個小鎮"))
        self.assertEqual(calls[0][0]["content"], GATE_SYSTEM)
        self.assertIn("我們住這個小鎮", calls[0][1]["content"])

    def test_local_gate_fails_closed(self):
        def broken(messages):
            raise TimeoutError("slow")

        gate = LocalGate(broken)
        self.assertFalse(gate.remember("我們住這個小鎮"))
        self.assertEqual(gate.last_error, "TimeoutError")

    def test_local_gate_rejects_non_prefiltered_without_calling(self):
        calls = []
        gate = LocalGate(lambda messages: calls.append(messages) or "是")
        self.assertFalse(gate.remember("嗯"))
        self.assertEqual(calls, [])

    def test_gate_requires_callable(self):
        with self.assertRaises(ValueError):
            LocalGate("nope")


class GatedSelectionTests(unittest.TestCase):
    def test_only_approved_sentences_are_picked(self):
        gate = FakeGate([True, False])
        picked = gated_candidates("我們住這個小鎮。妳確定妳不用加班嗎？我也很想阿。", gate)
        self.assertEqual(picked, ["我們住這個小鎮"])

    def test_cap_per_message(self):
        gate = FakeGate([True, True, True, True])
        picked = gated_candidates("我喜歡烏龍。我們住這個小鎮。小孩讀國小。那時我還沒退伍。", gate, 2)
        self.assertEqual(len(picked), 2)

    def test_question_only_message_never_calls_gate(self):
        gate = FakeGate([True])
        self.assertEqual(gated_candidates("妳喜歡喝茶嗎？", gate), [])
        self.assertEqual(gate.seen, [])


class GatedWriteTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_approved_sentence_is_stored_verbatim(self):
        gate = FakeGate([True])
        content = "我們住這個小鎮。"
        message_id, memory_ids = record_user_message(self.store, OWNER, CHARACTER, content, gate=gate)
        self.assertEqual(len(memory_ids), 1)
        record = self.store.get_memory(OWNER, CHARACTER, memory_ids[0])
        self.assertEqual(record["text"], "我們住這個小鎮")
        self.assertIn(record["text"], content)
        self.assertEqual(record["source_ids"], [message_id])

    def test_rejected_sentence_stores_message_only(self):
        gate = FakeGate([False])
        message_id, memory_ids = record_user_message(self.store, OWNER, CHARACTER,
                                                     "我也很想阿。", gate=gate)
        self.assertTrue(message_id)
        self.assertEqual(memory_ids, [])

    def test_gate_failure_stores_nothing_and_does_not_raise(self):
        message_id, memory_ids = record_user_message(self.store, OWNER, CHARACTER,
                                                     "我們住這個小鎮。", gate=FakeGate([True], fail=True))
        self.assertTrue(message_id)
        self.assertEqual(memory_ids, [])

    def test_repeat_with_gate_still_dedupes(self):
        for _ in range(3):
            record_user_message(self.store, OWNER, CHARACTER, "我們住這個小鎮。", gate=FakeGate([True]))
        self.assertEqual(len(self.store.search(OWNER, CHARACTER, "這個小鎮")), 1)

    def test_same_sentence_from_two_messages_does_not_raise(self):
        first = "我們住這個小鎮。"
        second = "對阿，我們住這個小鎮。"
        _id1, ids1 = record_user_message(self.store, OWNER, CHARACTER, first, gate=FakeGate([True]))
        _id2, ids2 = record_user_message(self.store, OWNER, CHARACTER, second, gate=FakeGate([True]))
        self.assertEqual(len(ids1), 1)
        self.assertEqual(ids2, [])  # already remembered, kept the first source link
        self.assertEqual(len(self.store.search(OWNER, CHARACTER, "這個小鎮")), 1)


if __name__ == "__main__":
    unittest.main()
