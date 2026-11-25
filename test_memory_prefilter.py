# -*- coding: utf-8 -*-
"""前置篩（DecisionPrefilter）與第二段主題選擇的測試。

設計原則：決策模型是「便宜的前置篩」，它只能擋掉不值得記的句子，
不能改變 LLM 的判斷邏輯；而且它自己壞掉時必須**退回原本行為**，不是漏記。
"""
import unittest
from unittest import mock

import memory_keyer as k


class FakeChooser:
    """舊介面 chooser（不吃 category）。"""

    def __init__(self, reply=(("喜好/茶"), "validated")):
        self.reply = reply
        self.calls = []

    def choose(self, sentence, candidates=()):
        self.calls.append((sentence, tuple(candidates), None))
        return self.reply


class CategoryChooser:
    """新介面 chooser：記錄收到的 category。"""

    def __init__(self, reply=("喜好/茶", "candidate")):
        self.reply = reply
        self.calls = []

    def choose(self, sentence, candidates=(), category=None):
        self.calls.append((sentence, tuple(candidates), category))
        return self.reply


def prefilter_returning(payload):
    """造一個 DecisionPrefilter，但 _post_json 直接回 payload。"""
    pf = k.DecisionPrefilter("http://127.0.0.1:8092/v1/systemone")
    with mock.patch.object(k, "_post_json", return_value=payload):
        return pf.judge("測試句")


def ok_payload(remember, category=None):
    return {"answers": {"remember": {"type": "noul", "noul": remember},
                        "category": {"type": "choice", "choice": category}}}


class TestDecisionPrefilter(unittest.TestCase):
    def test_remember_true_returns_category(self):
        remember, category, how = prefilter_returning(ok_payload(0.92, "喜好"))
        self.assertIs(remember, True)
        self.assertEqual(category, "喜好")
        self.assertEqual(how, "ok")

    def test_remember_false_skips_category(self):
        remember, category, how = prefilter_returning(ok_payload(0.03, None))
        self.assertIs(remember, False)
        self.assertIsNone(category)
        self.assertEqual(how, "ok")

    def test_uncertain_probability_defers_to_full_model(self):
        # A 0.34 score rejected two actual family facts; deferral is not auto-acceptance.
        for score in (0.30, 0.34, 0.499):
            with self.subTest(score=score):
                self.assertEqual(prefilter_returning(ok_payload(score)),
                                 (None, None, "uncertain"))

    def test_clear_no_and_yes_keep_fast_paths(self):
        self.assertEqual(prefilter_returning(ok_payload(0.299)), (False, None, "ok"))
        self.assertEqual(prefilter_returning(ok_payload(0.5, "喜好")), (True, "喜好", "ok"))

    def test_transport_error_is_not_a_no(self):
        """決策模型掛掉不可以被當成「不記」——那會靜默漏記。"""
        pf = k.DecisionPrefilter("http://127.0.0.1:8092/v1/systemone")
        with mock.patch.object(k, "_post_json", side_effect=OSError("boom")):
            remember, category, how = pf.judge("測試句")
        self.assertIsNone(remember)
        self.assertIsNone(category)
        self.assertEqual(how, "error")

    def test_category_outside_vocabulary_is_invalid(self):
        remember, category, how = prefilter_returning(ok_payload(0.9, "不存在的類"))
        self.assertIsNone(remember)
        self.assertEqual(how, "invalid")

    def test_missing_probability_is_invalid(self):
        remember, category, how = prefilter_returning({"answers": {}})
        self.assertIsNone(remember)
        self.assertEqual(how, "invalid")


class TestKeyDecisionWithPrefilter(unittest.TestCase):
    def test_prefilter_no_skips_chooser_entirely(self):
        chooser = CategoryChooser()
        pf = k.DecisionPrefilter("http://x")
        with mock.patch.object(pf, "judge", return_value=(False, None, "ok")):
            decision = k.KeyDecision(chooser=chooser, prefilter=pf)
            out = decision.decide("今天天氣真好")
        self.assertEqual(out["how"], "prefilter_no")
        self.assertIsNone(out["key"])
        self.assertEqual(chooser.calls, [], "prefilter 說不記時不該再叫 LLM")

    def test_prefilter_yes_passes_category_to_chooser(self):
        chooser = CategoryChooser()
        pf = k.DecisionPrefilter("http://x")
        with mock.patch.object(pf, "judge", return_value=(True, "喜好", "ok")):
            decision = k.KeyDecision(chooser=chooser, prefilter=pf)
            out = decision.decide("我喜歡喝高山烏龍")
        self.assertEqual(out["how"], "model")
        self.assertEqual(out["key"], "喜好/茶")
        self.assertEqual(chooser.calls[0][2], "喜好")

    def test_prefilter_failure_falls_back_to_full_llm_judgement(self):
        """決策模型壞了 → category=None，LLM 從零判斷（維持原本行為）。"""
        chooser = CategoryChooser(reply=("家人/配偶", "validated"))
        pf = k.DecisionPrefilter("http://x")
        with mock.patch.object(pf, "judge", return_value=(None, None, "error")):
            decision = k.KeyDecision(chooser=chooser, prefilter=pf)
            out = decision.decide("我太太在醫院上班")
        self.assertEqual(out["how"], "model")
        self.assertEqual(out["key"], "家人/配偶")
        self.assertIsNone(chooser.calls[0][2])

    def test_uncertain_score_runs_full_model_and_preserves_refusal(self):
        pf = k.DecisionPrefilter("http://x")
        with mock.patch.object(k, "_post_json", return_value=ok_payload(0.34)):
            chooser = CategoryChooser(reply=("家人/長輩", "validated"))
            accepted = k.KeyDecision(chooser=chooser, prefilter=pf).decide("老爸行動可自理")
            self.assertEqual(accepted["key"], "家人/長輩")
            self.assertIsNone(chooser.calls[0][2])
            self.assertEqual(accepted["trace"][0]["how"], "uncertain")
            chooser = CategoryChooser(reply=(None, "no"))
            refused = k.KeyDecision(chooser=chooser, prefilter=pf).decide("只是開玩笑")
            self.assertIsNone(refused["key"])
            self.assertIsNone(chooser.calls[0][2])

    def test_old_style_chooser_still_works(self):
        """舊介面 chooser 不吃 category，不能因為多傳一個參數就炸。"""
        chooser = FakeChooser()
        pf = k.DecisionPrefilter("http://x")
        with mock.patch.object(pf, "judge", return_value=(True, "喜好", "ok")):
            decision = k.KeyDecision(chooser=chooser, prefilter=pf)
            out = decision.decide("我喜歡喝茶")
        self.assertEqual(out["how"], "model")
        self.assertEqual(out["key"], "喜好/茶")

    def test_no_prefilter_keeps_original_behaviour(self):
        chooser = CategoryChooser()
        decision = k.KeyDecision(chooser=chooser)
        out = decision.decide("我喜歡喝茶")
        self.assertEqual(out["how"], "model")
        self.assertIsNone(chooser.calls[0][2])


class TestPersonaRelationContract(unittest.TestCase):
    """關係由人設宣告（personas/relationship.md），完全不從字詞猜「我／妳」。"""

    def decide_with_reply(self, sentence, reply, retriever=None, addressee_family_key=None):
        messages_seen = []
        def model_reply(messages):
            messages_seen.append(messages)
            return reply
        pf = k.DecisionPrefilter("http://x")
        with mock.patch.object(pf, "judge", return_value=(True, "工作與身分", "ok")):
            out = k.KeyDecision(
                chooser=k.ModelKeyChooser(model_reply), prefilter=pf,
                retriever=retriever, addressee_family_key=addressee_family_key,
                fallback_rule=lambda _: ("工作與身分/職業", False),
            ).decide(sentence)
        return out, messages_seen

    def test_relation_switches_to_full_gate_and_offers_the_key(self):
        for sentence in ("妳放假還回醫院加班", "我太太在醫院輪班"):
            with self.subTest(sentence=sentence):
                out, seen = self.decide_with_reply(sentence, "家人/配偶",
                                                   addressee_family_key="家人/配偶")
                self.assertEqual(out["key"], "家人/配偶")
                self.assertEqual(seen[0][0]["content"], k.KEY_SYSTEM)
                self.assertIn("家人/配偶", seen[0][1]["content"])
                self.assertIn("persona_relation", [t["stage"] for t in out["trace"]])

    def test_model_reply_no_means_not_stored(self):
        # 模型說「否」就是不記。舊版靠字詞補救（`當` 誤中「當然」），已整組移除。
        key = "家人/配偶"
        for negative in ("妳的同事在醫院加班", "妳爸在醫院加班",
                         "妳看我又要去醫院上班了", "如果妳以後去醫院上班",
                         "妳不用加班", "妳當然不用加班", "妳沒上班"):
            with self.subTest(negative=negative):
                out, _ = self.decide_with_reply(negative, "否", addressee_family_key=key)
                self.assertIsNone(out["key"])
                self.assertEqual(out["how"], "no")

    def test_no_word_guessing_of_subject(self):
        # 句子裡有「她」不再觸發任何字詞判斷：模型選什麼就是什麼。
        out, _ = self.decide_with_reply("她現在在醫院，常常要輪班", "工作與身分/職業",
                                        addressee_family_key="家人/配偶")
        self.assertEqual(out["key"], "工作與身分/職業")
        self.assertEqual(out["how"], "model")

    def test_relation_cannot_be_an_unvalidated_arbitrary_key(self):
        with self.assertRaises(ValueError):
            k.KeyDecision(addressee_family_key="工作與身分/職業")

    def test_no_relation_keeps_category_hint_fast_path(self):
        out, seen = self.decide_with_reply("我在醫院上班", "職業")
        self.assertEqual(out["key"], "工作與身分/職業")
        self.assertNotEqual(seen[0][0]["content"], k.KEY_SYSTEM)
        self.assertNotIn("persona_relation", [t["stage"] for t in out["trace"]])

    def test_model_no_never_writes_through_real_entry(self):
        from memory_auto import record_user_message_keyed
        from memory_core import MemoryStore
        sentence = "妳當然不用加班"
        pf = k.DecisionPrefilter("http://x")
        keyer = k.KeyDecision(
            chooser=k.ModelKeyChooser(lambda _: "否"), prefilter=pf,
            fallback_rule=lambda _: ("工作與身分/職業", False),
            addressee_family_key="家人/配偶",
        )
        store = MemoryStore(":memory:")
        try:
            with mock.patch.object(pf, "judge", return_value=(True, "工作與身分", "ok")):
                outcome = record_user_message_keyed(
                    store, "audit-owner", "audit-char", sentence, keyer=keyer)
            self.assertTrue(outcome["message_id"])
            self.assertEqual(store.list_memories("audit-owner", "audit-char")["items"], [])
            self.assertEqual(outcome["trace"][-1]["action"], "refused")
        finally:
            store.close()


class TestParseTopic(unittest.TestCase):
    def test_bare_topic_becomes_key(self):
        key, how = k.parse_topic("茶", "喜好")
        self.assertEqual(key, "喜好/茶")
        self.assertEqual(how, "validated")

    def test_bare_category_is_accepted_but_flagged(self):
        """模型逃避選主題時仍接受，但要標 category_only 讓上游統計。"""
        key, how = k.parse_topic("喜好", "喜好")
        self.assertEqual(key, "喜好")
        self.assertEqual(how, "category_only")

    def test_candidate_is_preferred_verbatim(self):
        key, how = k.parse_topic("茶", "喜好", ["喜好/茶"])
        self.assertEqual(key, "喜好/茶")
        self.assertEqual(how, "candidate")

    def test_invented_topic_is_coerced_to_category(self):
        key, _how = k.parse_topic("血壓值", "健康")
        self.assertEqual(key, "健康", "詞表外的主題要降級成純分類")

    def test_model_may_overrule_the_prefiltered_category(self):
        """前置篩會把「我太太在醫院輪班」判成工作與身分；
        第二段必須能回「家人/配偶」把它救回來。"""
        key, how = k.parse_topic("家人/配偶", "工作與身分")
        self.assertEqual(key, "家人/配偶")
        self.assertEqual(how, "validated")

    def test_topic_prompt_states_the_escape_hatch(self):
        prompt = k.topic_prompt("工作與身分")
        self.assertIn("其他分類", prompt)
        self.assertIn("初步判定", prompt)


class TestTopicHints(unittest.TestCase):
    """模型常無視「一定要選主題」直接回分類名（實測 6 句裡 4 句），
    所以程式層用關鍵詞決定性補一刀。"""

    def test_child_course(self):
        self.assertEqual(k._hint_topic("約定與計畫", "小明早上十點半英文課，下午兩點圍棋課"), "子女課程")

    def test_travel(self):
        self.assertEqual(k._hint_topic("約定與計畫", "這次家族旅遊要去日本日本"), "出遊")

    def test_child_medical(self):
        self.assertEqual(k._hint_topic("約定與計畫", "大寶眼科我已經帶他看過了"), "子女就醫")

    def test_grocery(self):
        self.assertEqual(k._hint_topic("約定與計畫", "巷口那間涼麵六日都有擺攤"), "採買")

    def test_housework(self):
        self.assertEqual(k._hint_topic("約定與計畫", "洗碗洗衣服都歸我"), "家務分工")

    def test_dietary_restriction_beats_meal_time(self):
        self.assertEqual(k._hint_topic("飲食", "我不喜歡吃魚耶"), "忌口")

    def test_unknown_sentence_stays_at_category(self):
        self.assertIsNone(k._hint_topic("約定與計畫", "這個我再想想"))

    def test_unknown_category_has_no_hints(self):
        self.assertIsNone(k._hint_topic("親密", "小明英文課"))

    def test_hint_wins_over_bare_category_from_the_model(self):
        """模型回純分類時，choose() 要用關鍵詞把主題補上。"""
        chooser = k.ModelKeyChooser(lambda messages: "約定與計畫")
        key, how = chooser.choose("小明早上十點半英文課", (), category="約定與計畫")
        self.assertEqual(key, "約定與計畫/子女課程")
        self.assertEqual(how, "hinted")

    def test_hint_does_not_override_a_real_topic(self):
        chooser = k.ModelKeyChooser(lambda messages: "出遊")
        key, how = chooser.choose("小明英文課", (), category="約定與計畫")
        self.assertEqual(key, "約定與計畫/出遊")
        self.assertEqual(how, "validated")

    def test_empty_is_invalid(self):
        key, how = k.parse_topic("", "喜好")
        self.assertIsNone(key)
        self.assertEqual(how, "invalid")


if __name__ == "__main__":
    unittest.main()
