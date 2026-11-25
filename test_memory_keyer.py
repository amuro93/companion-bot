"""Unit tests for P2o-4: model-chosen keys, similarity candidates, serial decision."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import memory_auto as ma
import memory_keyer as mkz
from memory_core import MemoryStore

OWNER = "owner_a"
CHAR = "char_1"


class FakeEmbedder:
    """Character-histogram vectors: deterministic, and similar text scores higher."""

    def embed(self, texts):
        out = []
        for text in texts:
            vector = [0.0] * 16
            for char in text:
                vector[ord(char) % 16] += 1.0
            out.append(vector)
        return out


class FlakyEmbedder(FakeEmbedder):
    def embed(self, texts):
        raise RuntimeError("embedder offline")


class TestParseChoice(unittest.TestCase):
    def test_no(self) -> None:
        self.assertEqual(mkz.parse_choice("否"), (None, "no"))

    def test_valid_key(self) -> None:
        self.assertEqual(mkz.parse_choice("喜好/茶"), ("喜好/茶", "validated"))

    def test_candidate_is_preferred_verbatim(self) -> None:
        key, how = mkz.parse_choice("親密/性偏好", ("親密/性偏好",))
        self.assertEqual((key, how), ("親密/性偏好", "candidate"))

    def test_unknown_category_is_invalid(self) -> None:
        self.assertEqual(mkz.parse_choice("政治/立場"), (None, "invalid"))

    def test_too_long_is_invalid(self) -> None:
        self.assertEqual(mkz.parse_choice("喜" * 30), (None, "invalid"))

    def test_non_string_is_invalid(self) -> None:
        self.assertEqual(mkz.parse_choice(None), (None, "invalid"))


class TestCosine(unittest.TestCase):
    def test_parallel_vectors(self) -> None:
        self.assertAlmostEqual(mkz.cosine([1, 0], [2, 0]), 1.0)

    def test_zero_vector(self) -> None:
        self.assertEqual(mkz.cosine([0, 0], [1, 1]), 0.0)

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            mkz.cosine([1, 0], [1, 0, 0])


class TestDecisionOrder(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "keyer.db"
        self.store = MemoryStore(self.db_path)
        self.retriever = mkz.CandidateRetriever(self.store, FakeEmbedder(), OWNER, CHAR)
        self.msg = self.store.append_message(
            owner_id=OWNER, character_id=CHAR, external_id="m1", role="user",
            content="我喜歡很濃的高山烏龍茶",
        )
        self.store.save_keyed_memory(OWNER, CHAR, "auto-1", "preference",
                                     "我喜歡很濃的高山烏龍茶", [self.msg], "喜好/茶")

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()

    def test_model_answer_wins(self) -> None:
        chooser = mkz.ModelKeyChooser(lambda messages: "健康/血壓")
        decision = mkz.KeyDecision(chooser=chooser, retriever=self.retriever).decide("我血壓有點高")
        self.assertEqual(decision["key"], "健康/血壓")
        self.assertEqual(decision["how"], "model")

    def test_model_no_means_do_not_remember(self) -> None:
        chooser = mkz.ModelKeyChooser(lambda messages: "否")
        decision = mkz.KeyDecision(chooser=chooser, retriever=self.retriever).decide("我血壓有點高")
        self.assertIsNone(decision["key"])
        self.assertEqual(decision["how"], "no")

    def test_candidate_list_is_offered_to_the_model(self) -> None:
        seen = {}

        def model(messages):
            seen["prompt"] = messages[-1]["content"]
            return "喜好/茶"

        chooser = mkz.ModelKeyChooser(model)
        decision = mkz.KeyDecision(chooser=chooser, retriever=self.retriever).decide(
            "我喜歡很濃的高山烏龍茶")
        self.assertIn("喜好/茶", seen["prompt"])
        self.assertEqual(decision["how"], "model")
        self.assertIn("喜好/茶", decision["candidates"])

    def test_model_failure_falls_back_to_similarity(self) -> None:
        def boom(messages):
            raise RuntimeError("model down")

        chooser = mkz.ModelKeyChooser(boom)
        decision = mkz.KeyDecision(chooser=chooser, retriever=self.retriever,
                                   floor=0.5).decide("我喜歡很濃的高山烏龍茶")
        self.assertEqual(decision["how"], "similarity")
        self.assertEqual(decision["key"], "喜好/茶")

    def test_similarity_below_floor_falls_through_to_rule(self) -> None:
        class FixedRetriever:
            """Candidates exist but the best score is under the floor."""

            def top_keys(self, sentence, k=5):
                return ["喜好/茶"]

            def top(self, sentence, k=1):
                return [("喜好/茶", 0.31)]

        def boom(messages):
            raise RuntimeError("model down")

        chooser = mkz.ModelKeyChooser(boom)
        decision = mkz.KeyDecision(
            chooser=chooser, retriever=FixedRetriever(), floor=0.80,
            fallback_rule=ma.derive_key,
        ).decide("我最近血糖有點高")
        self.assertEqual(decision["how"], "rule")
        self.assertTrue(decision["key"].startswith(("健康/", "經歷/")))

    def test_similarity_never_overrides_a_model_answer(self) -> None:
        chooser = mkz.ModelKeyChooser(lambda messages: "經歷/重大事件")
        decision = mkz.KeyDecision(chooser=chooser, retriever=self.retriever,
                                   floor=0.0).decide("我喜歡很濃的高山烏龍茶")
        self.assertEqual(decision["key"], "經歷/重大事件")

    def test_invented_topic_is_coerced_to_the_category(self) -> None:
        key, how = mkz.parse_choice("健康/血壓值")
        self.assertEqual((key, how), ("健康", "validated"))
        self.assertEqual(mkz.parse_choice("喜好/茶"), ("喜好/茶", "validated"))

    def test_broken_retriever_does_not_break_the_model_path(self) -> None:
        broken = mkz.CandidateRetriever(self.store, FlakyEmbedder(), OWNER, CHAR)
        chooser = mkz.ModelKeyChooser(lambda messages: "喜好/茶")
        decision = mkz.KeyDecision(chooser=chooser, retriever=broken).decide("我喜歡喝茶")
        self.assertEqual(decision["how"], "model")
        self.assertEqual(decision["candidates"], [])


class TestKeyedPipelineWithKeyer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "flow.db"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()

    def test_model_chosen_key_is_used(self) -> None:
        chooser = mkz.ModelKeyChooser(lambda messages: "健康/血壓")
        keyer = mkz.KeyDecision(chooser=chooser)
        out = ma.record_user_message_keyed(
            self.store, OWNER, CHAR, "我血壓有點高，醫生叫我少鹽", keyer=keyer)
        self.assertEqual(out["results"][0]["key"], "健康/血壓")
        self.assertEqual(out["results"][0]["how"], "model")
        self.assertIsNotNone(self.store.find_active_by_key(OWNER, CHAR, "健康/血壓"))

    def test_model_no_writes_nothing(self) -> None:
        chooser = mkz.ModelKeyChooser(lambda messages: "否")
        keyer = mkz.KeyDecision(chooser=chooser)
        out = ma.record_user_message_keyed(
            self.store, OWNER, CHAR, "我血壓有點高，醫生叫我少鹽", keyer=keyer)
        self.assertEqual(out["results"], [])
        self.assertEqual(self.store.list_memories(OWNER, CHAR)["total"], 0)

    def test_identical_sentence_does_not_call_the_model_twice(self) -> None:
        calls = {"n": 0}

        def model(messages):
            calls["n"] += 1
            return "喜好/茶"

        keyer = mkz.KeyDecision(chooser=mkz.ModelKeyChooser(model))
        ma.record_user_message_keyed(self.store, OWNER, CHAR, "我喜歡很濃的烏龍茶", keyer=keyer)
        ma.record_user_message_keyed(self.store, OWNER, CHAR, "我喜歡很濃的烏龍茶", keyer=keyer)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(self.store.list_memories(OWNER, CHAR)["total"], 1)

    def test_two_sentences_same_key_correct(self) -> None:
        answers = iter(["喜好/茶", "喜好/茶"])
        chooser = mkz.ModelKeyChooser(lambda messages: next(answers))
        keyer = mkz.KeyDecision(chooser=chooser)
        ma.record_user_message_keyed(
            self.store, OWNER, CHAR, "我喜歡很濃的烏龍茶。我最近改喝麥茶。", keyer=keyer)
        row = self.store.find_active_by_key(OWNER, CHAR, "喜好/茶")
        self.assertEqual(row["text"], "我最近改喝麥茶")
        self.assertEqual(self.store.list_memories(OWNER, CHAR)["total"], 1)


if __name__ == "__main__":
    unittest.main()
