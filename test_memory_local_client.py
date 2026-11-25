"""Concise unit tests for InProcessMemoryClient and real CompanionChat integration."""
from __future__ import annotations

import math
import unittest
from typing import Any, Dict, List, Optional

from companion_chat import (
    ChatScope,
    CompanionChat,
    MemoryUnavailable,
    PRIOR_USER_HEADER,
)
from memory_http import Scope
from memory_local_client import InProcessMemoryClient

# Fictional tokens (>=32 chars, ascii, printable, no space)
TOKEN_A = "tok_test_fictional_scope_000000001_abc"
TOKEN_B = "tok_test_fictional_scope_000000002_def"
TOKEN_DIS = "tok_test_fictional_scope_disabled_0003"
TOKEN_UNKNOWN = "tok_test_fictional_scope_unknown_00004"

SCOPE_A = Scope(owner_id="owner_alpha", character_id="char_alpha", enabled=True)
SCOPE_B = Scope(owner_id="owner_beta", character_id="char_beta", enabled=True)
SCOPE_DIS = Scope(owner_id="owner_gamma", character_id="char_gamma", enabled=False)


class DeterministicFakeSearcher:
    """Deterministic in-memory searcher recording invocations."""

    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None, error: Optional[Exception] = None):
        self.calls: List[tuple[str, str, str, int, Optional[str]]] = []
        self.hits = hits if hits is not None else []
        self.error = error

    def search(
        self,
        owner_id: str,
        character_id: str,
        query: str,
        limit: int,
        domain: Optional[str],
    ) -> List[Dict[str, Any]]:
        self.calls.append((owner_id, character_id, query, limit, domain))
        if self.error is not None:
            raise self.error
        return [dict(h) for h in self.hits]


# Minimal dummy objects to simulate Telegram update for CompanionChat
class DummyUser:
    def __init__(self, uid: int):
        self.id = uid


class DummyChat:
    def __init__(self, cid: int):
        self.id = cid
        self.type = "private"


class DummyMessage:
    def __init__(self, text: str):
        self.text = text


class DummyUpdate:
    def __init__(self, user_id: int, text: str):
        self.effective_user = DummyUser(user_id)
        self.effective_chat = DummyChat(user_id)
        self.message = DummyMessage(text)


class TestInProcessMemoryClient(unittest.TestCase):
    def setUp(self):
        self.searcher = DeterministicFakeSearcher()
        self.credentials = {
            TOKEN_A: SCOPE_A,
            TOKEN_B: SCOPE_B,
            TOKEN_DIS: SCOPE_DIS,
        }
        self.client = InProcessMemoryClient(self.searcher, self.credentials)

    def test_constructor_validation(self):
        # Missing or invalid credentials
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {})
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, None)  # type: ignore

        # Invalid token (< 32 chars, unicode, spaces, non-str)
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {"short": SCOPE_A})
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {TOKEN_A + "★": SCOPE_A})
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {TOKEN_A + " ": SCOPE_A})
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {12345678901234567890123456789012: SCOPE_A})  # type: ignore

        # Invalid scope
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {TOKEN_A: "not-a-scope"})  # type: ignore
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, {TOKEN_A: Scope(owner_id="", character_id="char")})

        # Duplicate scope
        duplicate_creds = {
            TOKEN_A: SCOPE_A,
            TOKEN_B: Scope(owner_id=SCOPE_A.owner_id, character_id=SCOPE_A.character_id),
        }
        with self.assertRaises(ValueError):
            InProcessMemoryClient(self.searcher, duplicate_creds)

        # Missing callable search
        with self.assertRaises(ValueError):
            InProcessMemoryClient(object(), self.credentials)

    def test_immutable_snapshot_caller_mutation(self):
        mutable_creds = dict(self.credentials)
        client = InProcessMemoryClient(self.searcher, mutable_creds)

        # Caller mutates their dictionary
        new_token = "tok_test_fictional_scope_new_caller_mut"
        mutable_creds[new_token] = Scope(owner_id="new_owner", character_id="new_char")
        del mutable_creds[TOKEN_A]

        # Original token still functions
        hits = client.search(TOKEN_A, "test query", 5, "real")
        self.assertEqual(hits, [])

        # Injected token rejected
        with self.assertRaises(MemoryUnavailable):
            client.search(new_token, "test query", 5, "real")

    def test_proper_scope_args(self):
        self.client.search(TOKEN_A, "user memory search query", 3, "real")
        self.assertEqual(len(self.searcher.calls), 1)
        self.assertEqual(
            self.searcher.calls[0],
            ("owner_alpha", "char_alpha", "user memory search query", 3, "real"),
        )

    def test_unknown_disabled_unicode_long_token_rejected_without_backend_call(self):
        bad_tokens = [
            TOKEN_UNKNOWN,
            TOKEN_DIS,
            TOKEN_A + "★",
            "A" * 1025,
            "short_token_under_32_chars",
            "",
            None,
            12345678901234567890123456789012,
        ]
        for token in bad_tokens:
            with self.subTest(token=token):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(token, "test query", 5, "real")  # type: ignore
                self.assertEqual(len(self.searcher.calls), 0)

    def test_invalid_query_limit_domain_rejected_without_backend_call(self):
        # Invalid queries
        for bad_query in ["", "   ", "\x00", "hello\x00world", "x" * 1025, None, 123]:
            with self.subTest(bad_query=bad_query):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, bad_query, 5, "real")  # type: ignore
                self.assertEqual(len(self.searcher.calls), 0)

        # Invalid limits
        for bad_limit in [0, -1, 1001, True, False, 1.5, "5", None]:
            with self.subTest(bad_limit=bad_limit):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", bad_limit, "real")  # type: ignore
                self.assertEqual(len(self.searcher.calls), 0)

        # Invalid domains (including unhashable types)
        for bad_domain in ["fantasy", "admin", "REAL", "", 123, [], {}]:
            with self.subTest(bad_domain=bad_domain):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", 5, bad_domain)  # type: ignore
                self.assertEqual(len(self.searcher.calls), 0)

    def test_domain_none_accepted(self):
        self.searcher.hits = [{
            "memory_id": "mem_001",
            "text": "statement",
            "domain": "roleplay",
            "score": 0.9,
            "source_ids": ["src_01"],
            "owner_id": "owner_alpha",
            "visible_to": ["char_alpha"],
        }]
        res = self.client.search(TOKEN_A, "test query", 5, None)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["domain"], "roleplay")
        self.assertEqual(self.searcher.calls[0][4], None)

    def test_empty_healthy_results(self):
        self.searcher.hits = []
        hits = self.client.search(TOKEN_A, "test query", 5, "real")
        self.assertEqual(hits, [])

    def test_stripped_result_structure_no_leakage(self):
        self.searcher.hits = [{
            "memory_id": "mem_001",
            "text": "valid memory text",
            "domain": "real",
            "score": 0.88,
            "source_ids": ["src_01"],
            "owner_id": "owner_alpha",
            "visible_to": ["char_alpha"],
            "internal_secret": "confidential_vector_data",
        }]
        results = self.client.search(TOKEN_A, "test query", 5, "real")
        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0],
            {
                "id": "mem_001",
                "text": "valid memory text",
                "domain": "real",
                "score": 0.88,
                "source_ids": ["src_01"],
            },
        )
        self.assertNotIn("owner_id", results[0])
        self.assertNotIn("visible_to", results[0])
        self.assertNotIn("internal_secret", results[0])

    def test_owner_reader_domain_forged_results_rejected(self):
        base_hit = {
            "memory_id": "mem_001",
            "text": "memory text",
            "domain": "real",
            "score": 0.75,
            "source_ids": ["src_01"],
            "owner_id": "owner_alpha",
            "visible_to": ["char_alpha"],
        }

        # Forged owner
        forged_owner = dict(base_hit, owner_id="owner_forged")
        self.searcher.hits = [forged_owner]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 5, "real")

        # Forged reader (character_id not in visible_to)
        forged_reader = dict(base_hit, visible_to=["char_other"])
        self.searcher.hits = [forged_reader]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 5, "real")

        # Forged domain (requested "real", backend returned "roleplay")
        forged_domain = dict(base_hit, domain="roleplay")
        self.searcher.hits = [forged_domain]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 5, "real")

        # Backend returned unknown domain
        unknown_domain = dict(base_hit, domain="unsupported_domain")
        self.searcher.hits = [unknown_domain]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 5, None)

        # Backend returned unhashable or non-string domain ([], {}, set(), int)
        for bad_domain in [[], {}, set(), 123]:
            self.searcher.hits = [dict(base_hit, domain=bad_domain)]
            with self.subTest(bad_domain=bad_domain):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", 5, "real")

    def test_malformed_score_source_id_text_length_rejected(self):
        base_hit = {
            "memory_id": "mem_001",
            "text": "memory text",
            "domain": "real",
            "score": 0.75,
            "source_ids": ["src_01"],
            "owner_id": "owner_alpha",
            "visible_to": ["char_alpha"],
        }

        # Malformed score: bool, NaN, inf, str, enormous int (OverflowError)
        for bad_score in [True, False, float("nan"), float("inf"), "0.75", None, 10**10000, -10**10000]:
            self.searcher.hits = [dict(base_hit, score=bad_score)]
            with self.subTest(bad_score=bad_score):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", 5, "real")

        # Malformed source_ids: empty, non-list, non-str, empty str, whitespace, controls, str > 128
        for bad_src in [
            [], "src_01", [""], ["   "], ["src\x00id"], ["src\nid"], ["src\rid"],
            [123], ["x" * 129], ["x" * 257], None,
        ]:
            self.searcher.hits = [dict(base_hit, source_ids=bad_src)]
            with self.subTest(bad_src=bad_src):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", 5, "real")

        # Malformed id: empty str, whitespace only, controls, str > 128, non-str
        for bad_id in ["", "   ", "mem\x00id", "mem\nid", "mem\rid", 12345, "x" * 129, "x" * 257, None]:
            self.searcher.hits = [dict(base_hit, memory_id=bad_id)]
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", 5, "real")

        # Malformed text: empty str, whitespace only, controls, non-str
        for bad_text in ["", "   ", "\x00", "text\x00here", None, 123]:
            self.searcher.hits = [dict(base_hit, text=bad_text)]
            with self.subTest(bad_text=bad_text):
                with self.assertRaises(MemoryUnavailable):
                    self.client.search(TOKEN_A, "test query", 5, "real")

        # Exceeding limit
        self.searcher.hits = [
            dict(base_hit, memory_id=f"mem_{i}") for i in range(4)
        ]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 3, "real")

    def test_missing_memory_id_rejected(self):
        # Hit with fallback 'id' but missing canonical 'memory_id' must be rejected
        self.searcher.hits = [{
            "id": "mem_001",
            "text": "statement",
            "domain": "real",
            "score": 0.9,
            "source_ids": ["src_01"],
            "owner_id": "owner_alpha",
            "visible_to": ["char_alpha"],
        }]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 5, "real")

    def test_backend_exception_sanitized(self):
        self.searcher.error = RuntimeError("Database connection password=SECRET_DB_PWD failed")
        with self.assertRaises(MemoryUnavailable) as ctx:
            self.client.search(TOKEN_A, "test query", 5, "real")
        self.assertEqual(str(ctx.exception), "memory search unavailable")
        self.assertNotIn("SECRET_DB_PWD", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)

    def test_no_partial_success_on_out_of_scope_record(self):
        # 1st hit is valid, 2nd hit is out of scope (different owner)
        self.searcher.hits = [
            {
                "memory_id": "mem_001",
                "text": "first valid memory",
                "domain": "real",
                "score": 0.85,
                "source_ids": ["src_01"],
                "owner_id": "owner_alpha",
                "visible_to": ["char_alpha"],
            },
            {
                "memory_id": "mem_002",
                "text": "second invalid memory",
                "domain": "real",
                "score": 0.82,
                "source_ids": ["src_02"],
                "owner_id": "owner_forged",
                "visible_to": ["char_alpha"],
            },
        ]
        with self.assertRaises(MemoryUnavailable):
            self.client.search(TOKEN_A, "test query", 5, "real")

    def test_injection_with_existing_real_companion_chat(self):
        # Configure searcher with valid candidate memory hit
        self.searcher.hits = [{
            "memory_id": "mem_fictional_prior_01",
            "text": "fictional statement: user likes tea",
            "domain": "real",
            "score": 0.72,
            "source_ids": ["fictional_src_01"],
            "owner_id": "owner_alpha",
            "visible_to": ["char_alpha"],
        }]

        # Fake LLM recording prompt contexts and returning fictional placeholder
        recorded_contexts: List[List[Dict[str, str]]] = []

        def fake_llm(context: List[Dict[str, str]]) -> str:
            recorded_contexts.append(context)
            return "[FICTIONAL_AI_REPLY] I remember you like tea."

        chat_scope = ChatScope(
            token=TOKEN_A,
            owner_id="owner_alpha",
            character_id="char_alpha",
            domain="real",
        )
        chat = CompanionChat(
            scopes={1001: chat_scope},
            memory_client=self.client,
            llm=fake_llm,
            persona="You are a fictional test companion.",
        )

        update = DummyUpdate(1001, "What drink do I enjoy?")
        reply = chat.turn(update)

        # Verify fictional placeholder reply returned
        self.assertEqual(reply, "[FICTIONAL_AI_REPLY] I remember you like tea.")

        # Verify last_injected_ids updated with prior statement memory ID
        self.assertEqual(chat.last_injected_ids, ("mem_fictional_prior_01",))

        # Verify LLM context contains prior statement header and memory line
        self.assertEqual(len(recorded_contexts), 1)
        prompt = recorded_contexts[0]
        memory_blocks = [m["content"] for m in prompt if PRIOR_USER_HEADER in m.get("content", "")]
        self.assertEqual(len(memory_blocks), 1)
        self.assertIn("fictional statement: user likes tea", memory_blocks[0])


if __name__ == "__main__":
    unittest.main()
