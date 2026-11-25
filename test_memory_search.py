import json
import unittest

from memory_core import MemoryStore
from memory_index import QdrantHTTPBackend
from memory_search import SemanticSearcher, SemanticSearchError


class Embedder:
    model_id = "test-model"
    dimension = 2
    def embed(self, text):
        return [0.1, 0.2]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")
        source = self.store.append_message("alice", "a", "msg", "user", "source")
        self.mid = self.store.save_memory("alice", "a", "mem", "fact", "secret text", [source], visible_to=["b"])
        self.calls = []
        self.points = [self.point(self.mid)]
        self.status = 200
        def transport(method, url, body, headers, timeout):
            self.calls.append((method, url, json.loads(body)))
            return self.status, json.dumps({"status": "ok", "result": {"points": self.points}}).encode()
        self.backend = QdrantHTTPBackend("http://localhost:6333", "memories", 2, "test-model", transport=transport)
        self.searcher = SemanticSearcher(self.store, Embedder(), self.backend)

    def tearDown(self):
        self.store.close()

    def point(self, mid, **changes):
        payload = {"memory_id": mid, "owner_id": "alice", "creator_character_id": "a", "visible_to": ["a", "b"], "domain": "real", "embedding_model_id": "test-model"}
        payload.update(changes)
        return {"id": mid, "score": 0.8, "payload": payload}

    def test_request_shape_and_canonical_text(self):
        hits = self.searcher.search("alice", "b", "private query", domain="real")
        self.assertEqual(hits[0]["text"], "secret text")
        method, url, body = self.calls[0]
        self.assertEqual(method, "POST")
        self.assertTrue(url.endswith(f"/collections/{self.backend.collection_name}/points/query"))
        self.assertEqual(body["query"], [0.1, 0.2])
        self.assertEqual(body["limit"], 20)
        self.assertFalse(body["with_vector"])
        self.assertEqual(body["filter"]["must"], [
            {"key": "owner_id", "match": {"value": "alice"}},
            {"key": "visible_to", "match": {"value": "b"}},
            {"key": "embedding_model_id", "match": {"value": "test-model"}},
            {"key": "domain", "match": {"value": "real"}},
        ])
        self.assertNotIn("secret text", json.dumps(body))
        self.assertNotIn("private query", json.dumps(body))

    def test_scope_acl_and_domain(self):
        self.assertEqual(self.searcher.search("other", "b", "q"), [])
        self.assertEqual(self.searcher.search("alice", "c", "q"), [])
        self.assertEqual(self.searcher.search("alice", "b", "q", domain="roleplay"), [])
        self.points = [self.point(self.mid, owner_id="other")]
        self.assertEqual(self.searcher.search("alice", "b", "q"), [])
        self.points = [self.point(self.mid, visible_to=["a"])]
        self.assertEqual(self.searcher.search("alice", "b", "q"), [])

    def test_stale_correction_and_delete(self):
        source = self.store.append_message("alice", "a", "msg2", "user", "new")
        new_id = self.store.correct_memory("alice", "a", self.mid, "correct", "updated", [source])
        self.assertEqual(self.searcher.search("alice", "b", "q"), [])
        self.points = [self.point(new_id)]
        self.assertEqual(self.searcher.search("alice", "b", "q")[0]["text"], "updated")
        self.store.delete_memory("alice", "a", new_id)
        self.assertEqual(self.searcher.search("alice", "b", "q"), [])

    def test_fail_closed(self):
        for points in (None, [{}], [{"id": "bad", "score": 0.5, "payload": {}}],
                       [{"id": self.mid, "score": float("nan"), "payload": {}}]):
            self.points = points
            with self.assertRaises(SemanticSearchError):
                self.searcher.search("alice", "a", "q")
        self.status = 503
        with self.assertRaises(SemanticSearchError):
            self.searcher.search("alice", "a", "q")
        with self.assertRaises(SemanticSearchError):
            SemanticSearcher(self.store, None, self.backend).search("alice", "a", "q")

    def test_embedding_errors_validation_and_bounded_overfetch(self):
        class Broken(Embedder):
            def embed(self, text):
                raise RuntimeError("private query")
        with self.assertRaisesRegex(SemanticSearchError, "embedding failed") as caught:
            SemanticSearcher(self.store, Broken(), self.backend).search("alice", "a", "private query")
        self.assertNotIn("private query", str(caught.exception))
        for query, limit, domain in ((" ", 5, None), ("q", 0, None), ("q", 5, "bad")):
            with self.assertRaises(ValueError):
                self.searcher.search("alice", "a", query, limit, domain)
        self.searcher.search("alice", "a", "q", limit=1000)
        self.assertEqual(self.calls[-1][2]["limit"], 1000)


if __name__ == "__main__":
    unittest.main()
