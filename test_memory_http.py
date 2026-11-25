import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from memory_http import Scope, create_server
from memory_core import MemoryStore
from memory_index import QdrantHTTPBackend
from memory_search import SemanticSearcher

A = 'a' * 40
B = 'b' * 40
C = 'c' * 40
D = 'd' * 40


class FakeSearcher:
    def __init__(self):
        self.calls = []
        self.fail = False

    def search(self, owner_id, character_id, query, limit, domain):
        self.calls.append((owner_id, character_id, query, limit, domain))
        if self.fail:
            raise RuntimeError('private backend detail')
        return [{'memory_id': 'memory-1', 'text': 'safe', 'domain': 'real', 'score': 0.8,
                 'source_ids': ['message-1'], 'owner_id': owner_id, 'visible_to': [character_id],
                 'internal_key': 'private-secret'}]


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.searcher = FakeSearcher()
        self.server = create_server(self.searcher, {A: Scope('alice', 'reader'), B: Scope('alice', 'other'),
                                                         C: Scope('bob', 'reader'), D: Scope('bob', 'disabled', False)})
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, token=A, body=None, content_type='application/json', headers=None):
        data = json.dumps({'query': 'hello'} if body is None else body).encode() if not isinstance(body, bytes) else body
        hdr = {'Content-Type': content_type}
        if token is not None:
            hdr['Authorization'] = 'Bearer ' + token
        hdr.update(headers or {})
        req = Request(f'http://127.0.0.1:{self.server.server_port}/memory/search', data=data, headers=hdr)
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, response.read().decode()
        except HTTPError as error:
            with error:
                return error.code, error.read().decode()

    def test_configuration(self):
        for config in ({}, {A[:31]: Scope('alice', 'reader')}, {A: Scope('', 'reader')},
                       {A: Scope('alice', 'reader'), B: Scope('alice', 'reader')}, {A: ('alice', 'reader')}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_server(self.searcher, config)
        self.assertEqual(self.server.server_address[0], '127.0.0.1')

    def test_auth_scopes_and_projection(self):
        for token, expected in ((A, ('alice', 'reader')), (B, ('alice', 'other')), (C, ('bob', 'reader'))):
            status, raw = self.request(token, {'query': 'hello', 'limit': 2, 'domain': 'real'})
            self.assertEqual(status, 200)
            self.assertEqual(self.searcher.calls[-1], (*expected, 'hello', 2, 'real'))
            self.assertEqual(json.loads(raw), {'results': [{'id': 'memory-1', 'text': 'safe',
                'domain': 'real', 'score': 0.8, 'source_ids': ['message-1']}]})
            self.assertNotIn('private-', raw)
        for token, expected in ((None, 401), ('wrong', 401), ('', 401), (D, 403)):
            self.assertEqual(self.request(token)[0], expected)
        self.assertEqual(self.request(A, headers={'Authorization': 'Basic x'})[0], 401)

    def test_reject_forgery_and_bad_input(self):
        invalid = [ {'query': 'q', 'owner_id': 'bob'}, {'query': 'q', 'character_id': 'other'},
                    {'query': 'q', 'session_id': 'x'}, {'query': 'q', 'limit': True},
                    {'query': 'x' * 1025}, {'query': 'q', 'domain': 'other'},
                    b'{"query":"q","query":"evil"}', b'{"query":', b'[]',
                    {'query': 'q', 'padding': 'x' * 5000}]
        for body in invalid:
            with self.subTest(body=str(body)[:30]):
                status, raw = self.request(A, body)
                self.assertIn(status, (400, 413))
                self.assertNotIn('evil', raw)
        self.assertEqual(self.request(A, content_type='text/plain')[0], 415)
        self.assertEqual(self.searcher.calls, [])

    def test_reject_out_of_scope_search_result(self):
        self.searcher.search = lambda *args: [{'memory_id': 'other', 'text': 'private secret',
            'domain': 'real', 'score': 0.8, 'source_ids': [], 'owner_id': 'bob',
            'visible_to': ['reader']}]
        status, raw = self.request(A)
        self.assertEqual(status, 503)
        self.assertNotIn('private secret', raw)

    def test_dependency_failure(self):
        self.searcher.fail = True
        status, raw = self.request(A, {'query': 'secret query'})
        self.assertEqual(status, 503)
        self.assertNotIn('secret', raw)
        self.assertNotIn('private', raw)


class CanonicalHTTPTests(HTTPTests):
    def setUp(self):
        self.store = MemoryStore(':memory:')
        source = self.store.append_message('alice', 'writer', 'msg', 'user', 'source')
        mid = self.store.save_memory('alice', 'writer', 'mem', 'fact', 'sqlite secret', [source], visible_to=['reader'])
        self.points = [{'id': mid, 'score': 0.8, 'payload': {'memory_id': mid, 'owner_id': 'alice',
            'creator_character_id': 'writer', 'visible_to': ['writer', 'reader'], 'domain': 'real',
            'embedding_model_id': 'test-model'}}]
        def transport(method, url, body, headers, timeout):
            return 200, json.dumps({'status': 'ok', 'result': {'points': self.points}}).encode()
        class Embedder:
            model_id = 'test-model'
            dimension = 2
            def embed(self, text):
                return [0.1, 0.2]
        self.searcher = SemanticSearcher(self.store, Embedder(), QdrantHTTPBackend(
            'http://localhost:6333', 'memories', 2, 'test-model', transport=transport))
        self.server = create_server(self.searcher, {A: Scope('alice', 'reader'), B: Scope('alice', 'other'),
                                                         C: Scope('bob', 'reader'), D: Scope('bob', 'disabled', False)})
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        super().tearDown()
        self.store.close()

    def test_sqlite_acl_through_http(self):
        self.assertEqual(json.loads(self.request(A)[1])['results'][0]['text'], 'sqlite secret')
        self.assertEqual(json.loads(self.request(B)[1])['results'], [])
        self.assertEqual(json.loads(self.request(C)[1])['results'], [])
        self.points[0]['payload']['owner_id'] = 'bob'
        self.assertEqual(json.loads(self.request(A)[1])['results'], [])

    # Fake-specific assertions belong only to HTTPTests.
    test_reject_out_of_scope_search_result = None
    test_auth_scopes_and_projection = None
    test_reject_forgery_and_bad_input = None
    test_dependency_failure = None
    test_configuration = None


if __name__ == '__main__':
    unittest.main()
