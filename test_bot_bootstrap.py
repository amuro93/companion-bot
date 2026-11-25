"""Offline-only P2c bootstrap tests: never call Telegram or real network."""
import io
import json
import unittest
from unittest.mock import patch

import bootstrap_test_bot as bot


class BootstrapTests(unittest.TestCase):
    def test_missing_credentials_fail_before_import_or_model(self):
        with patch.object(bot, "LocalEmbedder", side_effect=AssertionError("loaded")):
            for env in ({}, {"BOT_TELEGRAM_TOKEN": "secret"},
                        {"BOT_TELEGRAM_TOKEN": "secret", "BOT_TELEGRAM_USER_ID": "0"}):
                with self.assertRaises(bot.BootstrapError) as caught:
                    bot.run(env)
                self.assertNotIn("secret", str(caught.exception))

    def test_preflight_pure_and_rejects_invalid_ids(self):
        with patch.object(bot.urllib.request, "urlopen", side_effect=AssertionError("network")):
            with patch.object(bot, "LocalEmbedder", side_effect=AssertionError("model")):
                token, user = bot.preflight({"BOT_TELEGRAM_TOKEN": "fake-token",
                                             "BOT_TELEGRAM_USER_ID": "42"})
                self.assertEqual((token, user), ("fake-token", 42))
                for bad in ("-1", "0", "1.5", "١", " 4", ""):
                    with self.assertRaises(bot.BootstrapError):
                        bot.preflight({"BOT_TELEGRAM_TOKEN": "fake-token",
                                       "BOT_TELEGRAM_USER_ID": bad})
        self.assertEqual(bot.DB.parent, bot.ROOT)
        self.assertTrue(bot.COLLECTION.startswith("synthetic_"))

    def test_seed_idempotence_and_scope(self):
        class Store:
            def __init__(self):
                self.messages = {}
                self.memories = {}
            def append_message(self, owner, character, external, role, content):
                self.assert_scope(owner, character)
                self.messages.setdefault(external, external + "-id")
                return self.messages[external]
            def save_memory(self, owner, character, external, kind, text, sources, **kwargs):
                self.assert_scope(owner, character)
                assert sources == [self.messages[external.replace("memory", "source")]]
                assert kwargs == {"domain": "real", "visible_to": [bot.CHARACTER]}
                self.memories.setdefault(external, text)
            def assert_scope(self, owner, character):
                assert (owner, character) == (bot.OWNER, bot.CHARACTER)
        store = Store()
        bot.seed(store)
        bot.seed(store)
        self.assertEqual(len(store.messages), 2)
        self.assertEqual(len(store.memories), 2)

    def test_model_request_and_sanitized_errors(self):
        class Response:
            status = 200
            def __init__(self, body):
                self.body = body
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self, size):
                return self.body
        observed = {}
        def opener(req, timeout):
            observed.update(json.loads(req.data))
            self.assertEqual(req.full_url, bot.MODEL_URL)
            self.assertEqual(timeout, 40)
            return Response(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": " ready "}}]}).encode())
        self.assertEqual(bot.model_reply([{"role": "user", "content": "private"}], opener=opener), "ready")
        self.assertEqual(observed["model"], bot.MODEL_ID)
        self.assertEqual(observed["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(observed["max_tokens"], 512)
        self.assertFalse(observed["stream"])
        for failure in (lambda *a, **k: (_ for _ in ()).throw(RuntimeError("private token")),
                        lambda *a, **k: Response(b'{"choices":[{"finish_reason":"length","message":{"content":"secret"}}]}')):
            with self.assertRaises(bot.BootstrapError) as caught:
                bot.model_reply([], opener=failure)
            self.assertEqual(str(caught.exception), "model unavailable")


if __name__ == "__main__":
    unittest.main()
