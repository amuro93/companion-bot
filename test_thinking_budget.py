"""Thinking-budget payload tests for the isolated Rean chat endpoint.

The hybrid-reasoning model degrades when thinking is force-disabled, so bounded
thinking is the default. These tests pin the request payload contract only; they
never touch the network.
"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import bot_config
import bootstrap_test_bot as bot

BUDGET_URL = "http://llm-host:8081/v1/chat/completions"
THINK_MODEL = "qwen3.8-flash-next-iq3_xxs"


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body


class _Opener:
    def __init__(self):
        self.payloads = []

    def __call__(self, request, timeout=None):
        self.payloads.append(json.loads(request.data.decode("utf-8")))
        body = json.dumps({
            "choices": [{"message": {"content": "fictional reply"},
                         "finish_reason": "stop"}]
        }).encode("utf-8")
        return _Response(body)


def _config(model_url=BUDGET_URL, model_id=THINK_MODEL):
    return SimpleNamespace(model_url=model_url, model_id=model_id,
                           model_api_key_env=None)


class ThinkingBudgetPayloadTests(unittest.TestCase):
    def _send(self, env, config=None):
        opener = _Opener()
        with patch.object(bot_config, "APPROVED_REMOTE_MODEL_URL", BUDGET_URL):
            reply = bot.model_reply([{"role": "user", "content": "fictional prompt"}],
                                    opener=opener, config=config or _config(),
                                    env=env)
        self.assertEqual(reply, "fictional reply")
        self.assertEqual(len(opener.payloads), 1)
        return opener.payloads[0]

    def test_default_enables_bounded_thinking(self):
        payload = self._send({})
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(payload["reasoning_budget_tokens"], 128)
        self.assertEqual(payload["max_tokens"], 512)

    def test_explicit_budget_within_range(self):
        payload = self._send({"THINK_MODE": "128"})
        self.assertEqual(payload["reasoning_budget_tokens"], 128)
        self.assertNotIn("chat_template_kwargs", payload)

    def test_off_keeps_thinking_disabled(self):
        payload = self._send({"THINK_MODE": "off"})
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_budget_tokens", payload)

    def test_out_of_range_and_garbage_budgets_fail_closed(self):
        for mode in ("32", "49", "151", "0", "-64", "abc", "64.5", ""):
            with self.subTest(mode=mode):
                with self.assertRaises(bot.BootstrapError):
                    self._send({"THINK_MODE": mode})

    def test_other_endpoints_keep_thinking_disabled(self):
        config = _config(model_url="http://127.0.0.1:18081/v1/chat/completions",
                         model_id="other-model")
        payload = self._send({}, config)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_budget_tokens", payload)

    def test_other_model_id_keeps_thinking_disabled(self):
        config = _config(model_id="some-other-model")
        payload = self._send({"THINK_MODE": "128"}, config)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_budget_tokens", payload)

    def test_second_gpu_endpoint_uses_budget(self):
        config = _config(model_url="http://gpu-host:8080/v1/chat/completions",
                         model_id="qwen3.8-flash-next")
        payload = self._send({}, config)
        self.assertEqual(payload["reasoning_budget_tokens"], 128)
        self.assertNotIn("chat_template_kwargs", payload)

    def test_gpu_endpoint_with_other_model_id_keeps_thinking_disabled(self):
        config = _config(model_url="http://gpu-host:8080/v1/chat/completions",
                         model_id="some-other-model")
        payload = self._send({}, config)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_budget_tokens", payload)


class _SequenceOpener(_Opener):
    """Returns a fixed sequence of response bodies, then repeats the last one."""

    def __init__(self, bodies):
        super().__init__()
        self.bodies = list(bodies)

    def __call__(self, request, timeout=None):
        self.payloads.append(json.loads(request.data.decode("utf-8")))
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return _Response(body)


def _choice(content, finish="stop"):
    return json.dumps({"choices": [{"message": {"content": content},
                                    "finish_reason": finish}]}).encode("utf-8")


class EmptyCompletionRetryTests(unittest.TestCase):
    def _reply(self, opener):
        with patch.object(bot_config, "APPROVED_REMOTE_MODEL_URL", BUDGET_URL), \
                patch.object(bot.time, "sleep", lambda *_: None):
            return bot.model_reply([{"role": "user", "content": "fictional prompt"}],
                                   opener=opener, config=_config(), env={})

    def test_empty_content_is_retried_once(self):
        opener = _SequenceOpener([_choice(""), _choice("second try")])
        self.assertEqual(self._reply(opener), "second try")
        self.assertEqual(len(opener.payloads), 2)

    def test_non_stop_finish_is_retried_once(self):
        opener = _SequenceOpener([_choice("cut off", "length"), _choice("complete")])
        self.assertEqual(self._reply(opener), "complete")
        self.assertEqual(len(opener.payloads), 2)

    def test_two_empty_completions_fail_closed(self):
        opener = _SequenceOpener([_choice(""), _choice("")])
        with self.assertRaises(bot.BootstrapError):
            self._reply(opener)
        self.assertEqual(len(opener.payloads), 2)

    def test_retry_payload_keeps_the_same_budget(self):
        opener = _SequenceOpener([_choice(""), _choice("ok")])
        self._reply(opener)
        self.assertEqual([p.get("reasoning_budget_tokens") for p in opener.payloads],
                         [128, 128])


if __name__ == "__main__":
    unittest.main()
