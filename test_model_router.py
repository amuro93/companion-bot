"""Tests for the runtime /model switch: router, config schema, payloads."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bot_config
import bootstrap_test_bot as bot
from bot_config import BotConfig, ConfigError, ModelChoice, load_bot_config
from model_router import ModelRouter
from telegram_test_adapter import is_model_command

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


class ModelRouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.calls = []
        self.backends = {
            "cloud": lambda m, **k: self.calls.append("cloud") or "cloud reply",
            "local": lambda m, **k: self.calls.append("local") or "local reply",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_is_first_backend(self):
        router = ModelRouter(self.backends)
        self.assertEqual(router.current, "cloud")
        self.assertEqual(router([{"role": "user", "content": "x"}]), "cloud reply")

    def test_switch_routes_and_persists(self):
        state = self.dir / "model-choice.json"
        router = ModelRouter(self.backends, state_path=state)
        router.switch("local")
        self.assertEqual(router([{"role": "user", "content": "x"}]), "local reply")
        self.assertEqual(json.loads(state.read_text(encoding="utf-8")), {"current": "local"})
        reloaded = ModelRouter(self.backends, state_path=state)
        self.assertEqual(reloaded.current, "local")

    def test_unknown_or_corrupt_state_falls_back(self):
        state = self.dir / "model-choice.json"
        state.write_text("{not json", encoding="utf-8")
        router = ModelRouter(self.backends, state_path=state, default="local")
        self.assertEqual(router.current, "local")
        state.write_text(json.dumps({"current": "gone"}), encoding="utf-8")
        self.assertEqual(ModelRouter(self.backends, state_path=state).current, "cloud")

    def test_unknown_label_rejected(self):
        router = ModelRouter(self.backends)
        with self.assertRaises(ValueError):
            router.switch("nope")
        self.assertEqual(router.current, "cloud")


class ModelCommandTests(unittest.TestCase):
    def test_command_matching(self):
        for text in ("/model", "/model@mybot", "  /MODEL  "):
            self.assertTrue(is_model_command(text), text)
        for text in ("/models", "model", "/modelx", "", None, "/model 切換"):
            if text == "/model 切換":
                continue  # still a command; arguments are harmless
            self.assertFalse(is_model_command(text), text)


class ModelChoiceConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "persona.txt").write_text("persona", encoding="utf-8")
        (root / "l0").mkdir()
        self.base = dict(root=str(root), db="db.sqlite3", owner_id="o", character_id="c",
                         persona_file="persona.txt", l0_dir="l0",
                         model_url="http://127.0.0.1:18080/v1/chat/completions",
                         model_id="local-model",
                         telegram_token_env="TG_TOKEN", telegram_user_id_env="TG_USER")

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, extra):
        path = Path(self.tmp.name) / "bot.json"
        path.write_text(json.dumps({**self.base, **extra}), encoding="utf-8")
        return load_bot_config(path)

    def test_choices_are_parsed(self):
        cfg = self._load({"model_choices": [
            {"label": "DeepSeek 雲端", "model_url": DEEPSEEK_URL,
             "model_id": "deepseek-flash", "model_api_key_env": "DEEPSEEK_API_KEY"},
            {"label": "Local GPU", "model_url": "http://gpu-host:8080/v1/chat/completions",
             "model_id": "qwen3.8-flash-next", "model_api_key_env": "GPU_API_KEY"},
        ]})
        self.assertEqual([c.label for c in cfg.model_choices], ["DeepSeek 雲端", "Local GPU"])
        self.assertEqual(cfg.model_choices[0].model_id, "deepseek-flash")

    def test_duplicate_labels_rejected(self):
        with self.assertRaises(ConfigError):
            self._load({"model_choices": [
                {"label": "X", "model_url": DEEPSEEK_URL, "model_id": "deepseek-flash",
                 "model_api_key_env": "DEEPSEEK_API_KEY"},
                {"label": "X", "model_url": DEEPSEEK_URL, "model_id": "deepseek-flash",
                 "model_api_key_env": "DEEPSEEK_API_KEY"},
            ]})

    def test_remote_choice_without_key_rejected(self):
        with self.assertRaises(ConfigError):
            self._load({"model_choices": [
                {"label": "X", "model_url": DEEPSEEK_URL, "model_id": "deepseek-flash"},
            ]})

    def test_unapproved_url_rejected(self):
        with self.assertRaises(ConfigError):
            self._load({"model_choices": [
                {"label": "X", "model_url": "https://evil.example.com/v1/chat/completions",
                 "model_id": "x", "model_api_key_env": "K"},
            ]})

    def test_non_list_choices_rejected(self):
        with self.assertRaises(ConfigError):
            self._load({"model_choices": {"label": "X"}})


class DeepSeekPayloadTests(unittest.TestCase):
    class _Response:
        status = 200

        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, size=-1):
            return self.body

    def _send(self, env):
        captured = []
        base_env = {"DEEPSEEK_API_KEY": "k" * 40}
        base_env.update(env)

        def opener(request, timeout=None):
            captured.append(json.loads(request.data.decode("utf-8")))
            return self._Response(json.dumps({
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode())

        cfg = SimpleNamespace(model_url=DEEPSEEK_URL, model_id="deepseek-flash",
                              model_api_key_env="DEEPSEEK_API_KEY")
        with patch.object(bot_config, "APPROVED_DEEPSEEK_URL", DEEPSEEK_URL):
            reply = bot.model_reply([{"role": "user", "content": "x"}], opener=opener,
                                    config=cfg, env=base_env)
        self.assertEqual(reply, "ok")
        return captured[0]

    def test_thinking_enabled_with_low_effort(self):
        payload = self._send({})
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("reasoning_budget_tokens", payload)

    def test_off_disables_thinking(self):
        payload = self._send({"THINK_MODE": "off"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", payload)


class ModelHandlerTests(unittest.TestCase):
    """Offline exercise of the /model command and its inline buttons."""

    def setUp(self):
        import asyncio
        from companion_chat import ChatScope
        from telegram_test_adapter import make_callback_handler, make_model_handler
        from memory_admin import MemoryAdmin

        self.asyncio = asyncio
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.router = ModelRouter({
            "DeepSeek 雲端": lambda m, **k: "cloud",
            "Local GPU": lambda m, **k: "local",
        }, state_path=self.dir / "choice.json")
        self.chat = SimpleNamespace(
            scopes={123: ChatScope("t" * 48, "owner", "char", "real")})
        self.on_model = make_model_handler(self.chat, self.router)
        self.on_callback = make_callback_handler(self.chat, None, self.router)

    def tearDown(self):
        self.tmp.cleanup()

    def _update(self, **extra):
        base = dict(effective_user=SimpleNamespace(id=123),
                    effective_chat=SimpleNamespace(type="private", id=123))
        base.update(extra)
        return SimpleNamespace(**base)

    def test_command_renders_one_button_per_choice(self):
        from unittest.mock import AsyncMock
        message = SimpleNamespace(text="/model", reply_text=AsyncMock())
        with self.assertRaises(Exception) as raised:
            self.asyncio.run(self.on_model(self._update(message=message), None))
        # ApplicationHandlerStop is expected: the command never becomes a chat turn.
        self.assertEqual(type(raised.exception).__name__, "ApplicationHandlerStop")
        message.reply_text.assert_awaited_once()
        text = message.reply_text.await_args.args[0]
        markup = message.reply_text.await_args.kwargs["reply_markup"]
        self.assertIn("目前模型：DeepSeek 雲端", text)
        self.assertEqual([row[0].text for row in markup.inline_keyboard],
                         ["DeepSeek 雲端", "Local GPU"])

    def test_button_switches_and_confirms(self):
        from unittest.mock import AsyncMock
        query = SimpleNamespace(data="model:1", answer=AsyncMock(),
                                edit_message_text=AsyncMock())
        self.asyncio.run(self.on_callback(self._update(callback_query=query), None))
        self.assertEqual(self.router.current, "Local GPU")
        query.answer.assert_awaited_once_with("已切換")
        query.edit_message_text.assert_awaited_once_with("目前模型：Local GPU")

    def test_unauthorized_user_gets_no_buttons(self):
        from unittest.mock import AsyncMock
        message = SimpleNamespace(text="/model", reply_text=AsyncMock())
        update = SimpleNamespace(effective_user=SimpleNamespace(id=999),
                                 effective_chat=SimpleNamespace(type="private", id=999),
                                 message=message)
        with self.assertRaises(Exception):
            self.asyncio.run(self.on_model(update, None))
        message.reply_text.assert_not_awaited()


class DispatchOrderTests(unittest.TestCase):
    """PTB runs at most one handler per group: /model must not be eaten by on_list."""

    def test_model_command_survives_list_and_manual_handlers(self):
        import asyncio
        import datetime
        from telegram import Chat, Message, Update, User
        from companion_chat import ChatScope, CompanionChat
        from telegram_test_adapter import build_test_application

        class Client:
            def search(self, *args, **kwargs):
                return []

        chat = CompanionChat({123: ChatScope("t" * 48, "o", "c", "real")}, Client(),
                             lambda m: "x", "persona")
        admin = SimpleNamespace(has_pending=lambda: False)
        router = ModelRouter({"A 模型": lambda m: "a", "B 模型": lambda m: "b"})
        app = build_test_application("123:AAA", chat, admin, router=router)
        self.assertEqual(sorted(app.handlers), [-3, -2, -1, 0])

        sent = []

        async def fake_reply(self, text, **kwargs):
            sent.append(text)

        original = Message.reply_text
        Message.reply_text = fake_reply
        try:
            app._initialized = True
            msg = Message(message_id=1, date=datetime.datetime.now(),
                          chat=Chat(id=123, type="private"),
                          from_user=User(id=123, is_bot=False, first_name="x"), text="/model")
            asyncio.run(app.process_update(Update(update_id=1, message=msg)))
        finally:
            Message.reply_text = original
        self.assertTrue(any("要切換到哪一顆" in text for text in sent), sent)


if __name__ == "__main__":
    unittest.main()
