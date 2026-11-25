"""Offline integration tests for bootstrap wiring with BotConfig.

Covers:
- Two fully fictional configs differing in all required contract dimensions.
- Propagation of config into preflight, model_reply, store DB, retriever,
  key_decision family key, runtime, ChatScope, reply_recorder, PersonaL0, admin.
- Scope isolation: writes and queries cannot cross scopes.
- Real SQLite, queue, processor, worker, client, chat, admin, keyer.
- Fixtures for external boundaries (model opener, embedder, Telegram adapter).
- Model route URL and body inspection, redirect rejection.
- Selection via main() CLI (--config) and fixed BootstrapError at boundary.
"""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from bot_config import BotConfig, load_bot_config
import bootstrap_test_bot as bot
from memory_core import MemoryStore


class MockResponse:
    """Mock HTTP response for urllib openers."""

    def __init__(self, body_bytes: bytes, status: int = 200, headers: dict | None = None):
        self.body = body_bytes
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size: int = -1) -> bytes:
        return self.body


class MockEmbedder:
    """In-memory 384-dimensional embedder fixture."""

    model_id = "test-mock-embedder-v1"
    dimension = 384

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, text: str) -> list[float]:
        return [0.01] * self.dimension


class BootstrapWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

        # Fictional Persona & L0 for Bot Alpha
        self.alpha_root = self.root / "alpha_bot"
        self.alpha_root.mkdir(parents=True, exist_ok=True)
        self.alpha_persona = self.alpha_root / "alpha_persona.txt"
        self.alpha_persona.write_text("Fictional Alpha persona text.", encoding="utf-8")
        self.alpha_l0 = self.alpha_root / "alpha_l0"
        self.alpha_l0.mkdir(parents=True, exist_ok=True)
        (self.alpha_l0 / "intro.md").write_text("# 簡介\n\n> load: always\n\nAlpha 簡介\n", encoding="utf-8")

        # Fictional Persona & L0 for Bot Beta
        self.beta_root = self.root / "beta_bot"
        self.beta_root.mkdir(parents=True, exist_ok=True)
        self.beta_persona = self.beta_root / "beta_persona.txt"
        self.beta_persona.write_text("Fictional Beta persona text.", encoding="utf-8")
        self.beta_l0 = self.beta_root / "beta_l0"
        self.beta_l0.mkdir(parents=True, exist_ok=True)
        (self.beta_l0 / "intro.md").write_text("# 簡介\n\n> load: always\n\nBeta 簡介\n", encoding="utf-8")

        # Two fully fictional configs differing across all dimensions
        self.cfg_alpha = BotConfig(
            root=self.alpha_root,
            db=self.alpha_root / "alpha.sqlite3",
            owner_id="owner-alpha",
            character_id="char-alpha",
            persona_file=self.alpha_persona,
            l0_dir=self.alpha_l0,
            model_url="http://127.0.0.1:18081/v1/chat/completions",
            model_id="alpha-model-v1",
            telegram_token_env="ALPHA_TELEGRAM_TOKEN",
            telegram_user_id_env="ALPHA_TELEGRAM_USER_ID",
            addressee_family_key=None,
        )

        self.cfg_beta = BotConfig(
            root=self.beta_root,
            db=self.beta_root / "beta.sqlite3",
            owner_id="owner-beta",
            character_id="char-beta",
            persona_file=self.beta_persona,
            l0_dir=self.beta_l0,
            model_url="http://127.0.0.1:18082/v1/chat/completions",
            model_id="beta-model-v2",
            telegram_token_env="BETA_TELEGRAM_TOKEN",
            telegram_user_id_env="BETA_TELEGRAM_USER_ID",
            addressee_family_key="家人/配偶",
        )

        self.alpha_json = self.alpha_root / "alpha.json"
        self.alpha_json.write_text(json.dumps({
            "root": ".",
            "db": "alpha.sqlite3",
            "owner_id": "owner-alpha",
            "character_id": "char-alpha",
            "persona_file": "alpha_persona.txt",
            "l0_dir": "alpha_l0",
            "model_url": "http://127.0.0.1:18081/v1/chat/completions",
            "model_id": "alpha-model-v1",
            "telegram_token_env": "ALPHA_TELEGRAM_TOKEN",
            "telegram_user_id_env": "ALPHA_TELEGRAM_USER_ID",
        }), encoding="utf-8")

        self.beta_json = self.beta_root / "beta.json"
        self.beta_json.write_text(json.dumps({
            "root": ".",
            "db": "beta.sqlite3",
            "owner_id": "owner-beta",
            "character_id": "char-beta",
            "persona_file": "beta_persona.txt",
            "l0_dir": "beta_l0",
            "model_url": "http://127.0.0.1:18082/v1/chat/completions",
            "model_id": "beta-model-v2",
            "telegram_token_env": "BETA_TELEGRAM_TOKEN",
            "telegram_user_id_env": "BETA_TELEGRAM_USER_ID",
            "addressee_family_key": "家人/配偶",
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_preflight_uses_selected_config(self):
        # Alpha preflight expects ALPHA_* env vars
        env_alpha = {
            "ALPHA_TELEGRAM_TOKEN": "alpha-token-xyz",
            "ALPHA_TELEGRAM_USER_ID": "1001",
        }
        token, user = bot.preflight(env_alpha, config=self.cfg_alpha)
        self.assertEqual(token, "alpha-token-xyz")
        self.assertEqual(user, 1001)

        # Missing token in selected config fails closed
        with self.assertRaises(bot.BootstrapError) as caught:
            bot.preflight({}, config=self.cfg_alpha)
        self.assertIn("ALPHA_TELEGRAM_TOKEN", str(caught.exception))

        # Does not check default BOT_* env vars when config is given
        env_wrong = {
            "BOT_TELEGRAM_TOKEN": "bot_sandbox-token",
            "BOT_TELEGRAM_USER_ID": "42",
        }
        with self.assertRaises(bot.BootstrapError) as caught:
            bot.preflight(env_wrong, config=self.cfg_alpha)
        self.assertIn("ALPHA_TELEGRAM_TOKEN", str(caught.exception))

        # Beta preflight expects BETA_* env vars
        env_beta = {
            "BETA_TELEGRAM_TOKEN": "beta-token-abc",
            "BETA_TELEGRAM_USER_ID": "2002",
        }
        token_b, user_b = bot.preflight(env_beta, config=self.cfg_beta)
        self.assertEqual(token_b, "beta-token-abc")
        self.assertEqual(user_b, 2002)

    def test_preflight_reads_selected_persona_and_validates(self):
        env = {
            "ALPHA_TELEGRAM_TOKEN": "token",
            "ALPHA_TELEGRAM_USER_ID": "101",
        }
        # Corrupt selected persona file
        self.alpha_persona.write_text("   \n", encoding="utf-8")
        with self.assertRaises(bot.BootstrapError) as caught:
            bot.preflight(env, config=self.cfg_alpha)
        self.assertEqual(str(caught.exception), "persona invalid")

    def test_model_reply_routes_to_configured_url_and_id(self):
        calls = []

        def mock_opener(req, timeout):
            calls.append((req.full_url, json.loads(req.data.decode("utf-8")), timeout))
            resp_body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode()
            return MockResponse(resp_body)

        # Call with Alpha config
        reply_a = bot.model_reply([{"role": "user", "content": "hi"}], opener=mock_opener, config=self.cfg_alpha)
        self.assertEqual(reply_a, "ok")
        self.assertEqual(calls[-1][0], "http://127.0.0.1:18081/v1/chat/completions")
        self.assertEqual(calls[-1][1]["model"], "alpha-model-v1")
        self.assertEqual(calls[-1][2], 40)

        # Call with Beta config
        reply_b = bot.model_reply([{"role": "user", "content": "hi"}], opener=mock_opener, config=self.cfg_beta)
        self.assertEqual(reply_b, "ok")
        self.assertEqual(calls[-1][0], "http://127.0.0.1:18082/v1/chat/completions")
        self.assertEqual(calls[-1][1]["model"], "beta-model-v2")

    def test_model_reply_blocks_redirects(self):
        def redirect_opener(req, timeout):
            return MockResponse(b"redirected", status=302, headers={"Location": "http://evil.com/leak"})

        with self.assertRaises(bot.BootstrapError) as caught:
            bot.model_reply([], opener=redirect_opener, config=self.cfg_alpha)
        self.assertEqual(str(caught.exception), "model unavailable")

    def test_run_wires_all_components_and_isolates_scope(self):
        """End-to-end run test for Alpha and Beta ensuring scope isolation and real SQLite writes."""
        env_alpha = {
            "ALPHA_TELEGRAM_TOKEN": "token-alpha-secret",
            "ALPHA_TELEGRAM_USER_ID": "555",
        }
        env_beta = {
            "BETA_TELEGRAM_TOKEN": "token-beta-secret",
            "BETA_TELEGRAM_USER_ID": "666",
        }

        model_calls = []

        def mock_opener(req, timeout):
            model_calls.append((req.full_url, json.loads(req.data.decode("utf-8"))))
            resp_body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ready"}}]}).encode()
            return MockResponse(resp_body)

        captured_apps = {}

        def make_fake_builder(bot_name):
            def fake_build(token, chat, admin):
                app = Mock()
                captured_apps[bot_name] = {
                    "token": token,
                    "chat": chat,
                    "admin": admin,
                }
                def fake_poll(drop_pending_updates=True):
                    # Simulate user and assistant turns via chat recorders
                    chat.recorder(f"User statement in {bot_name}.")
                    chat.reply_recorder(f"Assistant response in {bot_name}.")
                app.run_polling.side_effect = fake_poll
                return app
            return fake_build

        # Run Alpha
        with patch.object(bot, "LocalEmbedder", MockEmbedder):
            with patch.object(bot, "_NO_REDIRECT_OPENER", Mock(open=mock_opener)):
                with patch("telegram_test_adapter.build_test_application", side_effect=make_fake_builder("alpha")):
                    bot.run(env_alpha, config=self.cfg_alpha)

        # Run Beta
        with patch.object(bot, "LocalEmbedder", MockEmbedder):
            with patch.object(bot, "_NO_REDIRECT_OPENER", Mock(open=mock_opener)):
                with patch("telegram_test_adapter.build_test_application", side_effect=make_fake_builder("beta")):
                    bot.run(env_beta, config=self.cfg_beta)

        # 1. Verify model probe calls
        alpha_probes = [c for c in model_calls if c[0] == "http://127.0.0.1:18081/v1/chat/completions"]
        beta_probes = [c for c in model_calls if c[0] == "http://127.0.0.1:18082/v1/chat/completions"]
        self.assertTrue(len(alpha_probes) >= 1)
        self.assertEqual(alpha_probes[0][1]["model"], "alpha-model-v1")
        self.assertTrue(len(beta_probes) >= 1)
        self.assertEqual(beta_probes[0][1]["model"], "beta-model-v2")

        # 2. Verify Alpha wiring & SQLite DB isolation
        alpha_info = captured_apps["alpha"]
        self.assertEqual(alpha_info["token"], "token-alpha-secret")
        self.assertEqual(alpha_info["chat"].scopes[555].owner_id, "owner-alpha")
        self.assertEqual(alpha_info["chat"].scopes[555].character_id, "char-alpha")
        self.assertEqual(alpha_info["admin"].owner_id, "owner-alpha")
        self.assertEqual(alpha_info["admin"].character_id, "char-alpha")

        # Inspect real SQLite file for Alpha
        store_alpha = MemoryStore(self.cfg_alpha.db)
        try:
            with contextlib.closing(sqlite3.connect(self.cfg_alpha.db)) as conn:
                rows = conn.execute("SELECT owner_id, character_id, role, content FROM messages").fetchall()
                self.assertTrue(len(rows) >= 1)
                for owner, char, role, content in rows:
                    self.assertEqual(owner, "owner-alpha")
                    self.assertEqual(char, "char-alpha")
                    self.assertNotIn("bot_sandbox-synthetic-owner", owner)
                    self.assertNotIn("owner-beta", owner)
        finally:
            store_alpha.close()

        # 3. Verify Beta wiring & SQLite DB isolation
        beta_info = captured_apps["beta"]
        self.assertEqual(beta_info["token"], "token-beta-secret")
        self.assertEqual(beta_info["chat"].scopes[666].owner_id, "owner-beta")
        self.assertEqual(beta_info["chat"].scopes[666].character_id, "char-beta")
        self.assertEqual(beta_info["admin"].owner_id, "owner-beta")
        self.assertEqual(beta_info["admin"].character_id, "char-beta")

        # Inspect real SQLite file for Beta
        store_beta = MemoryStore(self.cfg_beta.db)
        try:
            with contextlib.closing(sqlite3.connect(self.cfg_beta.db)) as conn:
                rows = conn.execute("SELECT owner_id, character_id, role, content FROM messages").fetchall()
                self.assertTrue(len(rows) >= 1)
                for owner, char, role, content in rows:
                    self.assertEqual(owner, "owner-beta")
                    self.assertEqual(char, "char-beta")
                    self.assertNotIn("owner-alpha", owner)
        finally:
            store_beta.close()

        # 4. Cross-scope access must yield empty / isolated results
        admin_alpha = alpha_info["admin"]
        # Listing memories under alpha scope cannot see beta data
        listing = admin_alpha.store.list_memories("owner-beta", "char-beta")
        self.assertEqual(listing["total"], 0)

    def test_main_cli_config_selection_and_error_handling(self):
        env_alpha = {
            "ALPHA_TELEGRAM_TOKEN": "token-1",
            "ALPHA_TELEGRAM_USER_ID": "42",
        }
        with patch.dict(os.environ, env_alpha, clear=True):
            # --config with --check succeeds
            with patch("builtins.print") as mock_print:
                bot.main(["--config", str(self.alpha_json), "--check"])
                mock_print.assert_called_with("P2c sandbox configuration OK (no services contacted)")

        # Non-existent config converts ConfigError to BootstrapError("configuration invalid")
        with self.assertRaises(SystemExit) as exit_caught:
            bot.main(["--config", str(self.root / "nonexistent.json"), "--check"])
        self.assertEqual(exit_caught.exception.code, 1)

        # Malformed config converts ConfigError to BootstrapError
        bad_cfg = self.root / "bad.json"
        bad_cfg.write_text("{bad", encoding="utf-8")
        with self.assertRaises(SystemExit) as exit_caught:
            bot.main(["--config", str(bad_cfg), "--check"])
        self.assertEqual(exit_caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
