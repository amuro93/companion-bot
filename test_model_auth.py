"""Comprehensive tests for OpenAI-compatible model authentication and remote endpoint security.

Covers:
- BotConfig schema and strict validation (direct constructor and load_bot_config).
- Positional argument compatibility and default unauthenticated loopback.
- Approved remote endpoint (http://llm-host:8080/v1/chat/completions) requiring auth.
- Rejection of all remote URL variants (case, encoding, port, scheme, IP literals, FQDNs, query, fragment).
- Strict grammar and collision rejection for model_api_key_env.
- Missing/invalid/malformed key validation in preflight and model_reply before network.
- Explicit empty env mapping preventing leaked fallback to os.environ.
- Content-free exception boundaries and traceback sanitization.
- Real controlled loopback ThreadingHTTPServer on port 0 with synthetic keys only.
- Valid key 200, wrong key 401, missing key rejection before network.
- Deterministic server teardown and connection closure.
- HTTP redirect rejection for 301, 302, 303, 307, 308 (no second endpoint hit, no auth leakage).
- Full pipeline run() wiring auth in readiness, chooser, and chat with real SQLite storage.
"""
from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from bot_config import (
    APPROVED_REMOTE_MODEL_URL,
    BotConfig,
    ConfigError,
    is_remote_model_url,
    load_bot_config,
    validate_model_url,
)
import bootstrap_test_bot as bot
from companion_chat import CompanionChat
import memory_core
import telegram_test_adapter

SYNTHETIC_TEST_KEY = "SYNTH_AUTH_KEY_ALPHA_9876543210_XYZ"
SYNTHETIC_WRONG_KEY = "SYNTH_AUTH_KEY_WRONG_0000000000_ABC"
CANARY_SECRET = "SUPER_SECRET_CANARY_DO_NOT_LEAK_INTO_LOGS"


class MockEmbedder:
    """Fixture embedder matching dimension requirements."""

    model_id = "synthetic-model-auth-embedder"
    dimension = 384

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, text: str) -> list[float]:
        return [0.01] * self.dimension


class ControlledAuthHandler(BaseHTTPRequestHandler):
    """Loopback HTTP handler with configurable auth and redirect behaviors."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Suppress stdlib HTTP logging
        pass

    def do_POST(self):
        self.server.request_count += 1
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        self.server.recorded_requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        })

        if self.path == "/leak_dest":
            self.server.leak_hit_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'{"status":"leaked"}')
            return

        if self.server.forced_redirect_code is not None:
            code = self.server.forced_redirect_code
            self.send_response(code)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/leak_dest")
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return

        if self.path == "/v1/chat/completions":
            auth_header = self.headers.get("Authorization")
            expected_auth = f"Bearer {SYNTHETIC_TEST_KEY}"

            if self.server.require_auth:
                if auth_header != expected_auth:
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(b'{"error":{"message":"unauthorized"}}')
                    return

            resp_payload = json.dumps({
                "choices": [{
                    "finish_reason": "stop",
                    "message": {
                        "content": self.server.custom_response_content or "synthetic loopback response",
                    },
                }],
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(resp_payload)
            return

        self.send_response(404)
        self.send_header("Connection", "close")
        self.end_headers()


class ModelAuthConfigTests(unittest.TestCase):
    """Offline unit tests for BotConfig and load_bot_config authentication fields."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.subroot = self.root / "subroot"
        self.subroot.mkdir(parents=True, exist_ok=True)
        self.persona_file = self.subroot / "persona.txt"
        self.persona_file.write_text("Persona content.", encoding="utf-8")
        self.l0_dir = self.subroot / "l0"
        self.l0_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _valid_dict(self):
        return {
            "root": "subroot",
            "db": "test.sqlite3",
            "owner_id": "test-owner",
            "character_id": "test-char",
            "persona_file": "persona.txt",
            "l0_dir": "l0",
            "model_url": "http://127.0.0.1:18080/v1/chat/completions",
            "model_id": "test-model-v1",
            "telegram_token_env": "TEST_BOT_TOKEN",
            "telegram_user_id_env": "TEST_USER_ID",
            "addressee_family_key": None,
            "model_api_key_env": None,
        }

    def _write_json(self, data, filename="bot.json"):
        cfg_file = self.root / filename
        if isinstance(data, str):
            cfg_file.write_text(data, encoding="utf-8")
        else:
            cfg_file.write_text(json.dumps(data), encoding="utf-8")
        return cfg_file

    def test_positional_arguments_compatibility(self):
        # 10 positional arguments (preserves legacy unauthenticated loopback)
        cfg10 = BotConfig(
            self.subroot,
            self.subroot / "db.sqlite3",
            "owner",
            "char",
            self.persona_file,
            self.l0_dir,
            "http://127.0.0.1:18080/v1/chat/completions",
            "model",
            "TOKEN_ENV",
            "USER_ENV",
        )
        self.assertIsNone(cfg10.addressee_family_key)
        self.assertIsNone(cfg10.model_api_key_env)

        # 11 positional arguments
        cfg11 = BotConfig(
            self.subroot,
            self.subroot / "db.sqlite3",
            "owner",
            "char",
            self.persona_file,
            self.l0_dir,
            "http://127.0.0.1:18080/v1/chat/completions",
            "model",
            "TOKEN_ENV",
            "USER_ENV",
            "家人/配偶",
        )
        self.assertEqual(cfg11.addressee_family_key, "家人/配偶")
        self.assertIsNone(cfg11.model_api_key_env)

        # 12 positional arguments
        cfg12 = BotConfig(
            self.subroot,
            self.subroot / "db.sqlite3",
            "owner",
            "char",
            self.persona_file,
            self.l0_dir,
            "http://127.0.0.1:18080/v1/chat/completions",
            "model",
            "TOKEN_ENV",
            "USER_ENV",
            "家人/配偶",
            "MODEL_AUTH_KEY_ENV",
        )
        self.assertEqual(cfg12.addressee_family_key, "家人/配偶")
        self.assertEqual(cfg12.model_api_key_env, "MODEL_AUTH_KEY_ENV")

    def test_remote_endpoint_requires_model_api_key_env(self):
        # Direct constructor without model_api_key_env fails closed
        with self.assertRaises(ConfigError):
            BotConfig(
                root=self.subroot,
                db=self.subroot / "db.sqlite3",
                owner_id="owner",
                character_id="char",
                persona_file=self.persona_file,
                l0_dir=self.l0_dir,
                model_url=APPROVED_REMOTE_MODEL_URL,
                model_id="remote-model-v1",
                telegram_token_env="TOKEN_ENV",
                telegram_user_id_env="USER_ENV",
                model_api_key_env=None,
            )

        # Direct constructor with valid model_api_key_env succeeds
        cfg = BotConfig(
            root=self.subroot,
            db=self.subroot / "db.sqlite3",
            owner_id="owner",
            character_id="char",
            persona_file=self.persona_file,
            l0_dir=self.l0_dir,
            model_url=APPROVED_REMOTE_MODEL_URL,
            model_id="remote-model-v1",
            telegram_token_env="TOKEN_ENV",
            telegram_user_id_env="USER_ENV",
            model_api_key_env="MODEL_API_KEY",
        )
        self.assertEqual(cfg.model_url, APPROVED_REMOTE_MODEL_URL)
        self.assertEqual(cfg.model_api_key_env, "MODEL_API_KEY")

        # JSON loader without model_api_key_env fails closed
        data_no_auth = self._valid_dict()
        data_no_auth["model_url"] = APPROVED_REMOTE_MODEL_URL
        del data_no_auth["model_api_key_env"]
        p1 = self._write_json(data_no_auth, "remote_no_auth.json")
        with self.assertRaises(ConfigError):
            load_bot_config(p1)

        # JSON loader with explicit null model_api_key_env fails closed
        data_null_auth = self._valid_dict()
        data_null_auth["model_url"] = APPROVED_REMOTE_MODEL_URL
        data_null_auth["model_api_key_env"] = None
        p2 = self._write_json(data_null_auth, "remote_null_auth.json")
        with self.assertRaises(ConfigError):
            load_bot_config(p2)

        # JSON loader with valid model_api_key_env succeeds
        data_valid = self._valid_dict()
        data_valid["model_url"] = APPROVED_REMOTE_MODEL_URL
        data_valid["model_api_key_env"] = "MODEL_API_KEY"
        p3 = self._write_json(data_valid, "remote_valid.json")
        cfg_loaded = load_bot_config(p3)
        self.assertEqual(cfg_loaded.model_url, APPROVED_REMOTE_MODEL_URL)
        self.assertEqual(cfg_loaded.model_api_key_env, "MODEL_API_KEY")

    def test_remote_url_rejected_variants(self):
        rejected_variants = [
            "http://LLM-HOST:8080/v1/chat/completions",
            "HTTP://llm-host:8080/v1/chat/completions",
            "http://llm-host:8080/v1/chat/completions/",
            "http://llm-host:8080/v1/models",
            "http://llm-host:8080/health",
            "http://llm-host:8080/",
            "http://llm-host:8080",
            "https://llm-host:8080/v1/chat/completions",
            "http://llm-host:8081/v1/chat/completions",
            "http://llm-host/v1/chat/completions",
            "http://llm-host.local:8080/v1/chat/completions",
            "http://llm-host.lan:8080/v1/chat/completions",
            "http://example.com:8080/v1/chat/completions",
            "http://192.168.1.100:8080/v1/chat/completions",
            "http://100.64.0.1:8080/v1/chat/completions",
            "http://user:pass@llm-host:8080/v1/chat/completions",
            "http://llm-host:8080/v1/chat/completions?stream=false",
            "http://llm-host:8080/v1/chat/completions#fragment",
            "http://sub.llm-host:8080/v1/chat/completions",
            "http://%64rizztmacbook-pro:8080/v1/chat/completions",
        ]
        for url in rejected_variants:
            with self.subTest(url=url):
                with self.assertRaises(ConfigError):
                    validate_model_url(url)
                data = self._valid_dict()
                data["model_url"] = url
                data["model_api_key_env"] = "MODEL_API_KEY"
                f = self._write_json(data, "bad_remote.json")
                with self.assertRaises(ConfigError):
                    load_bot_config(f)

    def test_model_api_key_env_grammar_and_distinctness(self):
        bad_names = [
            "",
            "   ",
            "1BAD",
            "BAD-NAME",
            "BAD NAME",
            "BAD.NAME",
            "KEY\n",
            "KEY\x00",
            "x" * 129,
        ]
        for bad in bad_names:
            with self.subTest(bad=bad):
                data = self._valid_dict()
                data["model_api_key_env"] = bad
                f = self._write_json(data, "bad_name.json")
                with self.assertRaises(ConfigError):
                    load_bot_config(f)

        # Collision with telegram_token_env (case-insensitive)
        data_token_col = self._valid_dict()
        data_token_col["telegram_token_env"] = "SHARED_SECRET_ENV"
        data_token_col["model_api_key_env"] = "shared_secret_env"
        f_token = self._write_json(data_token_col, "token_collision.json")
        with self.assertRaises(ConfigError):
            load_bot_config(f_token)

        # Collision with telegram_user_id_env (case-insensitive)
        data_user_col = self._valid_dict()
        data_user_col["telegram_user_id_env"] = "USER_ID_ENV"
        data_user_col["model_api_key_env"] = "user_id_env"
        f_user = self._write_json(data_user_col, "user_collision.json")
        with self.assertRaises(ConfigError):
            load_bot_config(f_user)

    def test_loader_type_validation_and_no_side_effects(self):
        for bad_type in [123, True, ["key"], {"env": "key"}]:
            with self.subTest(bad_type=type(bad_type)):
                data = self._valid_dict()
                data["model_api_key_env"] = bad_type
                f = self._write_json(data, "bad_type.json")
                with self.assertRaises(ConfigError):
                    load_bot_config(f)

        # Ensure loading doesn't create database or persona files
        data = self._valid_dict()
        data["db"] = "uncreated_db.sqlite3"
        f = self._write_json(data, "side_effect.json")
        cfg = load_bot_config(f)
        self.assertFalse(cfg.db.exists())


class ModelAuthGuardsAndSanitizationTests(unittest.TestCase):
    """Verifies fail-closed preflight and model_reply guards before any network I/O."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.persona_file = self.root / "persona.txt"
        self.persona_file.write_text("Persona text.", encoding="utf-8")
        self.l0_dir = self.root / "l0"
        self.l0_dir.mkdir(parents=True, exist_ok=True)
        self.cfg_auth = BotConfig(
            root=self.root,
            db=self.root / "auth.sqlite3",
            owner_id="owner-auth",
            character_id="char-auth",
            persona_file=self.persona_file,
            l0_dir=self.l0_dir,
            model_url=APPROVED_REMOTE_MODEL_URL,
            model_id="remote-v1",
            telegram_token_env="AUTH_TOKEN",
            telegram_user_id_env="AUTH_USER",
            model_api_key_env="MODEL_API_KEY",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_preflight_missing_or_invalid_model_key_fails_closed_no_network(self):
        base_env = {"AUTH_TOKEN": "valid-token", "AUTH_USER": "12345"}
        bad_keys = [
            None,                       # missing
            "",                         # empty
            "   ",                      # whitespace-only
            " leading_space",           # leading whitespace
            "trailing_space ",          # trailing whitespace
            "inner space",              # inner whitespace
            "key\twith\ttab",           # tab
            "key\nwith\nnewline",       # newline
            "金鑰密碼",                  # non-ASCII
            "key\x00null",              # NUL control
            "key\x7fdel",               # DEL control
            "x" * 4097,                 # overlong (>4096)
            12345,                      # non-string
        ]

        for bad in bad_keys:
            with self.subTest(bad=repr(bad)):
                test_env = dict(base_env)
                if bad is not None:
                    test_env["MODEL_API_KEY"] = bad

                with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")) as net:
                    with self.assertRaises(bot.BootstrapError) as caught:
                        bot.preflight(test_env, config=self.cfg_auth)
                    self.assertEqual(net.call_count, 0)
                    self.assertNotIn(CANARY_SECRET, str(caught.exception))

    def test_preflight_empty_env_mapping_does_not_fallback_to_os_environ(self):
        # os.environ contains valid keys, but explicit mapping env={} is passed
        with patch.dict(os.environ, {
            "AUTH_TOKEN": "valid-token",
            "AUTH_USER": "12345",
            "MODEL_API_KEY": "valid-key-in-global-environ",
        }, clear=True):
            with self.assertRaises(bot.BootstrapError):
                bot.preflight({}, config=self.cfg_auth)

    def test_model_reply_missing_or_invalid_model_key_fails_closed_no_network(self):
        bad_keys = [
            None,
            "",
            "   ",
            " leading",
            "trailing ",
            "key\n",
            "key\x00",
            "非ASCII",
            "x" * 4097,
            12345,
        ]

        for bad in bad_keys:
            with self.subTest(bad=repr(bad)):
                test_env = {}
                if bad is not None:
                    test_env["MODEL_API_KEY"] = bad

                with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")) as net:
                    with self.assertRaises(bot.BootstrapError) as caught:
                        bot.model_reply(
                            [{"role": "user", "content": "hi"}],
                            config=self.cfg_auth,
                            env=test_env,
                        )
                    self.assertEqual(net.call_count, 0)
                    self.assertEqual(str(caught.exception), "model unavailable")
                    self.assertIsNone(caught.exception.__cause__)

    def test_model_reply_empty_env_does_not_fallback_to_os_environ(self):
        with patch.dict(os.environ, {"MODEL_API_KEY": "global-valid-key"}, clear=True):
            with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")) as net:
                with self.assertRaises(bot.BootstrapError) as caught:
                    bot.model_reply(
                        [{"role": "user", "content": "hi"}],
                        config=self.cfg_auth,
                        env={},
                    )
                self.assertEqual(net.call_count, 0)
                self.assertEqual(str(caught.exception), "model unavailable")

    def test_model_reply_traceback_markers_and_exception_sanitization(self):
        canary = f"canary-key-{CANARY_SECRET}"
        test_env = {"MODEL_API_KEY": f"{canary}\n"}

        try:
            bot.model_reply([{"role": "user", "content": "hi"}], config=self.cfg_auth, env=test_env)
        except bot.BootstrapError as exc:
            formatted = "".join(traceback.format_exception(exc))
            self.assertNotIn(CANARY_SECRET, formatted)
            self.assertNotIn(canary, formatted)
            self.assertEqual(str(exc), "model unavailable")
        else:
            self.fail("expected BootstrapError")

    def test_unauth_loopback_does_not_attach_auth_header(self):
        cfg_unauth = BotConfig(
            root=self.root,
            db=self.root / "unauth.sqlite3",
            owner_id="owner",
            character_id="char",
            persona_file=self.persona_file,
            l0_dir=self.l0_dir,
            model_url="http://127.0.0.1:18080/v1/chat/completions",
            model_id="mid",
            telegram_token_env="TOKEN",
            telegram_user_id_env="USER",
            model_api_key_env=None,
        )

        captured_requests = []

        def mock_opener(req, timeout):
            captured_requests.append(req)
            resp = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode()
            mock_resp = Mock()
            mock_resp.status = 200
            mock_resp.read.return_value = resp
            mock_resp.__enter__ = Mock(return_value=mock_resp)
            mock_resp.__exit__ = Mock(return_value=None)
            return mock_resp

        with patch.dict(os.environ, {"SOME_API_KEY": "leak-attempt"}, clear=True):
            res = bot.model_reply([{"role": "user", "content": "hi"}], config=cfg_unauth, opener=mock_opener)
            self.assertEqual(res, "ok")
            self.assertEqual(len(captured_requests), 1)
            self.assertNotIn("Authorization", captured_requests[0].headers)


class ControlledLoopbackServerTests(unittest.TestCase):
    """End-to-end HTTP tests against a real controlled ThreadingHTTPServer on port 0."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.persona_file = self.root / "persona.txt"
        self.persona_file.write_text("Persona.", encoding="utf-8")
        self.l0_dir = self.root / "l0"
        self.l0_dir.mkdir(parents=True, exist_ok=True)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ControlledAuthHandler)
        self.server.request_count = 0
        self.server.leak_hit_count = 0
        self.server.recorded_requests = []
        self.server.forced_redirect_code = None
        self.server.require_auth = True
        self.server.custom_response_content = None

        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.loopback_url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        self.cfg_loopback_auth = BotConfig(
            root=self.root,
            db=self.root / "loopback.sqlite3",
            owner_id="owner-lb",
            character_id="char-lb",
            persona_file=self.persona_file,
            l0_dir=self.l0_dir,
            model_url=self.loopback_url,
            model_id="lb-model",
            telegram_token_env="LB_TOKEN",
            telegram_user_id_env="LB_USER",
            model_api_key_env="LB_API_KEY",
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=3)
        self.tmp.cleanup()

    def test_valid_synthetic_key_reaches_server_with_authorization_bearer_200(self):
        env = {"LB_API_KEY": SYNTHETIC_TEST_KEY}
        reply = bot.model_reply(
            [{"role": "user", "content": "ping"}],
            config=self.cfg_loopback_auth,
            env=env,
        )
        self.assertEqual(reply, "synthetic loopback response")
        self.assertEqual(self.server.request_count, 1)

        req = self.server.recorded_requests[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["headers"].get("Authorization"), f"Bearer {SYNTHETIC_TEST_KEY}")
        self.assertEqual(req["headers"].get("Content-Type"), "application/json")

    def test_wrong_synthetic_key_returns_401_maps_to_bootstrap_error(self):
        env = {"LB_API_KEY": SYNTHETIC_WRONG_KEY}
        with self.assertRaises(bot.BootstrapError) as caught:
            bot.model_reply(
                [{"role": "user", "content": "ping"}],
                config=self.cfg_loopback_auth,
                env=env,
            )
        self.assertEqual(str(caught.exception), "model unavailable")
        self.assertEqual(self.server.request_count, 1)
        self.assertEqual(self.server.recorded_requests[0]["headers"].get("Authorization"), f"Bearer {SYNTHETIC_WRONG_KEY}")

    def test_missing_key_rejected_before_network(self):
        env = {}
        with self.assertRaises(bot.BootstrapError) as caught:
            bot.model_reply(
                [{"role": "user", "content": "ping"}],
                config=self.cfg_loopback_auth,
                env=env,
            )
        self.assertEqual(str(caught.exception), "model unavailable")
        # Ensure zero network calls were dispatched to the server
        self.assertEqual(self.server.request_count, 0)

    def test_redirect_rejection_all_codes_no_second_endpoint_hits_no_auth_leakage(self):
        redirect_codes = [301, 302, 303, 307, 308]
        env = {"LB_API_KEY": SYNTHETIC_TEST_KEY}

        for code in redirect_codes:
            with self.subTest(redirect_code=code):
                self.server.forced_redirect_code = code
                self.server.request_count = 0
                self.server.leak_hit_count = 0
                self.server.recorded_requests.clear()

                with self.assertRaises(bot.BootstrapError) as caught:
                    bot.model_reply(
                        [{"role": "user", "content": "secret prompt"}],
                        config=self.cfg_loopback_auth,
                        env=env,
                    )
                self.assertEqual(str(caught.exception), "model unavailable")
                self.assertIsNone(caught.exception.__cause__)

                # The first endpoint was hit and responded with redirect
                self.assertEqual(self.server.request_count, 1)
                # Crucial: the leak destination was NEVER hit
                self.assertEqual(self.server.leak_hit_count, 0)


class ModelAuthPipelineWiringTests(unittest.TestCase):
    """End-to-end run() wiring with real SQLite memory pipeline and synthetic auth credentials."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.persona_file = self.root / "persona.txt"
        self.persona_file.write_text("companion synthetic persona.", encoding="utf-8")
        self.l0_dir = self.root / "l0"
        self.l0_dir.mkdir(parents=True, exist_ok=True)
        (self.l0_dir / "fact.md").write_text("> load: always\n# Facts\nSynthetic fact.\n", encoding="utf-8")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ControlledAuthHandler)
        self.server.request_count = 0
        self.server.recorded_requests = []
        self.server.forced_redirect_code = None
        self.server.require_auth = True
        self.server.custom_response_content = "ready"

        self.port = self.server.server_address[1]
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.cfg_auth = BotConfig(
            root=self.root,
            db=self.root / "pipeline_auth.sqlite3",
            owner_id="owner-pipe",
            character_id="char-pipe",
            persona_file=self.persona_file,
            l0_dir=self.l0_dir,
            model_url=f"http://127.0.0.1:{self.port}/v1/chat/completions",
            model_id="pipe-model-v1",
            telegram_token_env="PIPE_TOKEN",
            telegram_user_id_env="PIPE_USER",
            addressee_family_key="家人/配偶",
            model_api_key_env="PIPE_MODEL_KEY",
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=3)
        self.tmp.cleanup()

    def test_run_wires_auth_consistently_across_readiness_chooser_and_chat(self):
        env = {
            "PIPE_TOKEN": "synthetic-pipe-token",
            "PIPE_USER": "777",
            "PIPE_MODEL_KEY": SYNTHETIC_TEST_KEY,
        }

        captured = {}

        def fake_builder(token, chat, admin):
            app = Mock()
            captured["token"] = token
            captured["chat"] = chat
            captured["admin"] = admin

            def fake_polling(drop_pending_updates=True):
                # Trigger user statement and assistant reply recording
                chat.recorder("Synthetic user statement about fictional tea.")
                chat.reply_recorder("Synthetic assistant reply.")
            app.run_polling.side_effect = fake_polling
            return app

        now = [100.0]
        real_factory = bot.build_local_memory_runtime

        def fast_factory(*args, **kwargs):
            kwargs.update(idle_seconds=1, clock=lambda: now[0])
            rt = real_factory(*args, **kwargs)
            rt.worker.poll_seconds = 0.01
            return rt

        with patch.object(bot, "LocalEmbedder", MockEmbedder):
            with patch.object(bot, "build_local_memory_runtime", fast_factory):
                with patch("telegram_test_adapter.build_test_application", side_effect=fake_builder):
                    bot.run(env, config=self.cfg_auth)

        # 1. Verify token and chat scope
        self.assertEqual(captured["token"], "synthetic-pipe-token")
        self.assertEqual(captured["chat"].scopes[777].owner_id, "owner-pipe")

        # 2. Verify all HTTP requests to server carried the exact Authorization Bearer token
        self.assertGreaterEqual(self.server.request_count, 1)
        for req in self.server.recorded_requests:
            self.assertEqual(req["headers"].get("Authorization"), f"Bearer {SYNTHETIC_TEST_KEY}")
            self.assertEqual(req["headers"].get("Content-Type"), "application/json")

        # 3. Verify real SQLite database was populated and isolated
        with contextlib.closing(sqlite3.connect(self.cfg_auth.db)) as conn:
            rows = conn.execute("SELECT owner_id, character_id, role, content FROM messages").fetchall()
            self.assertGreaterEqual(len(rows), 2)
            for r in rows:
                self.assertEqual(r[0], "owner-pipe")
                self.assertEqual(r[1], "char-pipe")
            roles = {r[2] for r in rows}
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)


if __name__ == "__main__":
    unittest.main()
