"""Offline unit tests for bot_config.py.

Verifies strict validation, schema compliance, path confinement,
no-side-effect config loading, content-free errors, and default compatibility.
"""
from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path
import tempfile
import unittest

from bot_config import BotConfig, ConfigError, load_bot_config, validate_model_url
import bootstrap_test_bot as bot


class BotConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.subroot = self.root / "subroot"
        self.subroot.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _valid_dict(self):
        return {
            "root": "subroot",
            "db": "data/test.sqlite3",
            "owner_id": "test-owner",
            "character_id": "test-char",
            "persona_file": "personas/persona.txt",
            "l0_dir": "personas",
            "model_url": "http://127.0.0.1:18080/v1/chat/completions",
            "model_id": "test-model-v1",
            "telegram_token_env": "TEST_BOT_TOKEN",
            "telegram_user_id_env": "TEST_USER_ID",
            "addressee_family_key": "家人/配偶",
        }

    def _write_json(self, data, filename="bot.json"):
        cfg_file = self.root / filename
        if isinstance(data, str):
            cfg_file.write_text(data, encoding="utf-8")
        else:
            cfg_file.write_text(json.dumps(data), encoding="utf-8")
        return cfg_file

    def test_default_config_compatibility(self):
        cfg = bot.default_config()
        self.assertIsInstance(cfg, BotConfig)
        self.assertEqual(cfg.owner_id, "bot_sandbox-synthetic-owner")
        self.assertEqual(cfg.character_id, "bot_sandbox-synthetic-companion")
        self.assertEqual(cfg.telegram_token_env, "BOT_TELEGRAM_TOKEN")
        self.assertEqual(cfg.telegram_user_id_env, "BOT_TELEGRAM_USER_ID")
        self.assertEqual(cfg.addressee_family_key, "家人/配偶")
        self.assertEqual(cfg.model_id, "qwen3.6-35b")
        self.assertEqual(cfg.model_url, "http://127.0.0.1:18080/v1/chat/completions")

        # Generic BotConfig dataclass default must be None
        f_map = {f.name: f.default for f in fields(BotConfig)}
        self.assertIsNone(f_map["addressee_family_key"])

        # Frozen instance
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            cfg.owner_id = "mutated"

    def test_load_bot_config_valid_and_anchoring(self):
        data = self._valid_dict()
        cfg_file = self._write_json(data)
        cfg = load_bot_config(cfg_file)

        self.assertEqual(cfg.root, self.subroot)
        self.assertEqual(cfg.db, self.subroot / "data" / "test.sqlite3")
        self.assertEqual(cfg.persona_file, self.subroot / "personas" / "persona.txt")
        self.assertEqual(cfg.l0_dir, self.subroot / "personas")
        self.assertEqual(cfg.owner_id, "test-owner")
        self.assertEqual(cfg.character_id, "test-char")
        self.assertEqual(cfg.model_url, "http://127.0.0.1:18080/v1/chat/completions")
        self.assertEqual(cfg.model_id, "test-model-v1")
        self.assertEqual(cfg.telegram_token_env, "TEST_BOT_TOKEN")
        self.assertEqual(cfg.telegram_user_id_env, "TEST_USER_ID")
        self.assertEqual(cfg.addressee_family_key, "家人/配偶")

    def test_load_bot_config_no_side_effects(self):
        data = self._valid_dict()
        cfg_file = self._write_json(data)
        cfg = load_bot_config(cfg_file)

        # DB file or directories should NOT have been created
        self.assertFalse(cfg.db.exists())
        self.assertFalse(cfg.persona_file.exists())
        self.assertFalse(cfg.l0_dir.exists())

    def test_json_malformed(self):
        cfg_file = self._write_json("{invalid json")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

        cfg_file2 = self._write_json('["root", "."]')
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file2)

        cfg_file3 = self._write_json("42")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file3)

    def test_json_duplicate_keys(self):
        dup_json = '{"root": "subroot", "root": "subroot", "db": "db.sqlite3"}'
        cfg_file = self._write_json(dup_json)
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

    def test_missing_required_fields(self):
        base = self._valid_dict()
        required_fields = [
            "root", "db", "owner_id", "character_id", "persona_file", "l0_dir",
            "model_url", "model_id", "telegram_token_env", "telegram_user_id_env"
        ]
        for field in required_fields:
            data = dict(base)
            del data[field]
            cfg_file = self._write_json(data, f"missing_{field}.json")
            with self.subTest(missing=field), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

        # addressee_family_key is optional (defaults None)
        data = dict(base)
        del data["addressee_family_key"]
        cfg_file = self._write_json(data, "optional_family_key.json")
        cfg = load_bot_config(cfg_file)
        self.assertIsNone(cfg.addressee_family_key)

        # explicit null addressee_family_key
        data = dict(base)
        data["addressee_family_key"] = None
        cfg_file = self._write_json(data, "null_family_key.json")
        cfg = load_bot_config(cfg_file)
        self.assertIsNone(cfg.addressee_family_key)

    def test_unknown_fields(self):
        data = self._valid_dict()
        data["unexpected_admin_key"] = "forbidden"
        cfg_file = self._write_json(data)
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

    def test_invalid_types(self):
        for field in ("owner_id", "model_url", "root", "db"):
            data = self._valid_dict()
            data[field] = 12345
            cfg_file = self._write_json(data, f"type_{field}.json")
            with self.subTest(field=field), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

        data = self._valid_dict()
        data["addressee_family_key"] = 999
        cfg_file = self._write_json(data, "type_family_key.json")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

    def test_content_free_errors(self):
        canary = "super-secret-canary-never-leak"
        data = self._valid_dict()
        data["model_url"] = f"http://127.0.0.1:18080/v1/chat/completions?secret={canary}"
        cfg_file = self._write_json(data)
        with self.assertRaises(ConfigError) as caught:
            load_bot_config(cfg_file)
        self.assertNotIn(canary, str(caught.exception))

    def test_direct_construction_validation_no_bypass(self):
        # Escaping path
        with self.assertRaises(ConfigError):
            BotConfig(
                root=self.subroot,
                db=self.root / "escaped.sqlite3",
                owner_id="owner",
                character_id="char",
                persona_file=self.subroot / "p.txt",
                l0_dir=self.subroot,
                model_url="http://127.0.0.1:18080/v1/chat/completions",
                model_id="mid",
                telegram_token_env="TOKEN",
                telegram_user_id_env="USER",
            )

        # Invalid model URL
        with self.assertRaises(ConfigError):
            BotConfig(
                root=self.subroot,
                db=self.subroot / "ok.sqlite3",
                owner_id="owner",
                character_id="char",
                persona_file=self.subroot / "p.txt",
                l0_dir=self.subroot,
                model_url="https://evil.com/v1/chat/completions",
                model_id="mid",
                telegram_token_env="TOKEN",
                telegram_user_id_env="USER",
            )

    def test_path_validation_traversal_and_escapes(self):
        for bad_path in (
            "../outside.sqlite3",
            "../../etc/passwd",
            "\\\\unc\\share\\db",
            "//unc/share/db",
            "http://127.0.0.1/db",
            "file:///db",
            "C:rel/db",
            "db\x00evil.sqlite3",
            "",
            "   ",
        ):
            data = self._valid_dict()
            data["db"] = bad_path
            cfg_file = self._write_json(data, "bad_path.json")
            with self.subTest(bad=bad_path), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

        # Absolute path outside root
        data = self._valid_dict()
        data["db"] = str(self.root / "outside.sqlite3")
        cfg_file = self._write_json(data, "abs_outside.json")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

        # Absolute path inside root is allowed
        data = self._valid_dict()
        data["db"] = str(self.subroot / "inside.sqlite3")
        cfg_file = self._write_json(data, "abs_inside.json")
        cfg = load_bot_config(cfg_file)
        self.assertEqual(cfg.db, (self.subroot / "inside.sqlite3").resolve())

    def test_model_url_strict_loopback(self):
        valid_urls = (
            "http://127.0.0.1:18080/v1/chat/completions",
            "http://127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:19092/v1/chat/completions",
        )
        for url in valid_urls:
            self.assertEqual(validate_model_url(url), url)

        invalid_urls = (
            "https://127.0.0.1:18080/v1/chat/completions",
            "http://localhost:18080/v1/chat/completions",
            "http://0.0.0.0:18080/v1/chat/completions",
            "http://[::1]:18080/v1/chat/completions",
            "http://192.168.1.1:18080/v1/chat/completions",
            "http://127.0.0.1/v1/chat/completions",
            "http://127.0.0.1:0/v1/chat/completions",
            "http://127.0.0.1:70000/v1/chat/completions",
            "http://127.0.0.1:18080/v1/chat",
            "http://127.0.0.1:18080/",
            "http://user:pass@127.0.0.1:18080/v1/chat/completions",
            "http://127.0.0.1:18080/v1/chat/completions?query=1",
            "http://127.0.0.1:18080/v1/chat/completions#fragment",
            "",
            "   ",
        )
        for bad in invalid_urls:
            with self.subTest(url=bad), self.assertRaises(ConfigError):
                validate_model_url(bad)

    def test_env_var_validation(self):
        # Invalid syntax
        for bad in ("1BAD", "BAD-VAR", "BAD VAR", "", "   "):
            data = self._valid_dict()
            data["telegram_token_env"] = bad
            cfg_file = self._write_json(data, "bad_env.json")
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

        # Distinctness check
        data = self._valid_dict()
        data["telegram_user_id_env"] = data["telegram_token_env"]
        cfg_file = self._write_json(data, "same_env.json")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

    def test_identifier_validation(self):
        for bad_id in ("", "   ", " whitespace_leading", "trailing ", "x" * 129, "has\ncontrol"):
            data = self._valid_dict()
            data["owner_id"] = bad_id
            cfg_file = self._write_json(data, "bad_owner.json")
            with self.subTest(bad=bad_id), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

    def test_model_id_validation(self):
        for bad_mid in ("", "   ", " mid", "mid ", "x" * 129, "mid\x00", "mid\x7f"):
            data = self._valid_dict()
            data["model_id"] = bad_mid
            cfg_file = self._write_json(data, "bad_mid.json")
            with self.subTest(bad=bad_mid), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

    def test_strict_validation_boundary_repairs(self):
        # 1. URL malformed ports and brackets mapping to ConfigError
        for bad_url in (
            "http://127.0.0.1:65536/v1/chat/completions",
            "http://127.0.0.1:70000/v1/chat/completions",
            "http://[127.0.0.1:8080/v1/chat/completions",
            "http://127.0.0.1:notaport/v1/chat/completions",
            "http://127.0.0.1:-1/v1/chat/completions",
        ):
            with self.subTest(bad_url=bad_url), self.assertRaises(ConfigError) as caught:
                validate_model_url(bad_url)
            self.assertIsNone(caught.exception.__cause__)
            self.assertNotIn("65536", str(caught.exception))
            self.assertNotIn("70000", str(caught.exception))

        # 2. addressee_family_key supports ONLY None or '家人/配偶'
        for bad_key in ("家人/子女", "家人", "配偶", "other", "", "   "):
            data = self._valid_dict()
            data["addressee_family_key"] = bad_key
            cfg_file = self._write_json(data, "bad_family_key.json")
            with self.subTest(bad_key=bad_key), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)
            with self.assertRaises(ConfigError):
                BotConfig(
                    root=self.subroot,
                    db=self.subroot / "ok.sqlite3",
                    owner_id="owner",
                    character_id="char",
                    persona_file=self.subroot / "p.txt",
                    l0_dir=self.subroot,
                    model_url="http://127.0.0.1:18080/v1/chat/completions",
                    model_id="mid",
                    telegram_token_env="TOKEN",
                    telegram_user_id_env="USER",
                    addressee_family_key=bad_key,
                )

        # 3. env names: fullmatch, case-insensitive collision rejection, trailing newline rejection
        data = self._valid_dict()
        data["telegram_token_env"] = "TEST_TOKEN\n"
        cfg_file = self._write_json(data, "newline_env.json")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

        data = self._valid_dict()
        data["telegram_token_env"] = "token_name"
        data["telegram_user_id_env"] = "TOKEN_NAME"
        cfg_file = self._write_json(data, "case_env.json")
        with self.assertRaises(ConfigError):
            load_bot_config(cfg_file)

        # 4. raw path validation: reject whitespace instead of strip, controls, traversal, ADS, reserved devices
        for bad_p in (
            " subroot",
            "subroot ",
            "\tsubroot",
            "subroot\n",
            "C:",
            "c:",
            "C:rel/path",
            "subroot/../subroot",
            "data/../secret.txt",
            "data/file.txt:stream",
            "C:/foo:bar",
            "subroot/CON",
            "subroot/aux.txt",
            "subroot/nul.dat",
            "subroot/com1",
            "subroot/lpt9",
            "subroot/trailing_dot.",
            "subroot/trailing_space ",
            "subroot/del\x7fchar",
        ):
            data = self._valid_dict()
            data["db"] = bad_p
            cfg_file = self._write_json(data, "bad_raw_path.json")
            with self.subTest(bad_p=bad_p), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

        # 5. model_id and identifier rejecting DEL (\x7f)
        for bad_id in ("user\x7fid", "\x7f", "mid\x7f"):
            data = self._valid_dict()
            data["owner_id"] = bad_id
            cfg_file = self._write_json(data, "bad_id_del.json")
            with self.subTest(bad_id=bad_id), self.assertRaises(ConfigError):
                load_bot_config(cfg_file)

        # 6. Canary error content: invalid UTF-8 JSON
        bad_utf8_file = self.root / "bad_utf8.json"
        bad_utf8_file.write_bytes(b'{"root": "subroot", "bad": \xff\xfe}')
        with self.assertRaises(ConfigError) as caught:
            load_bot_config(bad_utf8_file)
        self.assertIsNone(caught.exception.__cause__)
        self.assertNotIn("0xff", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
