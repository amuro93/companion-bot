"""Minimal per-bot configuration decoupled from bootstrap runtime.

Configuration is trusted local input, not a remote administration interface.
Config loading and instantiation perform pure structural and filesystem validation
without opening SQLite databases, reading persona text or L0 files, contacting
network endpoints, or spawning threads.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

MAX_ID_LENGTH = 128
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")
_WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
})


class ConfigError(Exception):
    """Sanitized, content-free configuration validation error."""


def _validate_raw_path(raw: Any) -> str:
    """Validate that raw path representation is a safe, non-escaping string."""
    if not isinstance(raw, (str, Path)):
        raise ConfigError("invalid path")
    s = str(raw)
    if not s:
        raise ConfigError("invalid path")
    if s != s.strip():
        raise ConfigError("path must not contain leading or trailing whitespace")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in s):
        raise ConfigError("invalid path")
    if any(ch in '<>"|?*' for ch in s):
        raise ConfigError("invalid path")
    if s.startswith(("\\\\", "//")):
        raise ConfigError("UNC paths are forbidden")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        raise ConfigError("URLs are forbidden as filesystem paths")

    # Reject drive-relative paths (e.g. 'C:foo', 'C:', 'c:')
    if re.match(r"^[a-zA-Z]:(?:$|[^\\/])", s):
        raise ConfigError("drive-relative paths are forbidden")

    # Check for colons (ADS colon forbidden except single drive colon in absolute drive path like C:\)
    colon_idx = s.find(":")
    if colon_idx != -1:
        if colon_idx == 1 and s[0].isalpha() and len(s) >= 3 and s[2] in ("\\", "/"):
            # Only drive colon allowed; make sure no other colon exists
            if ":" in s[2:]:
                raise ConfigError("invalid path colon or alternate data stream")
        else:
            raise ConfigError("invalid path colon or alternate data stream")

    # Check components
    normalized_slash = s.replace("\\", "/")
    parts = normalized_slash.split("/")
    for part in parts:
        if not part:
            continue
        if part == "..":
            raise ConfigError("parent traversal components are forbidden")
        if part != "." and (part.endswith(".") or part.endswith(" ")):
            raise ConfigError("path components must not have trailing dot or space")
        base = part.split(".", 1)[0].upper()
        if base in _WINDOWS_RESERVED_NAMES:
            raise ConfigError("Windows reserved device names are forbidden")

    return s


def _resolve_subpath(raw: Any, root: Path, is_file: bool = False) -> Path:
    """Resolve subpath relative to root; forbid escaping and traversal."""
    raw_str = _validate_raw_path(raw)
    p = Path(raw_str)
    try:
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (root / p).resolve()

        if not resolved.is_relative_to(root):
            raise ConfigError("path escapes root directory")
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("path resolution error") from None

    if is_file and resolved == root:
        raise ConfigError("file path cannot be root directory")

    return resolved


APPROVED_REMOTE_MODEL_URL = "http://llm-host:8080/v1/chat/completions"
APPROVED_GPU_MODEL_URL = "http://gpu-host:8080/v1/chat/completions"
APPROVED_DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def approved_remote_model_urls() -> tuple[str, ...]:
    """Return the allow-list of approved non-loopback model endpoints.

    Kept as a function so tests (and launchers) can override the constants
    without having to rebuild an import-time tuple.
    """
    return (APPROVED_REMOTE_MODEL_URL, APPROVED_GPU_MODEL_URL, APPROVED_DEEPSEEK_URL)


def is_remote_model_url(url_str: str) -> bool:
    """Return True if url_str matches an approved remote endpoint."""
    return url_str in approved_remote_model_urls()


def validate_model_url(url_str: Any) -> str:
    """Validate OpenAI-compatible endpoint URL for loopback or approved remote compliance."""
    if not isinstance(url_str, str) or not url_str:
        raise ConfigError("invalid model URL")
    if url_str != url_str.strip():
        raise ConfigError("invalid model URL")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in url_str):
        raise ConfigError("invalid model URL")

    if url_str in approved_remote_model_urls():
        return url_str

    try:
        parsed = urlsplit(url_str)
    except ValueError:
        raise ConfigError("invalid model URL") from None

    if parsed.scheme != "http":
        raise ConfigError("invalid model URL")

    try:
        port = parsed.port
    except ValueError:
        raise ConfigError("invalid model URL") from None

    if port is None or not (1 <= port <= 65535):
        raise ConfigError("invalid model URL")

    if parsed.hostname != "127.0.0.1":
        raise ConfigError("invalid model URL")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ConfigError("invalid model URL")
    if parsed.netloc != f"127.0.0.1:{port}":
        raise ConfigError("invalid model URL")
    if parsed.path != "/v1/chat/completions":
        raise ConfigError("invalid model URL")
    if parsed.query or parsed.fragment:
        raise ConfigError("invalid model URL")
    if url_str != f"http://127.0.0.1:{port}/v1/chat/completions":
        raise ConfigError("invalid model URL")
    return url_str


def _validate_id(value: Any) -> str:
    """Validate that an identifier satisfies memory_core._validate_id rules."""
    if not isinstance(value, str):
        raise ConfigError("invalid identifier")
    if not value or not value.strip() or value != value.strip():
        raise ConfigError("invalid identifier")
    if len(value) > MAX_ID_LENGTH:
        raise ConfigError("invalid identifier")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ConfigError("invalid identifier")
    return value


def _validate_model_id(value: Any) -> str:
    """Validate model_id as non-empty bounded string without control characters."""
    if not isinstance(value, str):
        raise ConfigError("invalid model ID")
    if not value or not value.strip() or value != value.strip():
        raise ConfigError("invalid model ID")
    if len(value) > 128:
        raise ConfigError("invalid model ID")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ConfigError("invalid model ID")
    return value


def _validate_env_var_name(value: Any) -> str:
    """Strict environment variable name grammar."""
    if not isinstance(value, str):
        raise ConfigError("invalid environment variable name")
    if not value or len(value) > 128:
        raise ConfigError("invalid environment variable name")
    if not _ENV_VAR_RE.fullmatch(value):
        raise ConfigError("invalid environment variable name")
    return value


def _validate_family_key(value: Any) -> str | None:
    """Validate optional relationship family key."""
    if value is None:
        return None
    if value != "家人/配偶":
        raise ConfigError("unsupported addressee family key")
    return value


@dataclass(frozen=True)
class ModelChoice:
    """One selectable chat backend exposed by the /model switch."""

    label: str
    model_url: str
    model_id: str
    model_api_key_env: str | None = None


@dataclass(frozen=True)
class BotConfig:
    """Minimal frozen configuration specification for a single bot runtime."""

    root: Path
    db: Path
    owner_id: str
    character_id: str
    persona_file: Path
    l0_dir: Path
    model_url: str
    model_id: str
    telegram_token_env: str
    telegram_user_id_env: str
    addressee_family_key: str | None = None
    model_api_key_env: str | None = None
    model_choices: tuple[ModelChoice, ...] = ()

    def __post_init__(self):
        root_str = _validate_raw_path(self.root)
        try:
            root_path = Path(root_str).resolve()
        except Exception:
            raise ConfigError("root directory invalid") from None

        try:
            if not root_path.is_dir() or root_path.is_symlink():
                raise ConfigError("root directory invalid")
        except ConfigError:
            raise
        except Exception:
            raise ConfigError("root directory invalid") from None

        object.__setattr__(self, "root", root_path)

        db_path = _resolve_subpath(self.db, root_path, is_file=True)
        persona_path = _resolve_subpath(self.persona_file, root_path, is_file=True)
        l0_path = _resolve_subpath(self.l0_dir, root_path, is_file=False)

        object.__setattr__(self, "db", db_path)
        object.__setattr__(self, "persona_file", persona_path)
        object.__setattr__(self, "l0_dir", l0_path)

        owner = _validate_id(self.owner_id)
        character = _validate_id(self.character_id)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "character_id", character)

        url = validate_model_url(self.model_url)
        object.__setattr__(self, "model_url", url)

        mid = _validate_model_id(self.model_id)
        object.__setattr__(self, "model_id", mid)

        token_env = _validate_env_var_name(self.telegram_token_env)
        user_env = _validate_env_var_name(self.telegram_user_id_env)
        if token_env.upper() == user_env.upper():
            raise ConfigError("token and user environment variable names must be distinct")
        object.__setattr__(self, "telegram_token_env", token_env)
        object.__setattr__(self, "telegram_user_id_env", user_env)

        family_key = _validate_family_key(self.addressee_family_key)
        object.__setattr__(self, "addressee_family_key", family_key)

        if self.model_api_key_env is not None:
            key_env = _validate_env_var_name(self.model_api_key_env)
            if key_env.upper() in (token_env.upper(), user_env.upper()):
                raise ConfigError("model API key environment variable name must be distinct from Telegram credentials")
            object.__setattr__(self, "model_api_key_env", key_env)
        else:
            if is_remote_model_url(url):
                raise ConfigError("remote model endpoint requires model_api_key_env")
            object.__setattr__(self, "model_api_key_env", None)

        choices = self.model_choices
        if not isinstance(choices, tuple):
            raise ConfigError("model_choices must be a list")
        seen_labels = set()
        for choice in choices:
            if not isinstance(choice, ModelChoice):
                raise ConfigError("model_choices must contain objects")
            label = choice.label
            if (not isinstance(label, str) or not label.strip() or label != label.strip()
                    or len(label) > 40 or any(ord(c) < 32 or ord(c) == 127 for c in label)):
                raise ConfigError("invalid model choice label")
            if label in seen_labels:
                raise ConfigError("duplicate model choice label")
            seen_labels.add(label)
            validate_model_url(choice.model_url)
            _validate_model_id(choice.model_id)
            if choice.model_api_key_env is None:
                if is_remote_model_url(choice.model_url):
                    raise ConfigError("remote model choice requires model_api_key_env")
            else:
                _validate_env_var_name(choice.model_api_key_env)


def load_bot_config(path: str | Path) -> BotConfig:
    """Load, strictly validate and anchor a BotConfig from a JSON file.

    Config loading performs no network, DB, or persona read operations.
    """
    path_str = _validate_raw_path(path)
    try:
        file_path = Path(path_str).resolve()
        if not file_path.is_file():
            raise ConfigError("configuration file not found")
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("cannot access configuration file") from None

    try:
        raw_bytes = file_path.read_bytes()
    except OSError:
        raise ConfigError("cannot read configuration file") from None

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ConfigError("invalid UTF-8 configuration file") from None

    def _check_duplicates(pairs):
        keys = set()
        res = {}
        for k, v in pairs:
            if k in keys:
                raise ConfigError("duplicate key in configuration")
            keys.add(k)
            res[k] = v
        return res

    try:
        data = json.loads(content, object_pairs_hook=_check_duplicates)
    except json.JSONDecodeError:
        raise ConfigError("invalid JSON syntax") from None

    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a JSON object")

    allowed_keys = {
        "root", "db", "owner_id", "character_id", "persona_file", "l0_dir",
        "model_url", "model_id", "telegram_token_env", "telegram_user_id_env",
        "addressee_family_key", "model_api_key_env", "model_choices",
    }
    required_keys = allowed_keys - {"addressee_family_key", "model_api_key_env", "model_choices"}

    extra_keys = set(data.keys()) - allowed_keys
    if extra_keys:
        raise ConfigError("unknown configuration field")

    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ConfigError("missing required configuration field")

    for k in required_keys:
        if not isinstance(data[k], str):
            raise ConfigError("configuration field must be a string")

    if "addressee_family_key" in data and data["addressee_family_key"] is not None:
        if not isinstance(data["addressee_family_key"], str):
            raise ConfigError("addressee_family_key must be a string or null")

    if "model_api_key_env" in data and data["model_api_key_env"] is not None:
        if not isinstance(data["model_api_key_env"], str):
            raise ConfigError("model_api_key_env must be a string or null")

    choices_raw = data.get("model_choices", [])
    if not isinstance(choices_raw, list):
        raise ConfigError("model_choices must be a list")
    choices = []
    for item in choices_raw:
        if not isinstance(item, dict):
            raise ConfigError("model_choices must contain objects")
        if set(item) - {"label", "model_url", "model_id", "model_api_key_env"}:
            raise ConfigError("unknown model choice field")
        if not {"label", "model_url", "model_id"} <= set(item):
            raise ConfigError("missing model choice field")
        if not all(isinstance(item[k], str) for k in ("label", "model_url", "model_id")):
            raise ConfigError("model choice fields must be strings")
        key_env = item.get("model_api_key_env")
        if key_env is not None and not isinstance(key_env, str):
            raise ConfigError("model_api_key_env must be a string or null")
        choices.append(ModelChoice(label=item["label"], model_url=item["model_url"],
                                   model_id=item["model_id"], model_api_key_env=key_env))

    try:
        json_dir = file_path.parent.resolve()
        root_raw = _validate_raw_path(data["root"])
        root_p = Path(root_raw)
        if root_p.is_absolute():
            resolved_root = root_p.resolve()
        else:
            resolved_root = (json_dir / root_p).resolve()
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("root path resolution error") from None

    try:
        return BotConfig(
            root=resolved_root,
            db=data["db"],
            owner_id=data["owner_id"],
            character_id=data["character_id"],
            persona_file=data["persona_file"],
            l0_dir=data["l0_dir"],
            model_url=data["model_url"],
            model_id=data["model_id"],
            telegram_token_env=data["telegram_token_env"],
            telegram_user_id_env=data["telegram_user_id_env"],
            addressee_family_key=data.get("addressee_family_key"),
            model_api_key_env=data.get("model_api_key_env"),
            model_choices=tuple(choices),
        )
    except ConfigError:
        raise
    except Exception:
        raise ConfigError("invalid configuration") from None
