"""Isolated P2c test bot. Never import or launch the production bot.

Set BOT_TELEGRAM_TOKEN and BOT_TELEGRAM_USER_ID in the process environment.
--check validates configuration without importing optional dependencies or doing I/O.
Do not run concurrently with another poller using the same token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from bot_config import BotConfig, ConfigError, load_bot_config, validate_model_url
from companion_chat import PERSONA_MAX
from persona_l0 import PersonaL0

ROOT = Path(__file__).resolve().parent
DB = ROOT / "synthetic.sqlite3"
QDRANT_URL = "http://127.0.0.1:16333"
MODEL_URL = "http://127.0.0.1:18080/v1/chat/completions"
# Optional pre-filter decision model (cheap, no output tokens)
DECISION_URL = "http://127.0.0.1:18081/v1/systemone"
DECISION_KEY_FILE = ROOT / "startlux_8092.key"
MODEL_ID = "qwen3.6-35b"
EMBEDDING_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OWNER = "bot_sandbox-synthetic-owner"
CHARACTER = "bot_sandbox-synthetic-companion"
COLLECTION = "synthetic_sandbox_only"
# Condensed example persona (identity + speech + boundaries only; no tooling or runtime settings).
# The detailed life facts live in personas/*.md and are injected per turn by PersonaL0.
PERSONA_FILE = ROOT / "personas" / "example_companion.txt"
PERSONA_MAX_CHARS = PERSONA_MAX
L0_DIR = ROOT / "personas"


class BootstrapError(Exception):
    """Public, content-free startup failure."""


class _DisallowRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

    def http_error_301(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, "redirect forbidden", headers, fp)

    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


_NO_REDIRECT_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _DisallowRedirects,
)


def default_config() -> BotConfig:
    """Construct equivalent companion configuration lazily from module globals."""
    return BotConfig(
        root=ROOT,
        db=DB,
        owner_id=OWNER,
        character_id=CHARACTER,
        persona_file=PERSONA_FILE,
        l0_dir=L0_DIR,
        model_url=MODEL_URL,
        model_id=MODEL_ID,
        telegram_token_env="BOT_TELEGRAM_TOKEN",
        telegram_user_id_env="BOT_TELEGRAM_USER_ID",
        addressee_family_key="家人/配偶",
        model_api_key_env=None,
    )


def load_persona(path=None):
    """Load the sandbox character persona; fail closed when missing or malformed."""
    path = PERSONA_FILE if path is None else (Path(path) if isinstance(path, str) else path)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, AttributeError, TypeError):
        raise BootstrapError("persona unavailable") from None
    if not text or len(text) > PERSONA_MAX_CHARS or "\x00" in text:
        raise BootstrapError("persona invalid")
    return text


def preflight(env=None, *, config: BotConfig | None = None):
    """Pure config check: no DB, network, model, or Telegram contact."""
    env = os.environ if env is None else env
    if config is None:
        token = env.get("BOT_TELEGRAM_TOKEN", "")
        user = env.get("BOT_TELEGRAM_USER_ID", "")
        token_env = "BOT_TELEGRAM_TOKEN"
        user_env = "BOT_TELEGRAM_USER_ID"
        root_dir = ROOT
        db_path = DB
        model_url = MODEL_URL
        persona_path = None
    else:
        token = env.get(config.telegram_token_env, "")
        user = env.get(config.telegram_user_id_env, "")
        token_env = config.telegram_token_env
        user_env = config.telegram_user_id_env
        root_dir = config.root
        db_path = config.db
        model_url = config.model_url
        persona_path = config.persona_file

    if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
        raise BootstrapError(f"{token_env} must be set without whitespace")
    if not isinstance(user, str) or not user.isascii() or not user.isdecimal() or int(user) <= 0:
        raise BootstrapError(f"{user_env} must be a positive decimal integer")
    if not root_dir.is_dir() or root_dir.is_symlink() or db_path.is_symlink():
        raise BootstrapError("sandbox path is invalid")

    if config is None:
        if db_path.parent != root_dir:
            raise BootstrapError("sandbox path is invalid")
        if (urlsplit(model_url).hostname != "127.0.0.1"
                or urlsplit(model_url).port != 18080):
            raise BootstrapError("backend configuration is invalid")
    else:
        try:
            if not db_path.resolve().is_relative_to(root_dir.resolve()):
                raise BootstrapError("sandbox path is invalid")
        except (ValueError, OSError):
            raise BootstrapError("sandbox path is invalid")
        try:
            validate_model_url(model_url)
        except Exception:
            raise BootstrapError("backend configuration is invalid")

        if config.model_api_key_env is not None:
            try:
                raw_key = env.get(config.model_api_key_env)
                if (not isinstance(raw_key, str)
                        or not raw_key
                        or not raw_key.isascii()
                        or any(c.isspace() for c in raw_key)
                        or any(ord(c) < 32 or ord(c) == 127 for c in raw_key)
                        or len(raw_key) > 4096):
                    raise BootstrapError("backend configuration is invalid")
            except Exception:
                raise BootstrapError("backend configuration is invalid") from None

    if config is None:
        load_persona()
    else:
        load_persona(persona_path)
    return token, int(user)


class LocalEmbedder:
    model_id = EMBEDDING_ID
    dimension = 384

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_id, device="cpu")
        if self.model.get_sentence_embedding_dimension() != self.dimension:
            raise BootstrapError("embedding dimension mismatch")

    def embed(self, text):
        return self.model.encode(text).tolist()


def model_reply(messages, *, opener=None, config: BotConfig | None = None, env=None):
    """Non-streaming OpenAI-compatible llama-server call; sanitized failures."""
    url = config.model_url if config is not None else MODEL_URL
    model_id = config.model_id if config is not None else MODEL_ID
    opener = _NO_REDIRECT_OPENER.open if opener is None else opener

    # Validate endpoint again before sending any key
    try:
        validate_model_url(url)
    except Exception:
        raise BootstrapError("model unavailable") from None

    headers = {"Content-Type": "application/json"}

    if config is not None and config.model_api_key_env is not None:
        try:
            env_map = os.environ if env is None else env
            raw_key = env_map.get(config.model_api_key_env)
            if (not isinstance(raw_key, str)
                    or not raw_key
                    or not raw_key.isascii()
                    or any(c.isspace() for c in raw_key)
                    or any(ord(c) < 32 or ord(c) == 127 for c in raw_key)
                    or len(raw_key) > 4096):
                raise BootstrapError("model unavailable")
            headers["Authorization"] = f"Bearer {raw_key}"
        except Exception:
            raise BootstrapError("model unavailable") from None

    try:
        payload = {"model": model_id, "messages": messages,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "max_tokens": 512, "stream": False}
        budget_endpoint = (url == "http://llm-host:8081/v1/chat/completions"
                         and model_id == "qwen3.8-flash-next-iq3_xxs")
        gpu_endpoint = (url == "http://gpu-host:8080/v1/chat/completions"
                          and model_id == "qwen3.8-flash-next")
        deepseek_endpoint = (url == "https://api.deepseek.com/chat/completions"
                             and model_id == "deepseek-flash")
        if config is not None and deepseek_endpoint:
            # DeepSeek has no token budget knob; effort levels only. "low" measured
            # at 120-285 characters of reasoning, matching the 50-150 token target.
            mode = (os.environ if env is None else env).get("THINK_MODE", "128")
            del payload["chat_template_kwargs"]
            if mode == "off":
                payload["thinking"] = {"type": "disabled"}
            else:
                payload["thinking"] = {"type": "enabled"}
                payload["reasoning_effort"] = "low"
        elif config is not None and (budget_endpoint or gpu_endpoint):
            # Hybrid-reasoning models degrade (one-character or empty replies) when
            # thinking is force-disabled, so bounded thinking is the default here.
            # THINK_MODE=off is the explicit opt-out; otherwise an integer
            # 50-150 sets the per-turn reasoning budget.
            mode = (os.environ if env is None else env).get("THINK_MODE", "128")
            if mode != "off":
                try:
                    budget = int(mode)
                except (TypeError, ValueError):
                    raise ValueError("invalid isolated thinking mode") from None
                if not 50 <= budget <= 150:
                    raise ValueError("invalid isolated thinking mode")
                del payload["chat_template_kwargs"]
                payload["reasoning_budget_tokens"] = budget
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body,
                                         headers=headers, method="POST")
    except Exception:
        raise BootstrapError("model unavailable") from None

    # Empty or truncated completions are a known failure mode of the quantized
    # hybrid-reasoning model (it can spend the whole budget thinking and then
    # stop with an empty content field). Retry once before giving up so a
    # transient empty turn does not surface as "model unavailable".
    for attempt in range(2):
        try:
            with opener(request, timeout=40) as response:
                status = response.status
                raw = response.read(65537)
        except urllib.error.HTTPError as exc:
            try:
                if getattr(exc, "fp", None) is not None:
                    exc.fp.close()
            except Exception:
                pass
            # Authentication / redirect / server errors are not transient here.
            raise BootstrapError("model unavailable") from None
        except Exception:
            raw = None
            status = None

        if status is not None and status != 200:
            raise BootstrapError("model unavailable")
        if raw is not None and len(raw) > 65536:
            raise BootstrapError("model unavailable")
        if raw is not None:
            try:
                payload = json.loads(raw)
                choices = payload["choices"]
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("choices")
                choice = choices[0]
                content = choice["message"]["content"]
                if choice["finish_reason"] != "stop" or not isinstance(content, str) or not content.strip():
                    raise ValueError("incomplete response")
                return content.strip()
            except Exception:
                pass
        if attempt == 0:
            time.sleep(0.3)
    raise BootstrapError("model unavailable") from None


def seed(store):
    """Fixed, fabricated source and memory IDs; replay is idempotent.

    Kept for tests only. Retired from the live path (P2h) because real conversations now
    populate the store, and re-seeding after a deletion would try to resurrect a
    tombstoned external_id, which save_memory refuses by design.
    """
    for label, source, fact in (
        ("tea", "Synthetic test user says they like fictional jasmine tea.", "Synthetic test user prefers fictional jasmine tea."),
        ("book", "Synthetic test user says they collect imaginary blue books.", "Synthetic test user collects imaginary blue books."),
    ):
        source_id = store.append_message(OWNER, CHARACTER, "bot_sandbox-source-" + label, "user", source)
        store.save_memory(OWNER, CHARACTER, "bot_sandbox-memory-" + label, "fact", fact,
                          [source_id], domain="real", visible_to=[CHARACTER])


def build_local_memory_runtime(
    store,
    embedder,
    keyer,
    bearer,
    *,
    owner_id=OWNER,
    character_id=CHARACTER,
    idle_seconds=180,
    clock=time.time,
):
    try:
        from types import SimpleNamespace
        from memory_core import _validate_id
        from memory_http import Scope
        from memory_index import IndexWorker
        from memory_sqlite_vector import SQLiteVectorBackend, SQLiteSemanticSearcher
        from memory_local_client import InProcessMemoryClient
        from memory_deferred_queue import DeferredMemoryQueue
        from memory_deferred_processor import DeferredMemoryProcessor
        from memory_deferred_worker import DeferredMemoryWorker

        if (
            not isinstance(bearer, str)
            or len(bearer) < 32
            or len(bearer) > 1024
            or not bearer.isascii()
            or not bearer.isprintable()
            or any(c.isspace() for c in bearer)
        ):
            raise BootstrapError("local memory runtime unavailable")

        _validate_id(owner_id, "owner_id")
        _validate_id(character_id, "character_id")

        backend = SQLiteVectorBackend(store, embedder.dimension, embedder.model_id)
        indexer = IndexWorker(store, embedder, backend)
        searcher = SQLiteSemanticSearcher(store, embedder, backend)
        client = InProcessMemoryClient(searcher, {bearer: Scope(owner_id, character_id)})
        processor = DeferredMemoryProcessor(store, owner_id, character_id, keyer, indexer)
        queue = DeferredMemoryQueue(store, owner_id, character_id, idle_seconds=idle_seconds, clock=clock)
        worker = DeferredMemoryWorker(queue, processor)
        return SimpleNamespace(
            backend=backend,
            indexer=indexer,
            searcher=searcher,
            client=client,
            queue=queue,
            processor=processor,
            worker=worker,
        )
    except BootstrapError:
        raise BootstrapError("local memory runtime unavailable") from None
    except Exception:
        raise BootstrapError("local memory runtime unavailable") from None


def run(env=None, *, config: BotConfig | None = None):
    token, user_id = preflight(env, config=config)  # MUST happen before heavyweight imports or model load.
    from companion_chat import ChatScope, CompanionChat
    from memory_admin import MemoryAdmin
    from memory_core import MemoryStore
    from memory_auto import derive_key
    from memory_keyer import CandidateRetriever, KeyDecision, ModelKeyChooser
    from telegram_test_adapter import build_test_application

    store = None
    runtime = None
    router = None
    try:
        embedder = LocalEmbedder()

        if config is None:
            store_db = DB
            owner_id = OWNER
            character_id = CHARACTER
            family_key = "家人/配偶"
            l0_path = L0_DIR
            persona_text = load_persona()
            call_model = model_reply
        else:
            store_db = config.db
            owner_id = config.owner_id
            character_id = config.character_id
            family_key = config.addressee_family_key
            l0_path = config.l0_dir
            persona_text = load_persona(config.persona_file)
            if config.model_api_key_env is None:
                def bound_model(messages, *, opener=None):
                    return model_reply(messages, opener=opener, config=config)
            else:
                run_env = env
                def bound_model(messages, *, opener=None):
                    return model_reply(messages, opener=opener, config=config, env=run_env)
            if config.model_choices:
                from types import SimpleNamespace
                from model_router import ModelRouter
                backends = {}
                for choice in config.model_choices:
                    choice_cfg = SimpleNamespace(model_url=choice.model_url,
                                                 model_id=choice.model_id,
                                                 model_api_key_env=choice.model_api_key_env)

                    def _backend(messages, *, opener=None, _cfg=choice_cfg):
                        return model_reply(messages, opener=opener, config=_cfg, env=run_env)

                    backends[choice.label] = _backend
                call_model = ModelRouter(
                    backends,
                    state_path=Path(config.db).parent / "model-choice.json",
                )
                router = call_model
            else:
                call_model = bound_model

        store = MemoryStore(store_db)
        bearer = secrets.token_urlsafe(48)
        key_decision = KeyDecision(
            chooser=ModelKeyChooser(call_model),
            retriever=CandidateRetriever(store, embedder, owner_id, character_id),
            fallback_rule=derive_key,
            addressee_family_key=family_key,
        )
        runtime = build_local_memory_runtime(
            store, embedder, key_decision, bearer,
            owner_id=owner_id, character_id=character_id,
        )

        result = runtime.indexer.run_once(owner_id, character_id)
        if result["status"] != "ok" or result["pending_count"] != 0:
            raise BootstrapError("memory index unavailable")

        # Probe both services before polling; no prompt or response is logged.
        call_model([{"role": "user", "content": "Reply with one word: ready"}])
        runtime.client.search(bearer, "fictional jasmine tea", 1, "real")
        scope = ChatScope(bearer, owner_id, character_id, "real")

        def recorder(text):
            runtime.worker.capture(text)
            if store.is_course_reschedule(text):
                if store.needs_course_clarification(owner_id, character_id, text):
                    return "course_clarify"
                return "course_pending"

        def reply_recorder(text):
            """Archive her reply verbatim for review. Never a memory source, never deduped."""
            external = "asst-" + hashlib.sha256(
                f"{text}|{time.time_ns()}".encode("utf-8")).hexdigest()[:16]
            store.append_message(owner_id, character_id, external, "assistant", text)

        chat = CompanionChat({user_id: scope}, runtime.client, call_model, persona_text,
                             recorder=recorder, reply_recorder=reply_recorder,
                             persona_l0=PersonaL0(l0_path))
        admin = MemoryAdmin(store, runtime.indexer, owner_id, character_id)
        if router is not None:
            application = build_test_application(token, chat, admin, router=router)
        else:
            application = build_test_application(token, chat, admin)
        runtime.worker.start()
        # A retired bot may have queued old updates; never replay them into this sandbox.
        application.run_polling(drop_pending_updates=True)
    except Exception:
        raise BootstrapError("sandbox startup or polling failed") from None
    finally:
        if runtime is not None and getattr(runtime, "worker", None) is not None:
            stopped = False
            try:
                stopped = bool(runtime.worker.stop(timeout=5))
            except Exception:
                stopped = False
            if stopped:
                if store is not None:
                    try:
                        store.close()
                    except Exception:
                        raise BootstrapError("sandbox startup or polling failed") from None
            else:
                raise BootstrapError("sandbox startup or polling failed") from None
        elif store is not None:
            try:
                store.close()
            except Exception:
                raise BootstrapError("sandbox startup or polling failed") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Isolated P2c Telegram test bot")
    parser.add_argument("--check", action="store_true", help="pure configuration/path check; no services started")
    parser.add_argument("--config", dest="config", default=None, help="path to JSON bot configuration")
    args = parser.parse_args(argv)
    cfg = None
    try:
        if args.config is not None:
            try:
                cfg = load_bot_config(args.config)
            except ConfigError:
                raise BootstrapError("configuration invalid") from None
        if args.check:
            preflight(config=cfg)
            print("P2c sandbox configuration OK (no services contacted)")
        else:
            run(config=cfg)
    except BootstrapError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
