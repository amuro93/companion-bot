# Companion Bot — a Telegram companion with long-term memory

![tests](https://github.com/amuro93/companion-bot/actions/workflows/ci.yml/badge.svg)

[中文說明 →](README.md)

A companion-style chat bot built around SQLite that can run entirely on local models. The goal is not question answering — it is a character that remembers what you told it, sounds like a person, and **fails loudly instead of hallucinating** when the model misbehaves.

- **Chat backend:** any OpenAI-compatible `/v1/chat/completions` endpoint (llama.cpp, vLLM, Ollama, hosted APIs)
- **Memory:** SQLite (structured facts + local vector index + async extraction queue)
- **Interface:** Telegram, single owner, private chat

## Highlights

**1. Layered memory instead of dumping history into context**
Short-term history, fixed L0 persona settings, named facts (key-value), semantic retrieval, and a deferred extraction queue are stored separately. Writes go through a single SQLite transaction, so "check, then write" races cannot happen; when a source message is retracted, memories derived from it are excluded too.

**2. Fail-closed startup checks**
The config schema is strict (unknown fields are rejected), a missing persona file stops the bot, model endpoints must be on an explicit allow-list, and credentials are read from environment variables only (the config stores variable *names*). A misconfiguration cannot quietly turn into "it runs but behaves oddly".

**3. Bounded thinking for hybrid-reasoning models**
Supported endpoints receive `reasoning_budget_tokens` (default 128). Low-bit local models often collapse to one or two characters, or an empty reply, when thinking is fully disabled; a small thinking budget makes output far more stable, while a large budget makes it worse. The default came from measurement, not intuition.

**4. Runtime model switching**
`/model` in Telegram lists every configured backend as inline buttons. The choice takes effect immediately and persists across restarts, which makes a "cloud for daily chat, local for sensitive conversations" split practical.

**5. 555 automated tests**
Coverage runs from config validation, memory atomicity, and queue recovery to vector indexing and Telegram handler dispatch. One test drives a real `Application.process_update` to assert that `/model` is not swallowed by another handler — that bug actually happened, so the regression test stayed.

## Design philosophy: store the original sentence, never an LLM summary

Most memory systems (mem0, Letta, and similar) take the "let the LLM condense the conversation into facts" route: a user says "I haven't been sleeping well because the little one wakes up at night", and the system stores "the user has a sleep problem".

That saves context, but it costs three things: the model's misunderstanding gets frozen and cannot be traced back (you only see the summary, never the sentence); when the source is retracted, the derived summary cannot be undone; and two summaries can contradict each other.

This project deliberately does the opposite:

- **Only the user's original sentence is stored** — the model may decide whether a sentence is worth remembering, but it may not rewrite it into a summary
- **Deterministic rules filter first** (family, residence, health, schedules, and so on); only when the rules cannot decide is a local model asked — and if the model's judgement fails, **nothing is stored** rather than guessing
- **Manual memories outrank automatic ones**; a manual entry is never overwritten by the automatic pipeline
- Every correction and retraction leaves a record, and **retracting a source message also removes the memories derived from it**

The trade-off is a longer context and an extra retrieval step. What it buys is that every memory can be traced back to what the user actually said.

The endpoint allow-list and the fail-closed startup checks are the same principle applied elsewhere: fail explicitly rather than silently doing the wrong thing.

## Architecture

```
Telegram  ──►  telegram_test_adapter   (handler layering, permissions, /model buttons)
                    │
                    ▼
               companion_chat          (conversation assembly, persona injection, retrieval)
                 │        │
                 │        └──► model_router ──► OpenAI-compatible endpoint
                 │
                 ▼
            memory_* layer
              ├─ memory_core          structured facts, single-transaction writes
              ├─ memory_index/search  indexing and retrieval
              ├─ memory_sqlite_vector local vector index
              ├─ memory_deferred_*    deferred extraction (queue / worker / processor)
              ├─ memory_gate / trigger pre-write safety gate and triggers
              └─ memory_http          internal memory service (loopback only)
```

Key modules:

| Module | Responsibility |
|---|---|
| `bootstrap_test_bot.py` | Entry point, preflight checks, dependency wiring |
| `bot_config.py` | Config schema, strict validation, endpoint allow-list |
| `companion_chat.py` | Single-turn conversation flow and context assembly |
| `persona_l0.py` | Persona and L0 layered memory loading |
| `memory_core.py` | Fact storage and transaction boundaries |
| `model_router.py` | Multi-backend switching and persistent selection |
| `telegram_test_adapter.py` | Telegram handlers and interaction |

## Quick start

Requirements: Python 3.11+, and one OpenAI-compatible model endpoint.

```bash
git clone https://github.com/amuro93/companion-bot.git
cd companion-bot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp model-auth.synthetic.example.json bot.json
# edit bot.json: set model_url / model_id, point persona_file at your character

export BOT_TELEGRAM_TOKEN="your bot token"       # from @BotFather
export BOT_TELEGRAM_USER_ID="your numeric Telegram id"
export MODEL_API_KEY="key for the endpoint (empty string if not needed)"

python bootstrap_test_bot.py --check --config bot.json   # validate config only, no network
python bootstrap_test_bot.py --config bot.json           # start for real
```

`--check` validates config fields, paths, the persona file and the endpoint allow-list without starting the service or contacting Telegram — worth running after every config change.

## Configuration

Minimum viable config:

```json
{
  "root": ".",
  "db": "companion.sqlite3",
  "owner_id": "local-owner",
  "character_id": "example-companion",
  "persona_file": "personas/example_companion.txt",
  "l0_dir": "personas",
  "model_url": "http://localhost:8080/v1/chat/completions",
  "model_id": "your-model-name",
  "telegram_token_env": "BOT_TELEGRAM_TOKEN",
  "telegram_user_id_env": "BOT_TELEGRAM_USER_ID",
  "model_api_key_env": "MODEL_API_KEY"
}
```

- `*_env` fields hold **environment variable names**, not values. The config can be committed; credentials never are.
- `model_url` must be loopback or explicitly listed in the allow-list in `bot_config.py` (so a key cannot be shipped to an arbitrary domain).
- Optional `model_choices` enables `/model` switching between several backends.

## Switching models

```json
"model_choices": [
  {"label": "Cloud", "model_url": "https://api.example.com/chat/completions",
   "model_id": "your-cloud-model", "model_api_key_env": "CLOUD_API_KEY"},
  {"label": "Local GPU", "model_url": "http://localhost:8080/v1/chat/completions",
   "model_id": "your-local-model", "model_api_key_env": "MODEL_API_KEY"}
]
```

The first entry is the default. `/model` in Telegram shows one inline button per backend; pressing one switches immediately and writes the choice to `model-choice.json` next to the database.

## Thinking budget

| Endpoint | Behaviour |
|---|---|
| Generic endpoint | `chat_template_kwargs: {"enable_thinking": false}` by default |
| Budget-aware endpoint | sends `reasoning_budget_tokens` (default 128) |
| `THINK_MODE=off` | thinking explicitly disabled |

Measured on Qwen-family models at 3-bit / 4-bit: with thinking fully disabled, roughly a third of replies came back shorter than five characters or empty; 64 tokens was still unstable; 128 was stable; 150 was worse. If you use a different model, measure before trusting these numbers.

## Tests

```bash
python -m unittest discover -s . -p "test_*.py" -t .
```

555 tests. No network, no real model, no Telegram token required — every external call is replaced with a stub. CI runs the same suite on Python 3.11 and 3.12.

## Design decisions

- **SQLite over Postgres**: single user, must work offline, backup is one file. The cost is no horizontal scaling — this project does not need it.
- **Single owner**: `owner_id` runs through every table, and messages from anyone else are rejected at the handler layer. This is not a multi-tenant design.
- **No framework rewrite**: FastAPI/ORM would blur the transaction boundaries of the memory layer; the SQL is written directly so `BEGIN IMMEDIATE` scope and rollback range stay explicit.
- **The character is a character**: the persona file states outright that it is fiction and must admit it when asked directly.

## Status

A personal project in daily use; not yet validated for multi-instance or large-scale deployment. Issues and design discussions are welcome.

## License

MIT — see `LICENSE`.
