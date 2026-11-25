"""Manual memory entry (P2o-2): button flow, validation and value mapping.

Pure helpers only — no Telegram, no I/O — so the flow can be unit tested offline.
The Telegram adapter turns these (label, callback_data) rows into InlineKeyboardMarkup
and the writer in memory_admin.MemoryAdmin persists the result with source='manual'.
"""
from __future__ import annotations

import hashlib

import memory_keys as mk

MAX_CONTENT_LEN = 40
CALLBACK_ADD = "mem:add"
CALLBACK_SKIP = "mem:skip"
CALLBACK_CANCEL = "mem:cancel"
PENDING_TIMEOUT_SECONDS = 60

ADD_BUTTON_LABEL = "➕ 新增記憶"
SKIP_LABEL = "略過（用分類當主題）"
CANCEL_LABEL = "✖ 取消"

PROMPT_CATEGORY = "要記哪一類？（點一個）"
PROMPT_TOPIC = "主題是什麼？（6 字以內，例：茶、血壓、這個小鎮）"
PROMPT_CONTENT = "內容是什麼？（40 字以內，例：很愛喝高山烏龍，越濃越好）"

# Category -> the memories.kind that best matches it.
KIND_BY_CATEGORY: dict[str, str] = {
    "喜好": "preference",
    "飲食": "preference",
    "健康": "fact",
    "家人": "fact",
    "工作與身分": "fact",
    "住居與地緣": "fact",
    "經歷": "event",
    "約定與計畫": "promise",
    "親密": "preference",
}


def category_at(index: int) -> str:
    """Return the category name for a 0-based button index."""
    if type(index) is not int or not 0 <= index < len(mk.CATEGORIES):
        raise ValueError("invalid category index")
    return mk.CATEGORIES[index]


def category_index(category: str) -> int:
    if category not in mk.CATEGORIES:
        raise ValueError("unknown category")
    return mk.CATEGORIES.index(category)


def category_rows() -> list[list[tuple[str, str]]]:
    """Nine category buttons, three per row (two rows of 3 + one row of 3)."""
    rows: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for index, category in enumerate(mk.CATEGORIES):
        row.append((category, f"mem:cat:{index}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def topic_rows(category: str) -> list[list[tuple[str, str]]] | None:
    """Fixed-topic categories get their own buttons; free ones return None."""
    if category not in mk.CATEGORIES:
        raise ValueError("unknown category")
    topics = mk.FIXED_TOPICS.get(category)
    if not topics:
        return None
    index = category_index(category)
    rows: list[list[tuple[str, str]]] = []
    row: list[tuple[str, str]] = []
    for topic_index, topic in enumerate(topics):
        row.append((topic, f"mem:top:{index}:{topic_index}"))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def topic_at(category: str, topic_index: int) -> str:
    topics = mk.FIXED_TOPICS.get(category)
    if not topics:
        raise ValueError("category has no fixed topics")
    if type(topic_index) is not int or not 0 <= topic_index < len(topics):
        raise ValueError("invalid topic index")
    return topics[topic_index]


def clean_topic(raw) -> str:
    """Validate a typed topic (free categories only)."""
    if not isinstance(raw, str):
        raise ValueError("topic must be text")
    topic = "".join(raw.split())
    if not topic:
        raise ValueError("topic must not be empty")
    if "\x00" in topic:
        raise ValueError("topic must not contain NUL")
    if len(topic) > mk.MAX_TOPIC_LEN:
        raise ValueError(f"topic must be <= {mk.MAX_TOPIC_LEN} chars")
    return topic


def clean_content(raw) -> str:
    """Validate the memory body the user typed."""
    if not isinstance(raw, str):
        raise ValueError("content must be text")
    body = " ".join(raw.split())
    if not body:
        raise ValueError("content must not be empty")
    if "\x00" in body:
        raise ValueError("content must not contain NUL")
    if len(body) > MAX_CONTENT_LEN:
        raise ValueError(f"content must be <= {MAX_CONTENT_LEN} chars")
    return body


def kind_for(category: str) -> str:
    if category not in KIND_BY_CATEGORY:
        raise ValueError("unknown category")
    return KIND_BY_CATEGORY[category]


def build_key(category: str, topic: str | None) -> str:
    """Compose the canonical meta_key for a manual entry."""
    if category not in mk.CATEGORIES:
        raise ValueError("unknown category")
    fixed = mk.FIXED_TOPICS.get(category)
    if fixed:
        if topic is None:
            raise ValueError("this category needs a fixed topic")
        if topic not in fixed:
            raise ValueError("topic not allowed for this category")
        return f"{category}/{topic}"
    if topic is None:
        return category
    return f"{category}/{topic}"


def content_fingerprint(owner_id: str, character_id: str, key: str, content: str) -> str:
    """Content-addressed id: re-typing the same entry does not create a second row."""
    raw = "\x1f".join((owner_id, character_id, key, content)).encode("utf-8")
    return "manual-" + hashlib.sha256(raw).hexdigest()[:32]


def preview_line(item: dict, number: int) -> str:
    """One listing line: category/topic, manual lock, then the body."""
    body = " ".join(str(item.get("text") or "").split())
    if len(body) > 38:
        body = body[:38] + "…"
    key = item.get("meta_key")
    tag = key if isinstance(key, str) and key else "（未分類）"
    lock = "🔒 " if item.get("source") == "manual" else ""
    return f"{number}. {lock}{tag} → {body}"
