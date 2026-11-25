"""Memory write policy: a memory body is a verbatim slice of a user's own message.

Why this exists (P2d evidence): storing an extracted third-person fact
("the user prefers fictional jasmine tea") lets a model re-read that fact as its own
preference. Keeping the user's exact words removes the paraphrase step entirely, so
person and attribution cannot drift: the body is what the user actually typed.

No LLM, no rewriting: the output is always a substring of the input, which makes the
invariant machine-checkable instead of prompt-dependent.
"""
from __future__ import annotations

import re

MAX_MEMORY_TEXT = 400
VALID_KINDS = frozenset({"preference", "event", "promise", "fact"})

# A memory body must be the user's own words, never a third-person description of them.
_THIRD_PERSON = re.compile(r"使用者|用戶|the user|user prefers", re.IGNORECASE)


class MemoryWriteError(Exception):
    """Sanitized write-layer rejection; never echoes private content."""


def build_memory_text(source_message: str, max_length: int = MAX_MEMORY_TEXT) -> str:
    """Return a verified verbatim slice of `source_message` to be stored as a memory body."""
    if not isinstance(source_message, str):
        raise MemoryWriteError("source must be a string")
    if "\x00" in source_message:
        raise MemoryWriteError("invalid source")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= MAX_MEMORY_TEXT:
        raise MemoryWriteError("invalid length limit")
    text = source_message.strip()
    if not text:
        raise MemoryWriteError("empty source")
    if len(text) > max_length:
        text = text[:max_length].rstrip()
        if not text:
            raise MemoryWriteError("empty source")
    if _THIRD_PERSON.search(text):
        raise MemoryWriteError("memory body must be the user's own words")
    return text


def is_verbatim(text: str, source_message: str) -> bool:
    """True when `text` is a contiguous slice of `source_message` (the write-layer invariant)."""
    return isinstance(text, str) and isinstance(source_message, str) and bool(text) and text in source_message


def remember_statement(store, owner_id: str, character_id: str, external_id: str, kind: str,
                       source_message: str, source_ids, domain: str = "real",
                       visible_to=None) -> tuple[str, str]:
    """Persist a user statement as a verbatim memory; returns (memory_id, stored_text)."""
    if kind not in VALID_KINDS:
        raise MemoryWriteError("invalid kind")
    text = build_memory_text(source_message)
    memory_id = store.save_memory(owner_id, character_id, external_id, kind, text,
                                  list(source_ids), domain=domain, visible_to=visible_to)
    return memory_id, text
