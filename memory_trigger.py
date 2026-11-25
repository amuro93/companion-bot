"""Decide which user utterances become memories. Rules only: no model, no rewriting.

Design stance (P2g): a memory may only be text the user actually typed. This module
picks at most one sentence per incoming message and hands the verbatim slice to
`memory_write.remember_statement`; generated text has no path into the store.
Bias on purpose: a missed memory is cheaper than a wrong one.
"""
from __future__ import annotations

import hashlib
import re

MAX_CANDIDATES = 1
MIN_SENTENCE = 4
MAX_MESSAGE = 1024
# "我" must appear this early in the sentence to count as a first-person statement.
FIRST_PERSON_WINDOW = 6

_PREFERENCE = re.compile(r"我(?:也|最|比較|很|超|真|有點|稍微|不太|不)?(?:喜歡|愛|討厭|怕|習慣)")
_NEGATIVE_EATING = re.compile(r"我(?:不太|不)(?:吃|喝|能)")
_HABIT = re.compile(r"我.{0,6}?(?:每天|都會|平常|最近|這陣子|通常|每次都)")
_FAMILY = re.compile(r"我(?:的)?(?:老婆|老公|太太|先生|兒子|女兒|小孩|媽|爸|婆婆|公公|大寶|小明|家人)")
_EXPERIENCE = re.compile(r"我.{0,6}?(?:去過|做過|買了|帶了|學過|以前|上次|昨天|前天|前幾天|小時候)")
_PATTERNS = (_PREFERENCE, _NEGATIVE_EATING, _HABIT, _FAMILY, _EXPERIENCE)
_SPLIT = re.compile(r"[。！!；;\n]+")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SPLIT.split(text) if part.strip()]


def is_candidate(sentence: str) -> bool:
    """True when this single sentence states something about the user themselves."""
    text = sentence.strip()
    if len(text) < MIN_SENTENCE:
        return False
    if "?" in text or "？" in text:
        return False
    if "我" not in text[:FIRST_PERSON_WINDOW]:
        return False
    return any(pattern.search(text) for pattern in _PATTERNS)


def candidates(message: str) -> list[str]:
    """At most MAX_CANDIDATES verbatim sentences worth storing, in message order."""
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
        return []
    return [sentence for sentence in split_sentences(message) if is_candidate(sentence)][:MAX_CANDIDATES]


def kind_of(sentence: str) -> str:
    if _PREFERENCE.search(sentence) or _NEGATIVE_EATING.search(sentence):
        return "preference"
    if _FAMILY.search(sentence):
        return "fact"
    if _EXPERIENCE.search(sentence):
        return "event"
    return "fact"


def message_external_id(content: str) -> str:
    """Content-addressed: the same words map to the same message row."""
    return "tg-" + hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]


def memory_external_id(owner_id: str, character_id: str, sentence: str) -> str:
    """Content-addressed: repeating a sentence never creates a second memory."""
    digest = hashlib.sha256(f"{owner_id}|{character_id}|{sentence}".encode("utf-8")).hexdigest()[:16]
    return "auto-" + digest


def gated_candidates(message: str, gate, max_memories: int = 2) -> list[str]:
    """Rules prefilter, then the local model answers yes/no per sentence (P2j).

    The gate never supplies text: every returned sentence is a verbatim slice of the
    user's own message, so generated prose still has no path into the store.
    """
    from memory_gate import prefilter, split_sentences

    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE:
        return []
    if type(max_memories) is not int or not 1 <= max_memories <= 5:
        raise ValueError("invalid max_memories")
    picked = []
    for sentence in split_sentences(message):
        if not prefilter(sentence):
            continue
        try:
            verdict = gate.remember(sentence)
        except Exception:
            verdict = False
        if verdict is True:
            picked.append(sentence)
            if len(picked) >= max_memories:
                break
    return picked


def record_user_message(store, owner_id: str, character_id: str, content: str,
                        gate=None, max_memories: int = 2) -> tuple[str, list[str]]:
    """Append the message, then persist up to `max_memories` verbatim memories from it.

    Returns (message_id, [memory_id, ...]). The message row is written even when nothing
    qualifies as a memory, because the conversation is the raw material. Without a gate
    the original keyword rules decide; with a gate the local model decides per sentence.
    """
    from memory_write import remember_statement

    message_id = store.append_message(owner_id, character_id, message_external_id(content),
                                      "user", content)
    picked = candidates(content) if gate is None else gated_candidates(content, gate, max_memories)
    memory_ids = []
    for sentence in picked:
        try:
            memory_id, _body = remember_statement(
                store, owner_id, character_id,
                memory_external_id(owner_id, character_id, sentence),
                kind_of(sentence), sentence, [message_id], visible_to=[character_id])
        except ValueError as exc:
            # Same sentence reached us from a different message: the memory already
            # exists with a different source link. Keep the first one, do not fail.
            if "dempotency" not in str(exc):
                raise
            continue
        memory_ids.append(memory_id)
    return message_id, memory_ids
