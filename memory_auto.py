"""Automatic memory writes with a fixed key (P2o-3).

The P2j/P2k pipeline stays intact: rules prefilter, the local model answers yes/no,
and the stored body is still the user's own sentence. What changes is where it lands —
one active memory per key, manual entries win, deleted keys stay deleted until the user
writes them again.
"""
from __future__ import annotations

import re

import memory_keys as mk
import memory_trigger as mt

MAX_TOPIC_LEN = mk.MAX_TOPIC_LEN

# Family sentences must land on a fixed topic; guessing wrong is worse than parking it.
_FAMILY_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("配偶", ("老婆", "老公", "太太", "先生", "妻子", "配偶")),
    ("子女", ("兒子", "女兒", "小孩", "孩子", "大寶", "小明")),
    ("長輩", ("爸爸", "媽媽", "父親", "母親", "公婆", "婆婆", "公公", "長輩", "我爸", "我媽")),
)

_FOOD_HINTS = ("吃", "喝", "口味", "菜", "飯", "茶", "咖啡", "酒", "甜", "鹹", "辣", "鹹酥")

# Loose on purpose: memory_trigger's pattern anchors "我" to the first few characters,
# which misses "我最近很愛喝…" and pushes it into the habit branch by mistake.
_LOOSE_PREFERENCE = re.compile(r"(?:愛|喜歡|討厭|不吃|不喝|不太吃|不太喝)")

# Leading filler that should not become a topic. "我們" is a real subject, never filler.
_LEADING = ("我最近", "我平常", "我這陣子", "我通常", "我現在", "我也", "我比較",
            "最近", "平常", "這陣子", "後來", "然後", "以前")


def _family_topic(sentence: str) -> str | None:
    for topic, words in _FAMILY_TOPICS:
        if any(word in sentence for word in words):
            return topic
    return None


def _provisional_key(sentence: str) -> str:
    body = "".join(sentence.split())
    if not body.startswith("我們"):
        # "我們…" is a real subject and must stay intact.
        for prefix in _LEADING:
            if body.startswith(prefix):
                body = body[len(prefix):]
                break
        if body.startswith("我") and len(body) > 1:
            body = body[1:]
    body = body[:MAX_TOPIC_LEN] or "未分類"
    return f"經歷/{body}"


def derive_key(sentence) -> tuple[str, bool]:
    """Return (meta_key, needs_review) for one gated sentence.

    Deterministic: the same sentence always produces the same key, so repeated
    statements overwrite instead of piling up.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("sentence must be text")

    # 1. An alias spelled out in the sentence is the most specific signal.
    for alias in sorted(mk.ALIASES, key=len, reverse=True):
        if alias in sentence:
            return mk.ALIASES[alias], False

    # 2. Family: only fixed topics are allowed, so an unknown relative is parked.
    if mt._FAMILY.search(sentence):
        topic = _family_topic(sentence)
        if topic is not None:
            return f"家人/{topic}", False
        return _provisional_key(sentence), True

    # 3. Residence / habitual places.
    if "住" in sentence and "住院" not in sentence:
        return "住居與地緣/住處", False

    # 4. Preference / eating habits.
    if (
        mt._PREFERENCE.search(sentence)
        or mt._NEGATIVE_EATING.search(sentence)
        or _LOOSE_PREFERENCE.search(sentence)
    ):
        if any(hint in sentence for hint in _FOOD_HINTS):
            return "飲食/口味", False
        return "喜好", False

    # 5. Recurring habit.
    if mt._HABIT.search(sentence):
        return "健康/習慣", False

    # 6. Everything else is an experience and parked for review.
    return _provisional_key(sentence), True


def record_user_message_keyed(
    store,
    owner_id: str,
    character_id: str,
    content: str,
    gate=None,
    max_memories: int = 2,
    keyer=None,
) -> dict:
    """Store the message, then persist gated sentences under their keys.

    `keyer` (P2o-4 KeyDecision) decides twice in one call: remember or not, and which
    key. When it is absent the P2j gate decides and `derive_key` names the key.

    Returns {"message_id", "memory_ids", "results", "trace"}; each result carries its own
    action (insert / correct / unchanged / skipped / blocked / tombstoned / duplicate).
    """
    message_id = store.append_message(
        owner_id,
        character_id,
        mt.message_external_id(content),
        "user",
        content,
    )
    if store.is_message_excluded(owner_id, character_id, message_id):
        return {
            "message_id": message_id,
            "memory_ids": [],
            "results": [{"key": None, "action": "tombstoned", "reason": "source_excluded"}],
            "trace": [{"stage": "source_exclusion", "action": "tombstoned", "reason": "source_excluded"}],
        }
    if store.is_course_reschedule(content):
        external_id = mt.memory_external_id(owner_id, character_id, content)
        memory_id = store.reschedule_course(owner_id, character_id, external_id, content, message_id)
        action = "correct" if memory_id else "needs_clarification"
        return {"message_id": message_id, "memory_ids": [memory_id] if memory_id else [],
                "results": [{"action": action}], "trace": [{"stage": "course_reschedule", "action": action}]}
    memory_ids: list[str] = []
    results: list[dict] = []
    trace: list[dict] = []
    decisions: list[tuple[str, str | None, bool, str | None, list[dict]]] = []
    if keyer is not None:
        from memory_gate import prefilter, split_sentences

        for sentence in split_sentences(content):
            if len(decisions) >= max_memories:
                trace.append({
                    "sentence": sentence,
                    "stage": "budget",
                    "action": "max_memories_truncated",
                    "limit": max_memories,
                })
                continue

            if not prefilter(sentence):
                trace.append({
                    "sentence": sentence,
                    "stage": "prefilter_rule",
                    "action": "rejected",
                })
                continue

            # Same words already stored: keep the first key, never ask the model twice
            # (a second opinion used to file the identical sentence under a new key).
            already = store.find_by_external_id(
                owner_id, character_id, mt.memory_external_id(owner_id, character_id, sentence)
            )
            if already is not None:
                results.append({
                    "key": already.get("meta_key"),
                    "needs_review": False,
                    "sentence": sentence,
                    "action": "already_remembered",
                })
                trace.append({
                    "sentence": sentence,
                    "stage": "dedup",
                    "action": "already_remembered",
                    "key": already.get("meta_key"),
                })
                continue

            try:
                decision = keyer.decide(sentence)
            except Exception:
                decision = {"key": None, "how": "error", "trace": [{"stage": "decision", "how": "error"}]}

            dec_trace = decision.get("trace") or []
            if not decision.get("key"):
                trace.append({
                    "sentence": sentence,
                    "stage": "key_decision",
                    "action": (
                        "refused" if decision.get("how") in ("no", "prefilter_no")
                        else "error"
                    ),
                    "how": decision.get("how"),
                    "needs_review": bool(decision.get("needs_review", False)),
                    "decision_trace": dec_trace,
                })
                continue

            decided_key = str(decision["key"])
            decided_review = bool(decision.get("needs_review", False))
            decided_how = decision.get("how")
            trace.append({
                "sentence": sentence,
                "stage": "key_decision",
                "action": "accepted",
                "key": decided_key,
                "how": decided_how,
                "needs_review": decided_review,
                "decision_trace": dec_trace,
            })
            decisions.append((sentence, decided_key, decided_review, decided_how, dec_trace))
    else:
        picked = (
            mt.candidates(content)
            if gate is None
            else mt.gated_candidates(content, gate, max_memories)
        )
        decisions = [(sentence, None, False, None, []) for sentence in picked]

    for sentence, decided_key, decided_review, how, dec_trace in decisions:
        if decided_key is None:
            key, needs_review = derive_key(sentence)
        else:
            key, needs_review = decided_key, decided_review
        entry: dict = {"key": key, "needs_review": needs_review, "sentence": sentence}
        if how is not None:
            entry["how"] = how

        if store.is_key_blocked(owner_id, character_id, key):
            entry["action"] = "blocked"
            results.append(entry)
            trace.append({
                "sentence": sentence,
                "stage": "write",
                "action": "blocked",
                "key": key,
            })
            continue

        external_id = mt.memory_external_id(owner_id, character_id, sentence)
        try:
            outcome = store.save_keyed_memory(
                owner_id,
                character_id,
                external_id,
                mt.kind_of(sentence),
                sentence,
                [message_id],
                key,
                source="auto",
                visible_to=[character_id],
            )
        except ValueError as exc:
            message = str(exc)
            if "resurrect" in message.lower():
                entry["action"] = "tombstoned"
            elif "dempotency" in message:
                entry["action"] = "duplicate"
            else:
                raise
            results.append(entry)
            trace.append({
                "sentence": sentence,
                "stage": "write",
                "action": entry["action"],
                "key": key,
            })
            continue

        entry.update(outcome)
        results.append(entry)
        trace.append({
            "sentence": sentence,
            "stage": "write",
            "action": outcome.get("action"),
            "key": key,
            "memory_id": outcome.get("memory_id"),
        })
        if outcome["action"] in ("insert", "correct"):
            memory_ids.append(outcome["memory_id"])

    return {"message_id": message_id, "memory_ids": memory_ids, "results": results, "trace": trace}
