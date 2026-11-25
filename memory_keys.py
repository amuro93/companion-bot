"""Fixed key taxonomy for companion memory (P2o-1).

Key format: "<category>/<topic>".

The category is a closed set of 9; the topic is a short label (<= 6 chars) picked by
the user (manual entry) or derived by the extractor. Aliases map the many ways a model
phrases the same subject back onto one canonical key, so a repeated or corrected fact
overwrites the active row instead of piling up (same idea as character-studio's
user_memory(key, value) ON CONFLICT upsert).

Standard library only; no I/O, no model calls.
"""

from __future__ import annotations

import re

MAX_TOPIC_LEN = 6

# Closed set. Adding a category is a schema decision, not a runtime one.
CATEGORIES: tuple[str, ...] = (
    "喜好",
    "飲食",
    "健康",
    "家人",
    "工作與身分",
    "住居與地緣",
    "經歷",
    "約定與計畫",
    "親密",
)

# Categories whose topics are enumerations: letting other topics through would make
# spouse/children/elders (or two different sexual preferences) overwrite each other.
FIXED_TOPICS: dict[str, tuple[str, ...]] = {
    "家人": ("配偶", "子女", "長輩"),
    "親密": ("性偏好", "身體反應", "情境偏好", "節奏與習慣", "界線與禁忌", "稱呼與挑逗"),
}

# Free-text subject -> canonical key. Longest aliases first at match time.
#
# EXTRA_TOPICS: topics the extractor may pick but that have no natural alias word in
# everyday speech ("小明的英文課" contains no keyword that maps to 子女課程). They feed
# the model's suggestion menu only; ALIASES stays the deterministic rule path.
EXTRA_TOPICS: dict[str, tuple[str, ...]] = {
    "約定與計畫": ("子女課程", "子女就醫", "長輩照顧", "家務分工", "採買", "作息", "出遊"),
    "飲食": ("下廚",),
    "健康": ("運動傷害",),
    "工作與身分": ("職稱",),
    "住居與地緣": ("家鄉", "社區"),
    "經歷": ("當兵", "過去工作"),
}

ALIASES: dict[str, str] = {
    # 喜好
    "興趣": "喜好", "嗜好": "喜好", "愛好": "喜好", "休閒": "喜好", "娛樂": "喜好",
    "茶": "喜好/茶", "喝茶": "喜好/茶", "泡茶": "喜好/茶", "咖啡": "喜好/咖啡",
    "喝酒": "喜好/酒", "小酌": "喜好/酒", "酒量": "喜好/酒",
    "音樂": "喜好/音樂", "電影": "喜好/影劇", "影集": "喜好/影劇",
    "運動": "喜好/運動", "閱讀": "喜好/閱讀", "收藏": "喜好/收藏", "散步": "喜好/散步",
    # 飲食
    "口味": "飲食/口味", "飲食": "飲食/口味", "吃什麼": "飲食/口味",
    "忌口": "飲食/忌口", "不吃": "飲食/忌口", "過敏": "飲食/忌口",
    "三餐": "飲食/三餐習慣", "外食": "飲食/外食",
    # 健康
    "睡眠": "健康/睡眠", "失眠": "健康/睡眠", "睡不好": "健康/睡眠",
    "血壓": "健康/血壓", "血糖": "健康/血壓", "慢性病": "健康/血壓",
    "用藥": "健康/用藥", "吃藥": "健康/用藥", "就醫": "健康/就醫", "看醫生": "健康/就醫",
    "體重": "健康/體態", "身材": "健康/體態",
    # 家人
    "老婆": "家人/配偶", "太太": "家人/配偶", "妻子": "家人/配偶", "配偶": "家人/配偶",
    "先生": "家人/配偶", "老公": "家人/配偶",
    "兒子": "家人/子女", "女兒": "家人/子女", "小孩": "家人/子女", "孩子": "家人/子女",
    "大寶": "家人/子女", "小明": "家人/子女",
    "爸爸": "家人/長輩", "媽媽": "家人/長輩", "父母": "家人/長輩",
    "長輩": "家人/長輩", "公婆": "家人/長輩", "婆婆": "家人/長輩", "岳父": "家人/長輩",
    # 工作與身分
    "工作": "工作與身分/職業", "職業": "工作與身分/職業", "上班": "工作與身分/職業",
    "職稱": "工作與身分/職業",
    "退休": "工作與身分/職涯", "退伍": "工作與身分/職涯", "當兵": "工作與身分/職涯",
    "軍職": "工作與身分/職涯",
    "收入": "工作與身分/收入來源", "薪水": "工作與身分/收入來源",
    # 住居與地緣
    "住處": "住居與地緣/住處", "住哪": "住居與地緣/住處", "住居": "住居與地緣/住處",
    "家鄉": "住居與地緣/住處", "戶籍": "住居與地緣/住處",
    "常去": "住居與地緣/常去地點", "河邊": "住居與地緣/常去地點",
    "公園": "住居與地緣/常去地點",
    # 經歷
    "學歷": "經歷/學歷", "往事": "經歷/重大事件", "經歷": "經歷/重大事件",
    # 約定與計畫
    "約定": "約定與計畫/約定", "答應": "約定與計畫/約定", "承諾": "約定與計畫/約定",
    "計畫": "約定與計畫/計畫", "待辦": "約定與計畫/待辦", "打算": "約定與計畫/計畫",
    # 親密
    "性偏好": "親密/性偏好", "喜歡的體位": "親密/性偏好", "敏感帶": "親密/身體反應",
    "身體反應": "親密/身體反應", "生理反應": "親密/身體反應",
    "情境": "親密/情境偏好", "情境偏好": "親密/情境偏好",
    "節奏": "親密/節奏與習慣", "前戲": "親密/節奏與習慣",
    "界線": "親密/界線與禁忌", "禁忌": "親密/界線與禁忌",
    "稱呼": "親密/稱呼與挑逗", "叫法": "親密/稱呼與挑逗", "挑逗": "親密/稱呼與挑逗",
}

_SPACE = re.compile(r"\s+")


def _clean(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    cleaned = _SPACE.sub("", value).strip()
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in cleaned:
        raise ValueError(f"{name} must not contain NUL")
    return cleaned


def category_of(key: str) -> str:
    """Return the category segment of a canonical key, or raise."""
    return validate_key(key).split("/", 1)[0]


def is_intimate(key: str) -> bool:
    """True when the memory must only be recalled inside an intimate context."""
    return category_of(key) == "親密"


def resolve_alias(token: str) -> str | None:
    """Map a free-text subject onto a canonical key, or None when unknown."""
    cleaned = _clean(token, "token")
    if cleaned in ALIASES:
        return ALIASES[cleaned]
    # exact "category/topic" already canonical?
    if "/" in cleaned:
        try:
            return validate_key(cleaned)
        except ValueError:
            pass
    if cleaned in CATEGORIES:
        return cleaned
    # longest alias that is a substring of the token, e.g. "喝高山烏龍的喜好".
    # Single-character aliases are token-only: as substrings they match everything.
    for alias in sorted((a for a in ALIASES if len(a) >= 2), key=len, reverse=True):
        if alias in cleaned:
            return ALIASES[alias]
    # finally, a category word mentioned anywhere in the free text
    for category in sorted(CATEGORIES, key=len, reverse=True):
        if category in cleaned:
            return category
    return None


def validate_key(key: str) -> str:
    """Validate "category" or "category/topic" and return it unchanged."""
    cleaned = _clean(key, "key")
    parts = cleaned.split("/")
    if len(parts) > 2:
        raise ValueError("key must be '<category>' or '<category>/<topic>'")
    category = parts[0]
    if category not in CATEGORIES:
        raise ValueError(f"unknown category: {category!r}")
    if len(parts) == 1:
        return category
    topic = parts[1]
    if not topic:
        raise ValueError("topic must not be empty")
    if len(topic) > MAX_TOPIC_LEN:
        raise ValueError(f"topic must be <= {MAX_TOPIC_LEN} chars")
    fixed = FIXED_TOPICS.get(category)
    if fixed is not None and topic not in fixed:
        raise ValueError(f"category {category!r} only allows topics {fixed}")
    return f"{category}/{topic}"


def normalize_key(raw: str, topic: str | None = None) -> tuple[str, bool]:
    """Coerce a category/topic (or a free subject) into a canonical key.

    Returns (key, needs_review). needs_review is True when nothing matched and the
    caller must treat the key as provisional (P2o spec 3.2: mark for human review).
    """
    if topic is not None:
        category_raw = _clean(raw, "category")
        resolved = resolve_alias(category_raw)
        if resolved is None:
            raise ValueError(f"unknown category: {category_raw!r}")
        category = resolved.split("/", 1)[0]
        topic_clean = _clean(topic, "topic")
        fixed = FIXED_TOPICS.get(category)
        if fixed is not None:
            if topic_clean in fixed:
                return f"{category}/{topic_clean}", False
            raise ValueError(f"category {category!r} only allows topics {fixed}")
        if len(topic_clean) > MAX_TOPIC_LEN:
            topic_clean = topic_clean[:MAX_TOPIC_LEN]
        return f"{category}/{topic_clean}", False

    cleaned = _clean(raw, "key")
    hit = resolve_alias(cleaned)
    if hit is not None:
        return validate_key(hit), False
    # Unknown subject: park it under 經歷 so it is still recalled, and flag it.
    provisional = f"經歷/{cleaned[:MAX_TOPIC_LEN]}"
    return validate_key(provisional), True


def split_key(key: str) -> tuple[str, str | None]:
    """Return (category, topic_or_None) for a validated key."""
    parts = validate_key(key).split("/", 1)
    return parts[0], (parts[1] if len(parts) == 2 else None)
