"""Model-chosen memory keys with similarity candidates (P2o-4).

Division of labour (deliberately serial, never parallel):
  * B (CandidateRetriever) only *retrieves* the keys that already exist for this scope.
  * A (ModelKeyChooser) makes the one and only decision — remember or not, and in which key.
  * If A cannot answer (transport error, timeout, out-of-vocabulary reply), B decides:
    the top hit above `floor` wins, otherwise a rule-derived key is used.

B never overwrites A: one sentence, one decision.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

from memory_gate import RETENTION_POLICY, SHARED_RETENTION_POLICY
import memory_keys as mk

DEFAULT_FLOOR = 0.80
MAX_CANDIDATES = 5
MAX_CHOICE_CHARS = 24

_CATEGORY_LINE = "、".join(mk.CATEGORIES)


def _topics_by_category() -> dict[str, tuple[str, ...]]:
    """Closed topic menu per category, derived from the alias book."""
    out: dict[str, set[str]] = {category: set() for category in mk.CATEGORIES}
    for category, topics in mk.FIXED_TOPICS.items():
        out[category].update(topics)
    for category, topics in mk.EXTRA_TOPICS.items():
        if category in out:
            out[category].update(topics)
    for value in mk.ALIASES.values():
        if "/" in value:
            category, topic = value.split("/", 1)
            if category in out:
                out[category].add(topic)
    return {category: tuple(sorted(topics)) for category, topics in out.items() if topics}


_TOPIC_MENU = "\n".join(
    f"- {category}：{'、'.join(topics)}" for category, topics in _topics_by_category().items()
)

# Few-shot: the model kept stuffing sentences into "looks related" topics when the
# menu had no exact match ("洗車" -> 喜好/收藏). The last example of each block teaches
# the two failure modes: fall back to the bare category, and answer 否 for non-facts.
_EXAMPLES = (
    ("我偶爾喝點小酒", "喜好/酒"),
    ("我習慣自己洗車，一個月洗兩次", "喜好"),
    ("我假日喜歡自己修東西，水電都自己來", "喜好"),
    ("我拍照都只用一顆八五定焦鏡", "喜好"),
    ("我喜歡在河邊走一走", "喜好/散步"),
    ("我每天早上六點起床", "健康/睡眠"),
    ("我血壓有點高，醫生叫我少鹽", "健康/血壓"),
    ("我吃東西都點小辣以下", "飲食/口味"),
    ("我太太在醫院輪班，常常要加班", "家人/配偶"),
    ("我家老大念國中", "家人/子女"),
    ("我們住海邊小鎮，濕氣比較重", "住居與地緣/住處"),
    ("我以前是職業軍人，退伍幾年了", "工作與身分/職涯"),
    ("我打算下個月帶小孩去看眼睛", "約定與計畫/計畫"),
    ("你喜歡喝茶嗎？", "否"),
    ("今天天氣真好", "否"),
    ("哈哈真的假的", "否"),
    ("我等等要去倒垃圾", "否"),
)

_EXAMPLE_BLOCK = "\n".join(f"句子：{text}\n答案：{answer}" for text, answer in _EXAMPLES)

KEY_SYSTEM = (
    "你是記憶守門員兼分類器。使用者說了一句話，先判斷值不值得長期記住，再決定放哪一格。\n"
    f"{SHARED_RETENTION_POLICY}\n"
    f"可用分類：{_CATEGORY_LINE}。\n"
    "每個分類只接受下列主題（沒有合適的就只回分類、不要自創主題）：\n"
    f"{_TOPIC_MENU}\n"
    "只回覆一種答案，不要解釋：\n"
    "1. 不值得長期記住 → 回：否\n"
    "2. 要記 → 回：分類 或 分類/主題（例：喜好/茶）\n"
    "3. 若下方既有主題清單裡有同一件事，直接整串照抄那個主題\n"
    "主題菜單裡沒有合適的就只回分類（例：洗車→喜好），不要自創、不要湊近似詞。\n"
    "\n以下示範（照這個判斷標準）：\n"
    f"{_EXAMPLE_BLOCK}\n"
)


# Topics the model is allowed to invent. Everything else is coerced down to the bare
# category: without a closed topic space, every phrasing of the same fact becomes a
# new key ("血壓值" / "血壓習慣" / "少鹽") and the merge never happens.
ALLOWED_TOPICS: frozenset[str] = frozenset(
    {topic for topics in mk.FIXED_TOPICS.values() for topic in topics}
    # EXTRA_TOPICS 一定要在這裡，否則模型選了也會被 coerce_key 砍回純分類
    | {topic for topics in mk.EXTRA_TOPICS.values() for topic in topics}
    | {value.split("/", 1)[1] for value in mk.ALIASES.values() if "/" in value}
)


def coerce_key(key: str) -> str:
    """Drop a topic the model invented when it is outside the allowed topic set."""
    category, _, topic = key.partition("/")
    if topic and topic not in ALLOWED_TOPICS:
        return category
    return key


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def parse_choice(raw, candidates=()) -> tuple[str | None, str]:
    """Map a model reply to (key, how). key None means "do not remember"."""
    if not isinstance(raw, str):
        return None, "invalid"
    answer = _normalize(raw)
    if not answer or len(answer) > MAX_CHOICE_CHARS:
        return None, "invalid"
    if answer in {"否", "不是", "no", "NO", "No"}:
        return None, "no"

    for candidate in candidates:
        if answer == candidate:
            return candidate, "candidate"
    try:
        return coerce_key(mk.validate_key(answer)), "validated"
    except ValueError:
        return None, "invalid"


_CATEGORY_TOPICS = _topics_by_category()

# 模型常常無視「一定要選主題」的指示，直接回分類名（實測 6 句裡 4 句）。
# 這是 prompt 逼不動的，所以在程式層補一刀：拿到純分類時，用句子的關鍵詞
# 決定性挑一個主題。順序代表優先權，先命中的贏。
_TOPIC_HINTS: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "約定與計畫": (
        (("英文課", "圍棋課", "才藝", "接送", "上課", "安親", "補習", "下課"), "子女課程"),
        (("眼科", "看醫生", "就醫", "抽血", "回診", "看診", "牙醫", "住院"), "子女就醫"),
        (("出遊", "旅遊", "日本", "機票", "住宿", "民宿", "訂房", "行程", "出去玩"), "出遊"),
        (("買回來", "採買", "便當", "涼麵", "市場", "買菜", "超市"), "採買"),
        (("打掃", "洗碗", "洗衣", "家事", "掃地", "清潔"), "家務分工"),
        (("照顧", "阿公", "阿嬤", "老爸", "爸爸", "媽媽"), "長輩照顧"),
        (("加班", "下班", "回來", "出門", "睡覺", "起床"), "作息"),
    ),
    "飲食": (
        (("不吃", "不喜歡吃", "不敢吃", "忌口", "過敏"), "忌口"),
        (("早餐", "午餐", "晚餐", "三餐", "中午吃", "晚上吃"), "三餐習慣"),
        (("外食", "買回來吃", "外面吃"), "外食"),
        (("煮", "下廚", "炒", "燙", "味噌湯"), "下廚"),
    ),
    "家人": (
        (("大寶", "小明", "兒子", "女兒", "小孩", "孩子"), "子女"),
        (("老婆", "太太", "妻子"), "配偶"),
        (("爸爸", "媽媽", "老爸", "老媽", "公公", "婆婆", "長輩", "阿公", "阿嬤"), "長輩"),
    ),
}


def _hint_topic(category: str, sentence: str) -> str | None:
    """純分類時，用關鍵詞決定性補一個主題；認不出就回 None（保持純分類）。"""
    if not isinstance(sentence, str):
        return None
    for words, topic in _TOPIC_HINTS.get(category, ()):
        if any(word in sentence for word in words):
            return topic
    return None


def topic_prompt(category: str) -> str:
    """Prompt for the second stage: the category is already decided by the prefilter,
    so the model only has to pick a topic — no need to re-judge whether to remember.

    逃生條款：前置篩的分類會錯（實測「我太太在醫院輪班」被判成工作與身分），
    所以模型必須有權推翻，否則錯的類別會把整句話鎖進錯的格子；
    若前置篩誤判了非事實，也可回「否」拒絕。
    """
    topics = _CATEGORY_TOPICS.get(category) or ()
    topics_line = "、".join(topics) if topics else "無固定主題"
    return (
        f"這句話已被初步判定為「{category}」類，而且值得長期記住。\n"
        f"這一類只接受下列主題：{topics_line}\n"
        "只回覆一種答案，不要解釋：\n"
        "1. 主題符合上述清單 → 直接回主題名稱（例：茶）\n"
        "2. 若下方既有主題清單裡有同一件事，優先整串照抄那個主題\n"
        f"3. 若上述清單沒有合適主題 → 直接回分類名「{category}」，保留為未定主題，不要湊近似詞、不要自創主題\n"
        f"4. 若這句話其實屬於其他分類（例：被判成「{category}」，"
        "但講的是家人的事） → 可跨分類回覆「分類/主題」或「分類」（例：家人/配偶）\n"
    )


def parse_topic(raw, category: str, candidates=()) -> tuple[str | None, str]:
    """Second-stage reply (a bare topic, or the bare category) -> (key, how)."""
    if not isinstance(raw, str):
        return None, "invalid"
    answer = _normalize(raw)
    if not answer or len(answer) > MAX_CHOICE_CHARS:
        return None, "invalid"
    if answer in {"否", "不是", "no", "NO", "No"}:
        return None, "no"

    full = f"{category}/{answer}"
    for candidate in candidates:
        if answer == candidate or full == candidate:
            return candidate, "candidate"
        if candidate.partition("/")[2] == answer:
            return candidate, "candidate"
    if answer == category:
        # 模型逃避選主題：仍然接受，但標成 category_only 讓上游能統計還有多少在逃。
        return category, "category_only"
    if answer in mk.CATEGORIES:
        return answer, "category_only"
    # 逃生條款：第二段允許推翻前置篩的分類（回「別的分類/主題」）
    if "/" in answer:
        head = answer.partition("/")[0]
        if head in mk.CATEGORIES:
            full = answer
    elif answer.startswith(category + "/"):
        full = answer
    try:
        coerced = coerce_key(mk.validate_key(full))
        if "/" not in coerced:
            return coerced, "category_only"
        return coerced, "validated"
    except ValueError:
        return None, "invalid"


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        raise ValueError("vectors must be non-empty and the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class CandidateRetriever:
    """In-memory cosine search over this scope's active keyed memories (B)."""

    def __init__(self, store, embedder, owner_id: str, character_id: str,
                 max_entries: int = 200):
        if store is None or embedder is None:
            raise ValueError("store and embedder required")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.store = store
        self.embedder = embedder
        self.owner_id = owner_id
        self.character_id = character_id
        self.max_entries = max_entries
        self._cache: dict[str, list[float]] = {}

    def _rows(self):
        keyed_rows = []
        offset = 0
        examined = 0
        while examined < self.max_entries:
            page_size = min(50, self.max_entries - examined)
            try:
                listing = self.store.list_memories(
                    self.owner_id, self.character_id, limit=page_size, offset=offset
                )
            except TypeError:
                listing = self.store.list_memories(self.owner_id, self.character_id)
            if not isinstance(listing, dict):
                break
            items = listing.get("items")
            if not items:
                break
            for item in items:
                if examined >= self.max_entries:
                    break
                examined += 1
                if item.get("meta_key"):
                    keyed_rows.append(item)
            offset += len(items)
            total = listing.get("total")
            if total is not None and offset >= total:
                break
            if len(items) < page_size:
                break
        return keyed_rows

    def embed_text(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        out = self.embedder.embed(text)
        # sentence-transformers returns a flat vector for one string; the test
        # double takes a batch and returns [[...]].
        vector = list(out[0]) if out and isinstance(out[0], (list, tuple)) else list(out)
        self._cache[text] = vector
        return vector

    def top(self, sentence: str, k: int = MAX_CANDIDATES) -> list[tuple[str, float]]:
        """[(key, score)] best-first; the key list may contain duplicates by design."""
        rows = self._rows()
        if not rows:
            return []
        try:
            query = self.embed_text(sentence)
        except Exception:
            return []
        scored = []
        for item in rows:
            try:
                vector = self.embed_text(str(item["text"]))
            except Exception:
                continue
            scored.append((str(item["meta_key"]), cosine(query, vector)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def top_keys(self, sentence: str, k: int = MAX_CANDIDATES) -> list[str]:
        """Unique keys in best-first order."""
        out: list[str] = []
        for key, _score in self.top(sentence, k):
            if key not in out:
                out.append(key)
        return out


class ModelKeyChooser:
    """A: one gated question that also returns the key (P2o-4)."""

    def __init__(self, model_reply, max_tokens: int = 24):
        if not callable(model_reply):
            raise ValueError("model_reply must be callable")
        self.model_reply = model_reply
        self.max_tokens = max_tokens

    def _prompt(self, sentence: str, candidates) -> str:
        body = f"句子：{sentence}\n"
        if candidates:
            body += "既有主題清單（可能為空）：\n" + "\n".join(f"- {key}" for key in candidates) + "\n"
        else:
            body += "既有主題清單（可能為空）：無\n"
        body += "答案："
        return body

    def choose(self, sentence: str, candidates=(), category=None) -> tuple[str | None, str]:
        """When `category` is given (prefilter already decided 要記＋哪一類),
        the model only picks a topic inside that category."""
        system = topic_prompt(category) if category else KEY_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._prompt(sentence, candidates)},
        ]
        try:
            raw = self.model_reply(messages)
        except Exception:
            return None, "error"
        if category:
            key, how = parse_topic(raw, category, candidates)
            if how == "category_only":
                # 模型逃避選主題時，用關鍵詞決定性補一刀
                hinted = _hint_topic(category, sentence)
                if hinted:
                    hinted_key = f"{category}/{hinted}"
                    try:
                        mk.validate_key(hinted_key)
                    except ValueError:
                        pass
                    else:
                        return hinted_key, "hinted"
            return key, how
        return parse_choice(raw, candidates)


PREFILTER_REMEMBER = (
    "這句話是否包含值得長期記住的、關於這個人本人或家庭生活的事實或重要安排？\n"
    f"{SHARED_RETENTION_POLICY}\n"
    "注意：句子若前半是陳述、後面接一個問句或抱怨，要看陳述事實的部分，不要因為問句或抱怨語氣就整句丟掉。"
)


def _post_json(url: str, api_key: str, body: dict, timeout: float) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class DecisionPrefilter:
    """前置篩（決策模型）：只回答「要不要記」與「哪一大類」，不生成文字。

    這是便宜的那一步（零輸出 token、不吃 llama slot），把貴的 LLM key 推導
    從「每一句都要跑」降成「只有該記的才跑」。

    judge() 回 (remember, category, how)：
      remember=False       -> 直接不記，後面全部跳過
      remember=True + cat  -> 交給 ModelKeyChooser 只問該類的主題
      remember=None        -> 決策模型失敗／答案非法，或分數落在模糊區
                              (0.30 <= noul < 0.50)；交回主 LLM 從零判是／否及選 key，
                              不是直接准許入庫。
    """

    def __init__(self, url: str, api_key: str = "", timeout: float = 30.0,
                 think_mode: str = "never"):
        if not url:
            raise ValueError("url required")
        self.url = url
        self.api_key = api_key or ""
        self.timeout = timeout
        self.think_mode = think_mode
        self.categories = tuple(mk.CATEGORIES)

    def judge(self, sentence: str) -> tuple[bool | None, str | None, str]:
        body = {
            "state": sentence,
            "think_mode": self.think_mode,
            "questions": {
                "remember": {"type": "noul", "instructions": PREFILTER_REMEMBER},
                "category": {"type": "choice", "instructions": "若值得記，屬於哪一類？",
                             "criteria": {c: c for c in self.categories}},
            },
        }
        try:
            raw = _post_json(self.url, self.api_key, body, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None, None, "error"
        if not isinstance(raw, dict):
            return None, None, "invalid"
        answers = raw.get("answers")
        if not isinstance(answers, dict):
            return None, None, "invalid"
        remember_obj = answers.get("remember")
        if not isinstance(remember_obj, dict):
            return None, None, "invalid"
        probability = remember_obj.get("noul")
        if type(probability) is bool or not isinstance(probability, (int, float)):
            return None, None, "invalid"
        if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
            return None, None, "invalid"
        if probability < 0.30:
            return False, None, "ok"
        if probability < 0.5:
            # Borderline no is not a reliable rejection: keep the full model's
            # ability to say either yes or no, without treating it as accepted.
            return None, None, "uncertain"
        category_obj = answers.get("category")
        if not isinstance(category_obj, dict):
            return None, None, "invalid"
        category = category_obj.get("choice")
        if category not in self.categories:
            return None, None, "invalid"
        return True, category, "ok"


class KeyDecision:
    """Serial A-then-B decision for one sentence."""

    def __init__(self, chooser=None, retriever=None, floor: float = DEFAULT_FLOOR,
                 fallback_rule=None, prefilter=None, addressee_family_key: str | None = None):
        if not 0.0 <= float(floor) <= 1.0:
            raise ValueError("floor must be within 0..1")
        if addressee_family_key not in (None, "家人/配偶"):
            raise ValueError("unsupported addressee family key")
        self.chooser = chooser
        self.retriever = retriever
        self.floor = float(floor)
        self.fallback_rule = fallback_rule
        self.prefilter = prefilter
        self.addressee_family_key = addressee_family_key

    def decide(self, sentence: str) -> dict:
        """Return {"key": key|None, "how": model|hinted|category_only|similarity|rule|no|error, ...}."""
        trace: list[dict] = []
        category = None
        model_invalid = False

        if self.prefilter is not None:
            remember, category, prefilter_how = self.prefilter.judge(sentence)
            if remember is False:
                trace.append({"stage": "prefilter", "how": prefilter_how or "no"})
                return {
                    "key": None,
                    "how": "prefilter_no",
                    "candidates": [],
                    "needs_review": False,
                    "trace": trace,
                }
            if prefilter_how != "ok":
                trace.append({"stage": "prefilter", "how": prefilter_how})
                category = None
            else:
                trace.append({"stage": "prefilter", "how": "ok", "category": category})

        candidates: list[str] = []
        if self.retriever is not None:
            try:
                candidates = self.retriever.top_keys(sentence)
            except Exception:
                candidates = []
                trace.append({"stage": "retriever", "how": "error"})
        if self.addressee_family_key:
            # The persona declares who 「妳」 is (personas/relationship.md). Ask the
            # full gate and offer that key, so a fact about the addressee lands on
            # the relation instead of the speaker's key — never guessed from words.
            trace.append({"stage": "persona_relation", "key": self.addressee_family_key})
            category = None
            if self.addressee_family_key not in candidates:
                candidates = [*candidates, self.addressee_family_key]

        if self.chooser is not None:
            try:
                key, how = self.chooser.choose(sentence, candidates, category=category)
            except TypeError:
                try:
                    key, how = self.chooser.choose(sentence, candidates)
                except Exception:
                    key, how = None, "error"
            except Exception:
                key, how = None, "error"

            if how in {"candidate", "validated"}:
                is_cat_only = key is not None and ("/" not in key or key in mk.CATEGORIES)
                trace.append({"stage": "chooser", "how": how, "key": key})
                return {
                    "key": key,
                    "how": "model",
                    "candidates": candidates,
                    "needs_review": is_cat_only,
                    "trace": trace,
                }
            if how == "hinted":
                trace.append({"stage": "chooser", "how": "hinted", "key": key})
                return {
                    "key": key,
                    "how": "hinted",
                    "candidates": candidates,
                    "needs_review": False,
                    "trace": trace,
                }
            if how == "category_only":
                trace.append({"stage": "chooser", "how": "category_only", "key": key, "category": key})
                return {
                    "key": key,
                    "how": "category_only",
                    "candidates": candidates,
                    "needs_review": True,
                    "trace": trace,
                }
            if how == "no":
                # 第二段（選主題）不得否決第一段已核准的事實：模型只是在逃避菜單，
                # 保留純分類並標記待人工確認，避免已判定該記的內容被靜默丟掉。
                if category is not None:
                    trace.append({"stage": "chooser", "how": "no_downgraded", "key": category})
                    return {
                        "key": category,
                        "how": "category_only",
                        "candidates": candidates,
                        "needs_review": True,
                        "trace": trace,
                    }
                trace.append({"stage": "chooser", "how": "no"})
                return {
                    "key": None,
                    "how": "no",
                    "candidates": candidates,
                    "needs_review": False,
                    "trace": trace,
                }
            # error / invalid -> record fallback reason and fall through to B
            trace.append({"stage": "chooser", "how": how})
            model_invalid = how == "invalid"

        if self.retriever is not None and candidates:
            try:
                top = self.retriever.top(sentence, 1)
            except Exception:
                top = []
                trace.append({"stage": "similarity", "how": "error"})
            if top and top[0][1] >= self.floor:
                sim_key = top[0][0]
                is_cat_only = "/" not in sim_key or sim_key in mk.CATEGORIES
                trace.append({"stage": "similarity", "how": "ok", "key": sim_key})
                return {
                    "key": sim_key,
                    "how": "similarity",
                    "score": round(top[0][1], 4),
                    "candidates": candidates,
                    "needs_review": bool(is_cat_only or model_invalid),
                    "trace": trace,
                }
            elif top:
                trace.append({"stage": "similarity", "how": "below_floor"})

        if self.fallback_rule is not None:
            try:
                rule_res = self.fallback_rule(sentence)
                if isinstance(rule_res, tuple) and len(rule_res) == 2:
                    rule_key, rule_review = rule_res
                else:
                    rule_key, rule_review = rule_res, False
                is_cat_only = rule_key is not None and ("/" not in rule_key or rule_key in mk.CATEGORIES)
                trace.append({"stage": "rule", "how": "ok", "key": rule_key})
                return {
                    "key": rule_key,
                    "how": "rule",
                    "candidates": candidates,
                    "needs_review": bool(rule_review or is_cat_only or model_invalid),
                    "trace": trace,
                }
            except Exception:
                trace.append({"stage": "rule", "how": "error"})

        trace.append({"stage": "fallback", "how": "exhausted"})
        return {
            "key": None,
            "how": "error",
            "candidates": candidates,
            "needs_review": False,
            "trace": trace,
        }
