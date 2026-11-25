"""Local-model gatekeeper: is this sentence a long-term fact about the user? (P2j)

The gate answers yes/no only and never writes memory text. The stored memory stays a
verbatim substring of the user's own message, so model-generated prose has no path into
the store. Any failure (transport, malformed output, timeout) answers NO: missing a fact
is recoverable, inventing one is not.
"""
from __future__ import annotations

import re

MIN_SENTENCE = 4
MAX_MEMORIES_PER_MESSAGE = 2
FILLER_PREFIXES = ("嗯", "對阿", "對啊", "是啊", "是喔", "喔喔", "哈哈", "呵呵", "嘿嘿",
                   "好啊", "好喔", "好吧", "唉", "阿阿")
YES_TOKENS = ("是", "yes", "y", "true", "1")
NO_TOKENS = ("否", "no", "n", "false", "0")

SHARED_RETENTION_POLICY = (
    "要記：\n"
    "1. 本人偏好、習慣、住處、工作與身分、健康狀況與用藥、過去經歷、長期計畫；\n"
    "2. 家人與家庭狀況：配偶的工作與作息、孩子的年紀學校與課程、長輩狀況與照料；\n"
    "3. 家裡的持續性責任與固定安排（如家務分工、固定接送、家庭財務安排）；\n"
    "4. 進行中的家庭旅遊準備與聚會計畫/進度（去哪、訂了什麼、誰去、何時）；\n"
    "5. 後續仍會用到的具體日期計畫與安排（即使帶有抱怨、要求或問句，只要含有具體安排事實就要記）。\n"
    "判斷原則：\n"
    "- 具體日期的計畫只記錄該次計畫本身，切勿擅自推論為每週或固定重複的週期性常態；\n"
    "- 清楚區分發言者本人的工作/行程與家人/伴侶的工作/行程，不可混淆主體或自創模糊代名詞。\n"
    "不要記：\n"
    "- 純問句、寒暄、道別、當下短暫情緒、天氣；\n"
    "- 對伴侶的讚美或撒嬌、玩笑或吹牛；\n"
    "- 純假設或未來的假如（要是/假如哪天）；\n"
    "- 與長期生活無關的過眼即忘當下瑣事、臨時動作（如今天誰來打掃、垃圾車幾點來、當下吃了什麼）；\n"
    "- 轉述無關外人的事（同事說、朋友說）。"
)
RETENTION_POLICY = SHARED_RETENTION_POLICY

GATE_SYSTEM = (
    "你是記憶守門員。你會看到一個人對伴侶說的一句話，判斷這句話有沒有包含"
    "「關於他本人、或他家庭生活的長期事實」。\n"
    f"{SHARED_RETENTION_POLICY}\n"
    "只回答一個字：是 或 否。不要解釋。"
)

_SPLIT = re.compile(r"[。！!；;\n]+")
_QUESTION = re.compile(r"[？?]")
_CLAUSE = re.compile(r"[，,]")
_TAIL_QUESTION = re.compile(r"(嗎|呢|對吧|是吧|好不好|要不要|了沒|了嗎)$")
# 子句級的疑問偵測。原本只認句尾的「嗎／呢」，於是「妳要接送還是我接送」這種
# 二選一問句（結尾是「接送」）整句被留下，決策模型看到殘缺問句就判「不記」，
# 使得「小明十點半英文課…」這類句子的事實那半從來進不了記憶庫。
_QUESTION_CLAUSE = re.compile(
    r"(還是|是不是|要不要|有沒有|對吧|是吧|嗎|呢|"
    r"誰|什麼|哪個|哪裡|哪邊|幾點|幾個|多少|多久|怎麼|為什麼)"
)
_ADMIN = re.compile(r"^(記了什麼|記憶清單|記得什麼|/memories)")
QUESTION_TAILS = ("嗎", "呢", "對吧", "是吧", "好不好", "要不要", "了嗎", "了沒")


def _drop_trailing_questions(text: str) -> str:
    """「我血壓有點高，醫生叫我少鹽，這道菜不要太鹹好不好」 keeps the fact, drops the ask.

    最後一個子句只要是問句形式（含疑問詞）就丟掉，但永遠至少留一個子句，
    所以「還是我去買回來吃就好」這種結論型子句不見得每次都能保住 —— 事實優先。
    """
    clauses = [clause for clause in _CLAUSE.split(text)]
    while len(clauses) > 1 and (
        _TAIL_QUESTION.search(clauses[-1].strip())
        or _QUESTION_CLAUSE.search(clauses[-1].strip())
    ):
        clauses.pop()
    return "，".join(clauses).strip()


def split_sentences(text) -> list[str]:
    """Split on hard stops; keep the statement part and discard question clauses."""
    if not isinstance(text, str):
        return []
    pieces = []
    for chunk in _SPLIT.split(text):
        statement = _drop_trailing_questions(_QUESTION.split(chunk)[0].strip(" ，,、\t"))
        if statement:
            pieces.append(statement)
    return pieces


def prefilter(sentence) -> bool:
    """Cheap rules that avoid spending a model call on obvious non-facts."""
    if not isinstance(sentence, str):
        return False
    text = sentence.strip()
    if len(text) < MIN_SENTENCE:
        return False
    if _QUESTION.search(text) or _ADMIN.match(text):
        return False
    if text.endswith(QUESTION_TAILS):
        return False
    # A leading filler only disqualifies short utterances; long ones can still carry facts.
    if len(text) <= 12 and text.startswith(FILLER_PREFIXES):
        return False
    return True


def parse_verdict(reply) -> bool:
    """Map a model answer to a boolean; anything ambiguous is NO."""
    if not isinstance(reply, str):
        return False
    head = reply.strip().lower()[:8]
    if not head:
        return False
    # "否" contains neither yes token; check NO first so "不是" resolves to NO.
    if head.startswith(NO_TOKENS) or "不是" in head or "no" in head:
        return False
    return head.startswith(YES_TOKENS) or "是的" in head or "要記" in head


class LocalGate:
    """Ask the local model one yes/no question per sentence."""

    def __init__(self, model_reply, max_answer_chars: int = 16):
        if not callable(model_reply):
            raise ValueError("model_reply callable required")
        if type(max_answer_chars) is not int or not 1 <= max_answer_chars <= 64:
            raise ValueError("invalid max_answer_chars")
        self.model_reply = model_reply
        self.max_answer_chars = max_answer_chars
        self.last_error = None

    def remember(self, sentence: str) -> bool:
        if not prefilter(sentence):
            return False
        self.last_error = None
        try:
            answer = self.model_reply([
                {"role": "system", "content": GATE_SYSTEM},
                {"role": "user", "content": f"句子：「{sentence}」"},
            ])
        except Exception as exc:  # fail closed: never remember on a broken gate
            self.last_error = type(exc).__name__
            return False
        if not isinstance(answer, str) or len(answer) > 400:
            self.last_error = "bad_answer"
            return False
        return parse_verdict(answer[:self.max_answer_chars])
