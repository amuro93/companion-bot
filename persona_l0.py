"""L0 回憶層按需載入（`personas/*.md`）。

每個 L0 檔在開頭用一行宣告載入方式（沒宣告＝不載入，明確 opt-in）：

    > load: always
    > load: trigger 生日 幾歲 兒子 大寶

`render()` 回兩個部分：
  1. **persona_index** —— 所有 L0 檔的標題與觸發詞，很短，每輪都給；
  2. **本輪命中的檔案全文** —— `always` 的每輪都給，`trigger` 的命中觸發詞才給。

目的是「不把整份人生塞進每一輪 context，但被問到時答得出來」。
觸發詞表寫在檔案裡、不寫在程式裡，所以新增 L0 檔不必改 code。

⚠️ 這些檔案含真實家庭內容：不進 repo、不外流、不寫進 MEMORY.md／USER.md／skill。
"""
from __future__ import annotations

import re
from pathlib import Path

# 一輪最多塞多少字的 L0 全文（超過的 trigger 檔不展開，只留索引，避免吃掉對話預算）
MAX_L0_CHARS = 14000

_HEADER = re.compile(r"^>\s*load\s*:\s*(always|trigger)\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_TITLE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SPLIT = re.compile(r"[\s,，、;；]+")


def parse_triggers(rest: str) -> tuple[str, ...]:
    """`> load: trigger 生日 幾歲` 的第二段 → 觸發詞（去重、保序）。"""
    seen: set[str] = set()
    out: list[str] = []
    for word in _SPLIT.split(rest or ""):
        word = word.strip()
        if word and word not in seen:
            seen.add(word)
            out.append(word)
    return tuple(out)


class L0Doc:
    """一份 L0 檔：載入方式 ＋ 觸發詞 ＋ 全文。"""

    __slots__ = ("name", "title", "load", "triggers", "body")

    def __init__(self, name: str, title: str, load: str, triggers: tuple[str, ...], body: str):
        self.name = name
        self.title = title
        self.load = load
        self.triggers = triggers
        self.body = body

    def matches(self, user_text: str) -> bool:
        return bool(self.triggers) and any(word in user_text for word in self.triggers)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"L0Doc({self.name!r}, load={self.load!r}, triggers={len(self.triggers)})"


class PersonaL0:
    """把 `personas/` 底下的 L0 檔組成「索引 ＋ 按需全文」。"""

    def __init__(self, base_dir, max_chars: int = MAX_L0_CHARS):
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("invalid max_chars")
        path = Path(base_dir)
        if not path.is_dir():
            raise ValueError("persona dir not found")
        self.base_dir = path
        self.max_chars = max_chars
        self.docs = self._discover()

    # ── 掃描 ──────────────────────────────────────────────────────────
    def _discover(self) -> list[L0Doc]:
        docs: list[L0Doc] = []
        for path in sorted(self.base_dir.glob("*.md")):
            if path.name.startswith("_"):
                continue  # _TEMPLATE_* 是範本，不是資料檔
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            header = _HEADER.search(text)
            if header is None:
                continue  # 沒宣告載入方式 → 不載入
            load = header.group(1).lower()
            triggers = parse_triggers(header.group(2)) if load == "trigger" else ()
            title_match = _TITLE.search(text)
            title = title_match.group(1) if title_match else path.stem
            docs.append(L0Doc(path.name, title, load, triggers, text))
        return docs

    # ── 組裝 ──────────────────────────────────────────────────────────
    def select(self, user_text: str) -> list[L0Doc]:
        text = user_text or ""
        return [doc for doc in self.docs if doc.load == "always" or doc.matches(text)]

    def index_text(self) -> str:
        if not self.docs:
            return ""
        lines = ["\n\n【我記得的事（索引）】"]
        for doc in self.docs:
            if doc.load == "always":
                lines.append(f"- {doc.title}：全文在下面")
            else:
                sample = "、".join(doc.triggers[:8])
                lines.append(f"- {doc.title}：聊到「{sample}」這類話題時才會翻開")
        lines.append("索引裡有、但這次沒展開的，代表你此刻看不到細節 —— "
                     "不要憑空編造，就你確實記得的講，或直接問他。")
        return "\n".join(lines)

    def render(self, user_text: str) -> str:
        """回傳要接在人設後面的字串（索引 ＋ 命中檔全文）。"""
        chosen = self.select(user_text)
        parts = [self.index_text()]
        used = 0
        for doc in chosen:
            if doc.load != "always" and used + len(doc.body) > self.max_chars:
                parts.append(f"\n\n【{doc.title}】（這次沒展開：會超過提示預算）")
                continue
            parts.append("\n\n" + doc.body)
            used += len(doc.body)
        return "".join(parts)
