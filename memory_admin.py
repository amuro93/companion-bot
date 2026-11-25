"""In-chat listing and one-tap deletion of memories (P2i).

Pure helpers so the UX can be tested without Telegram: the adapter only converts the
(label, callback_data) rows into an InlineKeyboardMarkup.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

import memory_manual as mm

PAGE_SIZE = 8
TEXT_PREVIEW = 38
CALLBACK_PREFIX = "mem"
TAIPEI = timezone(timedelta(hours=8))
LIST_TRIGGERS = ("/memories", "記了什麼", "記憶清單", "記得什麼")
EMPTY_PAGE = ("她現在什麼都還沒記住。多跟她聊幾句，你講的話我才會記下來。\n"
              "也可以按下面的 ➕ 直接告訴她該記什麼。")


def is_list_command(text) -> bool:
    """True when the message is a listing request that must not reach the model."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return stripped in LIST_TRIGGERS


def callback_list(page: int) -> str:
    if type(page) is not int or page < 0:
        raise ValueError("invalid page")
    return f"{CALLBACK_PREFIX}:list:{page}"


def callback_delete(memory_id: str) -> str:
    if not isinstance(memory_id, str) or not memory_id.strip() or len(memory_id) > 48:
        raise ValueError("invalid memory id")
    return f"{CALLBACK_PREFIX}:del:{memory_id}"


def parse_callback(data) -> tuple[str, object] | None:
    """Decode a memory-panel callback into (kind, argument).

    Kinds: ("list", page) ("delete", memory_id) ("add", None) ("skip", None)
    ("category", index) ("topic", (category_index, topic_index)).
    """
    if not isinstance(data, str) or len(data) > 64:
        return None
    parts = data.split(":")
    if not parts or parts[0] != CALLBACK_PREFIX:
        return None
    if len(parts) == 2:
        if parts[1] == "add":
            return ("add", None)
        if parts[1] == "skip":
            return ("skip", None)
        if parts[1] == "cancel":
            return ("cancel", None)
        return None
    if len(parts) == 3:
        kind, arg = parts[1], parts[2]
        if kind == "list":
            if not arg.isdigit():
                return None
            return ("list", int(arg))
        if kind == "del":
            if not arg.strip() or len(arg) > 48:
                return None
            return ("delete", arg)
        if kind == "cat":
            if not arg.isdigit():
                return None
            return ("category", int(arg))
        return None
    if len(parts) == 4 and parts[1] == "top":
        if not parts[2].isdigit() or not parts[3].isdigit():
            return None
        return ("topic", (int(parts[2]), int(parts[3])))
    return None


def local_label(created_at) -> str:
    """UTC ISO timestamp -> Taiwan MM/DD HH:MM label; unparseable values degrade safely."""
    if not isinstance(created_at, str):
        return "--/--"
    try:
        stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return "--/--"
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(TAIPEI).strftime("%m/%d %H:%M")


def _preview(text) -> str:
    body = " ".join(str(text or "").split())
    if len(body) <= TEXT_PREVIEW:
        return body
    return body[:TEXT_PREVIEW] + "…"


def render_page(listing) -> str:
    """listing is the dict returned by MemoryStore.list_memories."""
    items = listing["items"]
    total = int(listing["total"])
    page = listing["offset"] // PAGE_SIZE
    pages = max(1, -(-total // PAGE_SIZE))
    if total == 0:
        return EMPTY_PAGE
    lines = [f"她目前記得的（共 {total} 筆）", f"第 {page + 1}/{pages} 頁", ""]
    for index, item in enumerate(items):
        number = listing["offset"] + index + 1
        lines.append(f"{mm.preview_line(item, number)}　({local_label(item.get('created_at'))})")
    lines.append("")
    lines.append("按 🗑 刪掉那一筆；🔒 是你手動加的，她不會自己蓋掉。")
    return "\n".join(lines)


def build_keyboard(listing) -> list[list[tuple[str, str]]]:
    """Rows of (button label, callback data). Empty rows list means no keyboard."""
    items = listing["items"]
    if not items:
        # P2o-2: an empty list still needs the ➕ button, or nobody can add the first memory.
        return [[(mm.ADD_BUTTON_LABEL, mm.CALLBACK_ADD)]]
    delete_row = [(f"🗑 {listing['offset'] + index + 1}", callback_delete(item["memory_id"]))
                  for index, item in enumerate(items)]
    rows = [delete_row]
    total = int(listing["total"])
    pages = max(1, -(-total // PAGE_SIZE))
    page = listing["offset"] // PAGE_SIZE
    nav = []
    if page > 0:
        nav.append(("◀ 上一頁", callback_list(page - 1)))
    if page + 1 < pages:
        nav.append(("下一頁 ▶", callback_list(page + 1)))
    if nav:
        rows.append(nav)
    rows.append([(mm.ADD_BUTTON_LABEL, mm.CALLBACK_ADD)])
    return rows


class MemoryAdmin:
    """Listing and deletion for one (owner, character) pair, with index sync on delete."""

    def __init__(self, store, indexer, owner_id: str, character_id: str):
        if store is None or indexer is None:
            raise ValueError("store and indexer required")
        if not owner_id or not character_id:
            raise ValueError("owner and character required")
        self.store = store
        self.indexer = indexer
        self.owner_id = owner_id
        self.character_id = character_id
        self.last_index_error = None
        self._pending: dict | None = None

    def listing(self, page: int = 0) -> dict:
        if type(page) is not int or page < 0:
            raise ValueError("invalid page")
        return self.store.list_memories(self.owner_id, self.character_id,
                                        limit=PAGE_SIZE, offset=page * PAGE_SIZE)

    def page(self, page: int = 0) -> tuple[str, list[list[tuple[str, str]]]]:
        listing = self.listing(page)
        if not listing["items"] and page > 0:
            listing = self.listing(0)
        return render_page(listing), build_keyboard(listing)

    # ── P2o-2: manual entry (button flow) ────────────────────────────────

    def _pending_alive(self) -> dict | None:
        """Pending step expires after a minute so a stale prompt cannot hijack a chat."""
        if self._pending is None:
            return None
        if time.monotonic() - self._pending["at"] > mm.PENDING_TIMEOUT_SECONDS:
            self._pending = None
            return None
        return self._pending

    def has_pending(self) -> bool:
        return self._pending_alive() is not None

    def cancel_pending(self) -> None:
        self._pending = None

    def start_add(self) -> tuple[str, list[list[tuple[str, str]]]]:
        self._pending = {"stage": "category", "at": time.monotonic()}
        return mm.PROMPT_CATEGORY, mm.category_rows()

    def choose_category(self, index: int) -> tuple[str, list[list[tuple[str, str]]]]:
        category = mm.category_at(index)
        rows = mm.topic_rows(category)
        if rows is not None:
            self._pending = {
                "stage": "topic",
                "category": category,
                "at": time.monotonic(),
            }
            return f"{category}：選一個主題", rows
        self._pending = {
            "stage": "topic_text",
            "category": category,
            "at": time.monotonic(),
        }
        return mm.PROMPT_TOPIC, [[(mm.SKIP_LABEL, mm.CALLBACK_SKIP)], [(mm.CANCEL_LABEL, mm.CALLBACK_CANCEL)]]

    def choose_topic_index(self, category_index: int, topic_index: int) -> tuple[str, list]:
        category = mm.category_at(category_index)
        topic = mm.topic_at(category, topic_index)
        self._pending = {
            "stage": "content",
            "category": category,
            "topic": topic,
            "at": time.monotonic(),
        }
        return f"{category}/{topic}：{mm.PROMPT_CONTENT}", [[(mm.CANCEL_LABEL, mm.CALLBACK_CANCEL)]]

    def skip_topic(self) -> tuple[str, list]:
        pending = self._pending_alive()
        if pending is None or pending["stage"] != "topic_text":
            return "已經取消囉，再按一次 ➕ 就好。", []
        self._pending = {
            "stage": "content",
            "category": pending["category"],
            "topic": None,
            "at": time.monotonic(),
        }
        return mm.PROMPT_CONTENT, [[(mm.CANCEL_LABEL, mm.CALLBACK_CANCEL)]]

    def submit_text(self, text) -> tuple[str, list]:
        """Advance the pending step with the user's typed text."""
        pending = self._pending_alive()
        if pending is None:
            return "沒有進行中的新增流程（可能超過一分鐘了）。按 ➕ 重新開始。", []
        if pending["stage"] == "topic_text":
            try:
                topic = mm.clean_topic(text)
            except ValueError:
                return f"主題要 6 字以內、不能空白。再打一次，或按「{mm.SKIP_LABEL}」。", [
                    [(mm.SKIP_LABEL, mm.CALLBACK_SKIP)]
                ]
            self._pending = {
                "stage": "content",
                "category": pending["category"],
                "topic": topic,
                "at": time.monotonic(),
            }
            return f"{pending['category']}/{topic}：{mm.PROMPT_CONTENT}", [
                [(mm.CANCEL_LABEL, mm.CALLBACK_CANCEL)]
            ]
        if pending["stage"] != "content":
            return "現在不是在輸入內容。按 ➕ 重新開始。", []
        try:
            content = mm.clean_content(text)
        except ValueError:
            return f"內容要 40 字以內、不能空白。再打一次就好。", []
        return self._finish(pending["category"], pending["topic"], content)

    def _finish(self, category: str, topic: str | None, content: str):
        self._pending = None
        try:
            out = self.create_manual(category, topic, content)
        except Exception:
            return "這次沒存進去（內部錯誤），沒有留下半筆。再試一次。", []
        key = out["meta_key"]
        if out["action"] == "unchanged":
            head = f"這一條本來就記著了：{key} → {content}"
        elif out["action"] == "correct":
            head = f"✅ 更新完成：{key} → {content}"
        elif out["action"] == "skipped":
            head = f"這條是你先前手動記的，我沒覆蓋：{key}"
        else:
            head = f"✅ 記住了：{key} → {content}"
        body, rows = self.page(0)
        return f"{head}\n\n{body}", rows

    def create_manual(self, category: str, topic: str | None, content: str) -> dict:
        """Persist a user-typed memory with source='manual' (and index it)."""
        body = mm.clean_content(content)
        key = mm.build_key(category, topic)
        kind = mm.kind_for(category)
        ext_id = mm.content_fingerprint(self.owner_id, self.character_id, key, body)

        # Typing it by hand is an explicit override of any earlier deletion block.
        try:
            self.store.unblock_key(self.owner_id, self.character_id, key)
        except Exception:
            pass

        existing = self.store.find_active_by_key(self.owner_id, self.character_id, key)
        if existing is not None and " ".join(existing["text"].split()) == body:
            return {
                "action": "unchanged",
                "memory_id": existing["memory_id"],
                "meta_key": key,
            }

        # The user's own words are the source; the evidence chain stays intact.
        source_msg = self.store.append_message(
            owner_id=self.owner_id,
            character_id=self.character_id,
            external_id=f"src-{ext_id}",
            role="user",
            content=body,
        )
        out = self.store.save_keyed_memory(
            self.owner_id,
            self.character_id,
            ext_id,
            kind,
            body,
            [source_msg],
            key,
            source="manual",
        )
        self.last_index_error = None
        if out["action"] in ("insert", "correct"):
            try:
                self.indexer.run_once(self.owner_id, self.character_id)
            except Exception as exc:  # index lag must not undo a saved memory
                self.last_index_error = type(exc).__name__
        return out

    def delete(self, memory_id: str) -> bool:
        """Delete one memory and push the vector deletion through the index worker.

        Deleting a memory the user typed also blocks its key, so the automatic layer
        cannot put the same thing back tomorrow (P2o-3).
        """
        before = self.store.get_memory(self.owner_id, self.character_id, memory_id)
        deleted = self.store.delete_memory(self.owner_id, self.character_id, memory_id)
        if deleted and before is not None and before.get("source") == "manual" and before.get("meta_key"):
            try:
                self.store.block_key(self.owner_id, self.character_id, before["meta_key"])
            except Exception:
                pass  # blocking is best-effort; the deletion itself already succeeded
        self.last_index_error = None
        if deleted:
            try:
                self.indexer.run_once(self.owner_id, self.character_id)
            except Exception as exc:  # index lag must not undo a successful deletion
                self.last_index_error = type(exc).__name__
        return deleted
