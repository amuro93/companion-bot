import asyncio
import unittest
from types import SimpleNamespace

from companion_chat import ChatScope
from memory_admin import MemoryAdmin, callback_delete
from memory_core import MemoryStore
from memory_write import remember_statement
from telegram_test_adapter import make_callback_handler, make_list_handler

OWNER, CHARACTER, USER_ID = "owner-adapter", "role-adapter", 424242


class FakeIndexer:
    def run_once(self, owner_id, character_id):
        return {"status": "ok"}


def make_admin(memories=()):
    store = MemoryStore(":memory:")
    for index, text in enumerate(memories):
        message_id = store.append_message(OWNER, CHARACTER, f"m{index}", "user", text)
        remember_statement(store, OWNER, CHARACTER, f"e{index}", "fact", text, [message_id],
                           visible_to=[CHARACTER])
    return MemoryAdmin(store, FakeIndexer(), OWNER, CHARACTER)


def make_chat():
    return SimpleNamespace(scopes={USER_ID: ChatScope("token-abc", OWNER, CHARACTER, "real")})


def make_update(text=None, user_id=USER_ID, chat_type="private", chat_id=None):
    message = SimpleNamespace(text=text, replies=[])

    async def reply_text(body, reply_markup=None):
        message.replies.append((body, reply_markup))

    message.reply_text = reply_text
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(type=chat_type, id=user_id if chat_id is None else chat_id),
        message=message,
    )


def make_query_update(data, user_id=USER_ID):
    query = SimpleNamespace(data=data, answers=[], edits=[])

    async def answer(text=None):
        query.answers.append(text)

    async def edit_message_text(body, reply_markup=None):
        query.edits.append((body, reply_markup))

    query.answer = answer
    query.edit_message_text = edit_message_text
    update = SimpleNamespace(effective_user=SimpleNamespace(id=user_id),
                             effective_chat=SimpleNamespace(type="private", id=user_id),
                             callback_query=query)
    return SimpleNamespace(update=update, query=query)


class ListHandlerTests(unittest.TestCase):
    def test_list_command_replies_and_stops_propagation(self):
        from telegram.ext import ApplicationHandlerStop
        admin = make_admin(["我比較愛茉莉花茶"])
        chat = make_chat()
        update = make_update(text="記了什麼")
        handler = make_list_handler(chat, admin)
        with self.assertRaises(ApplicationHandlerStop):
            asyncio.run(handler(update, None))
        self.assertEqual(len(update.message.replies), 1)
        body, markup = update.message.replies[0]
        self.assertIn("共 1 筆", body)
        self.assertIn("我比較愛茉莉花茶", body)
        self.assertIsNotNone(markup)

    def test_ordinary_text_is_not_intercepted(self):
        admin = make_admin(["我比較愛茉莉花茶"])
        update = make_update(text="我比較愛茉莉花茶")
        handler = make_list_handler(make_chat(), admin)
        asyncio.run(handler(update, None))  # returns without raising and without replying
        self.assertEqual(update.message.replies, [])

    def test_unauthorized_is_never_answered(self):
        from telegram.ext import ApplicationHandlerStop
        admin = make_admin(["我比較愛茉莉花茶"])
        update = make_update(text="記了什麼", user_id=1)
        handler = make_list_handler(make_chat(), admin)
        with self.assertRaises(ApplicationHandlerStop):
            asyncio.run(handler(update, None))
        self.assertEqual(update.message.replies, [])

    def test_group_chat_is_never_answered(self):
        from telegram.ext import ApplicationHandlerStop
        admin = make_admin(["我比較愛茉莉花茶"])
        update = make_update(text="記了什麼", chat_type="group")
        handler = make_list_handler(make_chat(), admin)
        with self.assertRaises(ApplicationHandlerStop):
            asyncio.run(handler(update, None))
        self.assertEqual(update.message.replies, [])


class CallbackHandlerTests(unittest.TestCase):
    def test_delete_button_removes_and_redraws(self):
        admin = make_admin(["我比較愛茉莉花茶", "我們住這個小鎮"])
        before = admin.listing(0)
        target = before["items"][0]
        built = make_query_update(callback_delete(target["memory_id"]))
        handler = make_callback_handler(make_chat(), admin)
        asyncio.run(handler(built.update, None))
        self.assertEqual(built.query.answers, ["已刪除"])
        self.assertEqual(len(built.query.edits), 1)
        body = built.query.edits[0][0]
        self.assertIn("共 1 筆", body)
        self.assertNotIn(target["text"], body)

    def test_bad_callback_is_acknowledged_only(self):
        admin = make_admin(["我比較愛茉莉花茶"])
        built = make_query_update("mem:del:")
        handler = make_callback_handler(make_chat(), admin)
        asyncio.run(handler(built.update, None))
        self.assertEqual(built.query.answers, [None])
        self.assertEqual(built.query.edits, [])

    def test_unauthorized_callback_is_refused(self):
        admin = make_admin(["我比較愛茉莉花茶"])
        built = make_query_update("mem:list:0", user_id=99)
        handler = make_callback_handler(make_chat(), admin)
        asyncio.run(handler(built.update, None))
        self.assertEqual(built.query.answers, ["沒有權限。"])
        self.assertEqual(built.query.edits, [])


if __name__ == "__main__":
    unittest.main()
