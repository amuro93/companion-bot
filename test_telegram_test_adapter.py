"""Offline PTB adapter tests: no Bot request or polling."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from companion_chat import ChatScope, CompanionChat
from telegram.ext import MessageHandler
from telegram_test_adapter import DENIED, build_test_application, make_message_handler


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.memory = Mock()
        self.memory.search.return_value = []
        self.model = Mock(return_value="hello")
        self.chat = CompanionChat({123: ChatScope("private-memory-token", "owner", "character")},
                                  self.memory, self.model, "persona")

    def update(self, user=123, chat_id=123, chat_type="private", text="hi"):
        message = SimpleNamespace(text=text, reply_text=AsyncMock())
        return SimpleNamespace(effective_user=SimpleNamespace(id=user) if user is not None else None,
                               effective_chat=SimpleNamespace(id=chat_id, type=chat_type), message=message)

    def invoke(self, update):
        asyncio.run(make_message_handler(self.chat)(update, SimpleNamespace()))
        return update.message.reply_text

    def test_bootstrap_requires_token_and_does_not_poll(self):
        for token in (None, "", "   "):
            with self.subTest(token=token), self.assertRaises(ValueError):
                build_test_application(token, self.chat)
        with patch("telegram_test_adapter.ApplicationBuilder") as builder:
            app = builder.return_value.token.return_value.build.return_value
            build_test_application("runtime-token", self.chat)
            builder.return_value.token.assert_called_once_with("runtime-token")
            app.add_handler.assert_called_once()
            handler = app.add_handler.call_args.args[0]
            self.assertIsInstance(handler, MessageHandler)
            from datetime import datetime, timezone
            from telegram import Chat, Message, Update, User
            def ptb_update(text):
                sender = User(id=123, first_name="Synthetic", is_bot=False)
                room = Chat(id=123, type="private")
                from telegram import MessageEntity
                entities = (MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=6),) if text == "/start" else ()
                msg = Message(message_id=1, date=datetime.now(timezone.utc), chat=room,
                              from_user=sender, text=text, entities=entities)
                return Update(update_id=1, message=msg)
            self.assertTrue(handler.filters.check_update(ptb_update("hi")))
            self.assertFalse(handler.filters.check_update(ptb_update("/start")))
            app.run_polling.assert_not_called()
            app.initialize.assert_not_called()
            app.start.assert_not_called()

    def test_authorized_response_and_off_thread_turn(self):
        update = self.update()
        original_turn = self.chat.turn
        thread_seen = []
        def turn(value):
            import threading
            thread_seen.append(threading.current_thread().name)
            return original_turn(value)
        self.chat.turn = turn
        send = self.invoke(update)
        send.assert_awaited_once_with("hello")
        self.assertTrue(thread_seen)
        self.assertNotEqual(thread_seen[0], "MainThread")
        self.memory.search.assert_called_once_with("private-memory-token", "hi", 3, "real")

    def test_no_model_or_response_before_authorization(self):
        for update in (self.update(user=999), self.update(user=None),
                       self.update(user=123, chat_id=999),
                       self.update(chat_type="group"), self.update(user="123")):
            with self.subTest(update=update):
                self.invoke(update).assert_not_awaited()
        self.model.assert_not_called()
        self.memory.search.assert_not_called()

    def test_failure_is_generic(self):
        update = self.update()
        self.memory.search.side_effect = RuntimeError("secret credential and memory")
        self.invoke(update).assert_awaited_once_with(DENIED)
        self.model.assert_not_called()

    def test_large_reply_split(self):
        update = self.update()
        self.model.return_value = "a" * 9000
        send = self.invoke(update)
        self.assertEqual([len(call.args[0]) for call in send.await_args_list], [4096, 4096, 808])


if __name__ == "__main__":
    unittest.main()
