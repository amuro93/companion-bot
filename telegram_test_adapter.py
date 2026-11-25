"""Opt-in PTB test adapter. Building an application never starts polling."""
from __future__ import annotations

import asyncio

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (ApplicationBuilder, ApplicationHandlerStop, CallbackQueryHandler,
                          ContextTypes, MessageHandler, filters)

from companion_chat import ChatScope, CompanionChat
from memory_admin import MemoryAdmin, is_list_command, parse_callback

TELEGRAM_TEXT_LIMIT = 4096
DENIED = "Unable to process this message."


def _authorized_user(chat: CompanionChat, update: Update) -> bool:
    """This private chat belongs to a server-owned scope for this user ID."""
    user = getattr(getattr(update, "effective_user", None), "id", None)
    conversation = getattr(update, "effective_chat", None)
    scope = chat.scopes.get(user) if type(user) is int and user > 0 else None
    return (getattr(conversation, "type", None) == "private"
            and type(getattr(conversation, "id", None)) is int
            and conversation.id == user
            and isinstance(scope, ChatScope)
            and bool(scope.token and scope.owner_id and scope.character_id)
            and scope.domain in {"real", "roleplay"})


def _authorized(chat: CompanionChat, update: Update) -> bool:
    """Only server-owned user-ID scopes can authorize a private message."""
    return _authorized_user(chat, update) and getattr(update, "message", None) is not None


def _markup(rows):
    if not rows:
        return None
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(label, callback_data=data) for label, data in row] for row in rows])


def make_message_handler(chat: CompanionChat):
    """Return an async callback suitable for PTB or direct offline testing."""
    if not isinstance(chat, CompanionChat):
        raise TypeError("CompanionChat required")

    async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(chat, update):
            return  # Do not send anything to unauthorized users or groups.
        try:
            reply = await asyncio.to_thread(chat.turn, update)
            if not isinstance(reply, str) or not reply.strip():
                raise ValueError("invalid reply")
            for offset in range(0, len(reply), TELEGRAM_TEXT_LIMIT):
                await update.message.reply_text(reply[offset:offset + TELEGRAM_TEXT_LIMIT])
        except Exception:
            # Never expose exception strings, model output, memory or credentials.
            # In particular do not leak a private reply via logs.
            try:
                await update.message.reply_text(DENIED)
            except Exception:
                pass

    return on_message


def make_list_handler(chat: CompanionChat, admin: MemoryAdmin):
    """Intercept the listing command so it never reaches the model."""
    async def on_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = getattr(getattr(update, "message", None), "text", None)
        if not is_list_command(text):
            return
        if not _authorized(chat, update):
            raise ApplicationHandlerStop
        try:
            body, rows = await asyncio.to_thread(admin.page, 0)
            await update.message.reply_text(body, reply_markup=_markup(rows))
        except Exception:
            try:
                await update.message.reply_text(DENIED)
            except Exception:
                pass
        raise ApplicationHandlerStop  # a listing request is never a chat turn

    return on_list


def make_callback_handler(chat: CompanionChat, admin: MemoryAdmin, router=None):
    """Handle the 🗑 delete buttons, pagination and the /model switch."""

    async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = getattr(update, "callback_query", None)
        if query is None:
            return
        data = getattr(query, "data", None)
        if router is not None and isinstance(data, str) and data.startswith("model:"):
            if not _authorized_user(chat, update):
                await query.answer("沒有權限。")
                return
            try:
                index = int(data.split(":", 1)[1])
                labels = router.labels()
                if not 0 <= index < len(labels):
                    raise ValueError("bad model index")
                label = router.switch(labels[index])
                await query.answer("已切換")
                await query.edit_message_text("目前模型：{}".format(label))
            except Exception:
                try:
                    await query.answer("切換失敗")
                except Exception:
                    pass
            return
        if not _authorized_user(chat, update):
            await query.answer("沒有權限。")
            return
        parsed = parse_callback(getattr(query, "data", None))
        if parsed is None:
            await query.answer()
            return
        kind, argument = parsed
        try:
            if kind == "delete":
                removed = await asyncio.to_thread(admin.delete, argument)
                await query.answer("已刪除" if removed else "找不到這一筆。")
                body, rows = await asyncio.to_thread(admin.page, 0)
            elif kind == "add":
                await query.answer()
                body, rows = await asyncio.to_thread(admin.start_add)
            elif kind == "category":
                await query.answer()
                body, rows = await asyncio.to_thread(admin.choose_category, argument)
            elif kind == "topic":
                await query.answer()
                body, rows = await asyncio.to_thread(
                    admin.choose_topic_index, argument[0], argument[1])
            elif kind == "skip":
                await query.answer()
                body, rows = await asyncio.to_thread(admin.skip_topic)
            elif kind == "cancel":
                await query.answer()
                await asyncio.to_thread(admin.cancel_pending)
                body, rows = await asyncio.to_thread(admin.page, 0)
            else:
                await query.answer()
                body, rows = await asyncio.to_thread(admin.page, argument)
            await query.edit_message_text(body, reply_markup=_markup(rows))
        except Exception:
            return

    return on_callback


def make_manual_handler(chat: CompanionChat, admin: MemoryAdmin):
    """Consume the typed topic/content while a manual-entry flow is pending."""

    async def on_manual_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not _authorized(chat, update):
            return
        if not admin.has_pending():
            return  # no flow in progress: let the chat handler take it
        text = getattr(update.message, "text", None)
        if not isinstance(text, str) or not text.strip():
            return
        try:
            body, rows = await asyncio.to_thread(admin.submit_text, text)
            await update.message.reply_text(body, reply_markup=_markup(rows))
        except Exception:
            try:
                await update.message.reply_text(DENIED)
            except Exception:
                pass
        raise ApplicationHandlerStop  # a pending flow must not become a chat turn

    return on_manual_text


def is_model_command(text) -> bool:
    """Match /model (and /model@BotName) exactly."""
    if not isinstance(text, str):
        return False
    head = text.strip().split(maxsplit=1)[0] if text.strip() else ""
    return head.split("@", 1)[0].lower() == "/model"


def make_model_handler(chat: CompanionChat, router):
    """Show the current backend and one inline button per choice."""

    async def on_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        text = getattr(getattr(update, "message", None), "text", None)
        if not is_model_command(text):
            return
        if not _authorized(chat, update):
            raise ApplicationHandlerStop
        try:
            rows = [[(label, "model:{}".format(index))]
                    for index, label in enumerate(router.labels())]
            await update.message.reply_text(
                "目前模型：{}\n要切換到哪一顆？".format(router.current),
                reply_markup=_markup(rows))
        except Exception:
            try:
                await update.message.reply_text(DENIED)
            except Exception:
                pass
        raise ApplicationHandlerStop  # never becomes a chat turn

    return on_model


def build_test_application(token: str, chat: CompanionChat, admin: MemoryAdmin | None = None,
                           router=None):
    """Register a private text handler; caller owns lifecycle and token.

    Never invokes initialize/start/run_polling or any Telegram API method.
    Do not attach this to an existing bot using the same token.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError("Telegram token required")
    if not isinstance(chat, CompanionChat):
        raise TypeError("CompanionChat required")
    application = ApplicationBuilder().token(token).build()
    if admin is not None or router is not None:
        # One handler per group: PTB stops a group after its first matching handler,
        # so these must live in separate groups to all get a chance.
        if admin is not None:
            application.add_handler(MessageHandler(filters.TEXT, make_list_handler(chat, admin)),
                                    group=-3)
            application.add_handler(MessageHandler(filters.TEXT, make_manual_handler(chat, admin)),
                                    group=-2)
        if router is not None:
            application.add_handler(MessageHandler(filters.TEXT, make_model_handler(chat, router)),
                                    group=-1)
        application.add_handler(CallbackQueryHandler(make_callback_handler(chat, admin, router)),
                                group=-1)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                           make_message_handler(chat)))
    return application
