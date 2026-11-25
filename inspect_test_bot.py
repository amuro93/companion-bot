"""Inspect an isolated test token without exposing it in logs or command-line arguments."""
import json
import os
import urllib.request


def main():
    token = os.environ.get("BOT_TELEGRAM_TOKEN", "")
    if not token or any(ch.isspace() for ch in token):
        raise SystemExit("Test token unavailable")
    try:
        def get(method):
            with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}", timeout=12) as response:
                data = json.load(response)
            if data.get("ok") is not True or not isinstance(data.get("result"), dict):
                raise ValueError("unexpected response")
            return data["result"]
        bot = get("getMe")
        hook = get("getWebhookInfo")
        name = bot["username"]
        ident = bot["id"]
        if not isinstance(ident, int) or not isinstance(name, str) or not bot.get("is_bot"):
            raise ValueError("not a bot")
        print(f"test_candidate=@{name} id={ident} webhook_configured={bool(hook.get('url'))}")
        if hook.get("url"):
            raise SystemExit("Webhook already configured: refusing polling")
    except SystemExit:
        raise
    except Exception:
        raise SystemExit("Telegram identity inspection failed (details hidden)") from None


if __name__ == "__main__":
    main()
