#!/usr/bin/env python3
"""Entry file for the Discord bot"""

import os
from dotenv import load_dotenv

# Load environment variables before importing modules that initialize LangChain
load_dotenv()

# Preserve LangSmith-specific envs, but also set the LangChain tracing flag
# LangChain uses `LANGCHAIN_TRACING` to enable tracing which routes to LangSmith.
ls_tracing = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_TRACING"] = ls_tracing
# If `LANGCHAIN_TRACING` is not explicitly set, default it to the LangSmith value
os.environ["LANGCHAIN_TRACING"] = os.getenv("LANGCHAIN_TRACING", ls_tracing)
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

from deps.bot_singleton import BotSingleton
from deps.mybot import MyBot
from deps.log import print_log

ENV = os.getenv("ENV")
TOKEN = os.getenv("BOT_TOKEN_DEV") if ENV == "dev" else os.getenv("BOT_TOKEN")
if TOKEN is None:
    print_log("BOT_TOKEN_DEV not found")
    exit()

TOKEN_STR: str = str(TOKEN)

bot: MyBot = BotSingleton().bot

print_log(f"Env: {ENV}")
print_log(f"Token: {TOKEN}")


def main() -> None:
    """Start the bot"""
    bot.run(TOKEN_STR)


if __name__ == "__main__":
    main()
