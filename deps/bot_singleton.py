"""A singleton class for the bot instance"""

from __future__ import annotations  # Enables forward reference resolution
from typing import Union
from deps.mybot import MyBot


class BotSingleton:
    """A singleton class for the bot instance"""

    _instance: Union[BotSingleton, None] = None

    _bot: MyBot

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._bot = MyBot()
        return cls._instance

    @property
    def bot(self) -> MyBot:
        """Get the bot instance"""
        return self._bot
