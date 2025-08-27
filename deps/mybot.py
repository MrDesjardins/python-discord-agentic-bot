"""Custom bot class for Discord bot"""

import os
import logging
import discord
from discord.ext import commands
from deps.log import print_log, print_error_log


class MyBot(commands.Bot):
    """Add attribute to the Discord bot"""

    is_running_ai_inquiry: bool = False

    def __init__(self, *args, **kwargs):
        """
        Enable additional intents
        """
        intents = discord.Intents.default()
        intents.messages = True  # Enable the messages intent
        intents.members = True  # Enable the messages intent
        intents.reactions = True  # Enable the reactions intent
        intents.message_content = True  # Enable the message content intent
        intents.guild_reactions = True  # Enable the guild reactions intent
        intents.voice_states = (
            True  # Enable voice states to track who is in voice channel
        )
        intents.presences = True  # Needed to see member activities
        super().__init__(command_prefix="!", intents=intents)
        self.allowed_mentions = discord.AllowedMentions(
            everyone=True, roles=True, users=True
        )

    async def setup_hook(self):
        await self.load_cogs()

    async def load_cogs(self):
        """
        Load all cogs from the cogs directory
        """
        for filename in os.listdir("./cogs"):
            if filename.endswith(".py") and filename != "__init__.py":
                try:
                    await self.load_extension(f"cogs.{filename[:-3]}")
                    print_log(f"✅ Loaded {filename}")
                except Exception as e:
                    print_error_log(f"❌ Failed to load {filename}: {e}")


class ClockDriftFilter(logging.Filter):
    """
    Detects clock issues
    """

    def filter(self, record):
        # Filter out the clock drift warning
        return "Clock drift detected" not in record.getMessage()


# Apply the filter to the discord.ext.tasks logger
logger = logging.getLogger("discord.ext.tasks")
logger.addFilter(ClockDriftFilter())
