"""
Events cog for the bot
Events are actions that the bot listens and reacts to
"""

import os
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
import discord
from deps.bot_singleton import BotSingleton
from deps.log import print_log, print_error_log
from deps.mybot import MyBot


load_dotenv()

ENV = os.getenv("ENV")


class MyEventsCog(commands.Cog):
    """
    Main events cog for the bot
    """

    lock = asyncio.Lock()

    def __init__(self, bot: MyBot):
        self.bot = bot
        self.last_task: dict[str, asyncio.Task] = {}

    @commands.Cog.listener()
    async def on_ready(self):
        """Main function to run when the bot is ready"""
        bot = self.bot
        print_log(f"{bot.user} has connected to Discord!")
        print_log(f"Bot latency: {bot.latency} seconds")
        tasks = []
        # Load ai data
        print_log("✅ AI Counted loaded")
        for guild in bot.guilds:
            print_log(
                f"Checking in guild: {guild.name} ({guild.id}) - Created the {guild.created_at}"
            )
            print_log(
                f"\tGuild {guild.name} has {guild.member_count} members, setting the commands"
            )
            guild_obj = discord.Object(id=guild.id)
            bot.tree.copy_global_to(guild=guild_obj)
            synced = await bot.tree.sync(guild=guild_obj)
            print_log(f"\tSynced {len(synced)} commands for guild {guild.name}.")

            commands_reg = await bot.tree.fetch_commands(guild=guild_obj)
            names = [command.name for command in commands_reg]
            sorted_names = sorted(names)
            for command in sorted_names:
                print_log(f"\t✅ /{command}")

        # Running all tasks concurrently and waiting for them to finish
        await asyncio.gather(*tasks)
        print_log("✅ on_ready() completed, bot is fully initialized.")

    @commands.Cog.listener()
    async def on_message(self, message) -> None:
        """
        Make the bot aware if someone mentions it in a message
        """
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        # Check if the bot was mentioned
        if self.bot.user in message.mentions:
            if BotSingleton().bot.is_running_ai_inquiry:
                await message.channel.send(
                    message.author.mention
                    + " I am already working on a inquiry. Give me more time and try later."
                )
            else:
                message_ref = await message.channel.send(
                    message.author.mention
                    + " Hi! I am here to help you. Please wait a moment (might take a minute) while I process your request... I will edit this message with the response if I can figure it out."
                )
                # Fetch the last 8 messages in the same channel
                before_replied_msg = []
                if message.reference is not None:
                    replied_message = await message.channel.fetch_message(
                        message.reference.message_id
                    )
                    # If the message is a reply, we can fetch the last x messages before the replied message
                    before_replied_msg = [
                        msg
                        async for msg in message.channel.history(
                            limit=8, before=replied_message.created_at
                        )
                    ]
                current_replied_msg = [message]
                last_messages_channel = [
                    msg async for msg in message.channel.history(limit=8)
                ]
                messages: list[discord.Message] = (
                    before_replied_msg + current_replied_msg + last_messages_channel
                )
                # Remove duplicates and sort by creation time
                messages = list({m.id: m for m in messages}.values())
                messages.sort(key=lambda m: m.created_at)
                context = "\n".join(
                    f"{m.author.display_name} said: {m.content}" for m in messages
                )
                try:
                    response = "Test"
                    if response is not None:
                        await message_ref.edit(
                            content="✅ " + message.author.mention + " " + response
                        )
                    else:
                        await message_ref.edit(
                            content="⛔ "
                            + message.author.mention
                            + " I am sorry, I could not process your request."
                        )
                except Exception as e:
                    print_error_log(f"on_message: Error processing message: {e}")
                    await message_ref.edit(
                        content=message.author.mention
                        + " I am sorry, I encountered an error while processing your request."
                    )
        # Make sure other commands still work
        await self.bot.process_commands(message)


async def setup(bot):
    """Setup function to add this cog to the bot"""
    await bot.add_cog(MyEventsCog(bot))
