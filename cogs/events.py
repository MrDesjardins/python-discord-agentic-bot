"""
Events cog for the bot
Events are actions that the bot listens and reacts to
"""

import os
import asyncio
from dotenv import load_dotenv
from discord.ext import commands
import discord
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from deps.agents.agent_workflow import (
    AIConversationCustomContext,
    AIConversationWorkflow,
)
from deps.bot_singleton import BotSingleton
from deps.log import print_log, print_error_log
from deps.mybot import MyBot
from deps.siege.rank import get_user_rank_siege
from deps.rules.system_instructions import system_instruction_when_bot_mentioned

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
    async def on_message(self, message: discord.Message) -> None:
        """
        Make the bot aware if someone mentions it in a message
        """
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        # Check if the bot was mentioned
        if self.bot.user in message.mentions:
            if BotSingleton().bot.is_running_ai_inquiry:
                await message.reply(
                    "I am already working on a inquiry. Give me more time and try later.",
                    mention_author=True,
                )
            else:
                message_ref = await message.reply(
                    "Hi! I am here to help you. Please wait a moment (might take a minute) while I process your request... I will edit this message with the response if I can figure it out.",
                    mention_author=True,
                )

                history = [
                    msg
                    async for msg in message.channel.history(
                        limit=50
                    )  # fetch a bit more
                ]

                # Get messages either from the user or the bot replying to them
                conversation = []
                for msg in history:
                    if msg.author == message.author:  # same user who pinged the bot
                        conversation.append(msg)
                    elif (
                        msg.author == self.bot.user
                        and msg.reference
                        and msg.reference.resolved
                        and isinstance(msg.reference.resolved, discord.Message)
                        and msg.reference.resolved.author == message.author
                    ):  # bot replying to user
                        conversation.append(msg)

                # Keep only the last 10 relevant ones, sorted chronologically
                conversation = list(reversed(conversation))[-10:]
                previous_messages = [
                    f"{m.author.display_name} (user_id: {m.author.id}) said: {m.content}"
                    for m in conversation
                ]

                try:
                    # Create runtime context with personalization info
                    ctx = AIConversationCustomContext(
                        provider="openai",
                        message_history=previous_messages,
                        user_discord_id=message.author.id,
                        user_rank=(
                            get_user_rank_siege(message.author)
                            if isinstance(message.author, discord.Member)
                            else "Copper"
                        ),
                        user_discord_display_name=message.author.display_name,
                    )

                    workflow = AIConversationWorkflow(ctx)

                    # Now when you start the workflow, it has access to ctx.user_name
                    response = await workflow.graph.ainvoke(
                        {
                            "messages": [
                                SystemMessage(
                                    content=system_instruction_when_bot_mentioned
                                ),
                                HumanMessage(content=message.content),
                            ]
                        },
                        config={"configurable": {"ctx": ctx}},
                    )

                    if response is not None:
                        last_text = response["messages"][-1].content
                        await message_ref.edit(
                            content=f"✅ {message.author.mention} {last_text}"
                        )
                    else:
                        await message_ref.edit(
                            content=f"⛔ {message.author.mention} I am sorry, I could not process your request."
                        )
                except Exception as e:
                    print_error_log(f"on_message: Error processing message: {e}")
                    await message_ref.edit(
                        content=f"{message.author.mention} I am sorry, I encountered an error while processing your request."
                    )
        # Make sure other commands still work
        await self.bot.process_commands(message)


async def setup(bot):
    """Setup function to add this cog to the bot"""
    await bot.add_cog(MyEventsCog(bot))
