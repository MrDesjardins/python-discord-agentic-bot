from typing import Any, Dict, Literal, Optional, TypedDict, Annotated
from typing_extensions import TypedDict
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from deps.database.utils_database import get_table_schema
from deps.database.system_database import DBName, DatabaseManager
from deps.log import print_error_log

MAX_RETRIES = 5
MAX_ITERATIONS = 3
RECURSION_LIMIT = 2 * MAX_ITERATIONS + 1

openai_model = init_chat_model("openai:gpt-4.1")
google_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


@tool
def get_stats_schema() -> str:
    """Get the schema for the Stats database."""
    return f"""{get_table_schema('user_full_match_info')}
\n{get_table_schema('user_full_stats_info')}"""


@tool
def get_tournament_schema() -> str:
    """Get the schema for the Tournament database."""
    return f"""{get_table_schema('tournament')}\n
{get_table_schema('tournament_guild')}\n
{get_table_schema('tournament_game')}\n
{get_table_schema('user_tournament')}\n
{get_table_schema('tournament_team_members')}"""


@tool
def get_activity_schema() -> str:
    """
    Query the Activity database. Input should be a SQL query.

    Use `get_activity_schema` tool if unsure about table names and fields.

    Notes for query logic:
    - "The `event` column can be `connect` or `disconnect` indicating when a user gets online and offline for a channel. Use the two value to know the time when a user is online for a channel."
    """
    return f"""{get_table_schema('user_activity')}"""


def database_tool(query: str, db_name: DBName) -> Any:
    with DatabaseManager.get_database_manager() as db:
        try:
            cursor = db.get_cursor(db_name)
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"SQL_ERROR: {str(e)}"


@tool
def stats_database(query: str) -> Any:
    """Query the Stats database. Input should be a SQL query. Use `get_stats_schema` tool if unsure about table names and fields."""
    return database_tool(query, DBName.SIEGE)


@tool
def tournament_database(query: str) -> Any:
    """Query the Tournament database. Input should be a SQL query. Use `get_tournament_schema` tool if unsure about table names and fields."""
    return database_tool(query, DBName.SIEGE)


@tool
def activity_database(query: str) -> Any:
    """Query the Activity database. Input should be a SQL query. Use `get_activity_schema` tool if unsure about table names and fields."""
    return database_tool(query, DBName.SIEGE)


def get_bot_role() -> SystemMessage:
    """Define the bot's role in the conversation."""
    return SystemMessage(
        content=(
            "You are a bot that is mentioned in a Discord server. You need to answer to the user who mentioned you."
            "You should not mention anything about your name or your purpose, just answer the question."
            "Keep responses concise."
        )
    )


async def execute_plain(agent, prompt_msgs: list[BaseMessage]):
    """Run agent without SQL-specific retry logic."""
    return await agent.ainvoke({"messages": prompt_msgs + [get_bot_role()]})


async def execute_sql_with_retry(agent, prompt_msgs: list[BaseMessage], user_id: int):
    """Retry SQL generation up to N times if tool execution fails."""
    context_msgs = (
        prompt_msgs
        + [
            SystemMessage(
                content=(
                    "You must generate only SELECT SQL queries that fetch relevant data. "
                    "Never update, insert, delete, truncate, alter, or drop. "
                    "Queries must be valid for SQLite 3.45. "
                    "Prefer aggregation (COUNT, SUM, AVG, MAX, MIN) to reduce result size. "
                    "If selecting raw rows, LIMIT to 100. "
                    "Do not mention SQL or schema in your response."
                    f"The user who asked the question user_id: `{str(user_id)}` which is relevant to the query. "
                )
            )
        ]
        + [get_bot_role()]
    )

    for _attempt in range(MAX_RETRIES):
        result = await agent.ainvoke({"messages": context_msgs})
        last_output = getattr(result, "content", str(result))

        if "SQL_ERROR" not in last_output:
            return result
        context_msgs.append(
            HumanMessage(
                content=f"Retry: The query failed with `{last_output}`. "
                "Here is the schema again to help fix it:\n"
                f"{get_table_schema('user_full_stats_info')}"
            )
        )

    return AIMessage(
        content=f"ERROR: Could not generate valid SQL after {MAX_RETRIES} attempts."
    )


@dataclass
class AIConversationCustomContext:
    """
    Define the runtime context
    """

    provider: Literal["openai", "google"] = "openai"
    message_history: list[str] = []
    user_discord_id: int = 0
    user_discord_display_name: str = ""
    user_rank: str = ""


def select_model(
    state: AgentState, runtime: Runtime[AIConversationCustomContext]
) -> BaseChatModel:
    """
    Return the right model depending of the provider
    """
    if runtime.context.provider == "google":
        model = google_model
    elif runtime.context.provider == "openai":
        model = openai_model
    else:
        raise ValueError(f"Unsupported provider: {runtime.context.provider}")

    # With dynamic model selection, you must bind tools explicitly
    return model


class State(TypedDict):
    """
    Messages have the type "list". The `add_messages` function
    in the annotation defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """

    messages: Annotated[list, add_messages]


class AIConversationWorkflow:

    def __init__(self):
        self.agent = create_react_agent(
            select_model,
            tools=[
                get_stats_schema,
                get_tournament_schema,
                get_activity_schema,
                stats_database,
                tournament_database,
                activity_database,
            ],
        )

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("message_gen", self.message_gen_step)
        graph_builder.add_node("personalize", self.personalize_step)

        # Order of execution
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", "message_gen")
        graph_builder.add_edge("message_gen", "personalize")
        graph_builder.add_edge("personalize", END)

        self.graph = graph_builder.compile()

    async def chatbot(
        self, state: State, runtime: Runtime[AIConversationCustomContext]
    ):
        try:
            # Take last N messages from context history (avoid exceeding token limits)
            history_to_include = runtime.context.message_history
            history_text = "\n".join(history_to_include)

            # Keyword-based routing
            tool_hint = []
            user_msg = (
                state["messages"][-1]
                if isinstance(state["messages"], list)
                else state["messages"]
            )

            user_msg_lower = user_msg.lower()

            keywords_full_match_info = [
                "stats",
                "match",
                "data",
                " kd ",
                "k/d",
                "kill",
                "death",
                "operator",
                "map",
            ]

            if any(keyword in user_msg_lower for keyword in keywords_full_match_info):
                tool_hint.append("stats_database")

            keywords_tournament = ["tournament", "bet"]
            if any(keyword in user_msg_lower for keyword in keywords_tournament):
                tool_hint.append("tournament_database")

            keywords_schedule = ["time", "date", "schedule"]
            if any(keyword in user_msg_lower for keyword in keywords_schedule):
                tool_hint.append("activity_database")

            # Construct prompt
            tool_hint_str = f"Use {','.join(tool_hint)} for data." if tool_hint else ""
            prompt_msgs: list[BaseMessage] = [
                HumanMessage(
                    content=f"Channel history:\n{history_text}\n\nCurrent message:\n{user_msg}\n{tool_hint_str}"
                )
            ]

            # Personalize
            context = ""
            if runtime.context.user_rank == "Champion":
                context += "In the message, call the user 'champion'. "
                context += "The user like sarcasm, so answer in a sarcastic tone. "
            else:
                context += "You are a bot that is friendly, helpful and professional. You should not be rude or sarcastic. "

            prompt_msgs.append(SystemMessage(content=context))

            # Pass hint to the LLM so it picks the right tool
            if tool_hint:
                output = await execute_sql_with_retry(
                    self.agent, prompt_msgs, runtime.context.user_discord_id
                )
            else:
                output = await execute_plain(self.agent, prompt_msgs)
            return output

        except GraphRecursionError:
            print_error_log("Agent stopped due to max iterations.")

    async def message_gen_step(self, state: State):
        """Generate human-friendly final message."""
        structured_msg = state["messages"][-1].content
        final_output = await openai_model.ainvoke(
            [
                HumanMessage(
                    content=f"Turn this into a user-friendly message: {structured_msg}"
                )
            ]
        )
        return {"messages": state["messages"] + [final_output]}

    async def personalize_step(
        self, state: State, runtime: Runtime[AIConversationCustomContext]
    ):
        """Add personal tone based on the user name from runtime context."""
        last_msg = state["messages"][-1]
        user_name = runtime.context.user_discord_display_name or "friend"

        context = ""
        if runtime.context.user_rank == "Champion":
            context += "In the message, call the user 'champion'. "
            context += "The user like sarcasm, so answer in a sarcastic tone. "
        else:
            context += "You are a bot that is friendly, helpful and professional. You should not be rude or sarcastic. "

        personalized = await openai_model.ainvoke(
            [
                SystemMessage(
                    content=(
                        f"Take this message:\n'{last_msg.content}'\n"
                        f"And make it more personal by addressing the user named {user_name}."
                        f"{context}"
                    )
                )
            ]
        )
        return {"messages": state["messages"] + [personalized]}
