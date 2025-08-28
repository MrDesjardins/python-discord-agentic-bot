from typing import Any, Dict, Literal, Optional, TypedDict, Annotated
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from typing_extensions import TypedDict
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


@tool
def stats_database(query: str) -> str:
    """Query the Stats database. Input should be a SQL query. Use `get_stats_schema` tool if unsure about table names and fields."""
    with DatabaseManager.get_database_manager() as db:
        try:
            cursor = db.get_cursor(DBName.SIEGE)
            cursor.execute(query)
            results = cursor.fetchall()
            return str(results)
        except Exception as e:
            # Return error back to LLM to retry
            return f"SQL_ERROR: {str(e)}"


@tool
def tournament_database(query: str) -> str:
    """Query the Tournament database. Input should be a SQL query. Use `get_tournament_schema` tool if unsure about table names and fields."""
    with DatabaseManager.get_database_manager() as db:
        try:
            cursor = db.get_cursor(DBName.SIEGE)
            cursor.execute(query)
            results = cursor.fetchall()
            return str(results)
        except Exception as e:
            # Return error back to LLM to retry
            return f"SQL_ERROR: {str(e)}"


@tool
def activity_database(query: str) -> str:
    """Query the Activity database. Input should be a SQL query. Use `get_activity_schema` tool if unsure about table names and fields."""
    with DatabaseManager.get_database_manager() as db:
        try:
            cursor = db.get_cursor(DBName.SIEGE)
            cursor.execute(query)
            results = cursor.fetchall()
            return str(results)
        except Exception as e:
            # Return error back to LLM to retry
            return f"SQL_ERROR: {str(e)}"


async def execute_with_retry(agent, user_msg: str, tool_hint: Optional[str]):
    """Retry SQL generation up to N times if tool execution fails."""
    context_msgs = [
        HumanMessage(
            content=f"{user_msg}."
            + (f" Use {tool_hint} for data." if tool_hint else "")
        )
    ]

    for attempt in range(MAX_RETRIES):
        result = await agent.ainvoke({"messages": context_msgs})
        last_output = result.content if hasattr(result, "content") else str(result)

        if "SQL_ERROR" not in last_output:
            return result  # ✅ Success
        else:
            # Add the error back into the context so LLM can self-correct
            context_msgs.append(
                HumanMessage(
                    content=f"Retry: Previous query failed with error: {last_output}. Fix the SQL."
                )
            )

    return AIMessage(
        content=f"ERROR: Could not generate a valid SQL query after {MAX_RETRIES} attempts."
    )


# -------------------
# Example Post-Processing
# -------------------


def postprocess_stats(data: Any) -> Dict:
    """Clean/reshape stats DB output into a structured object."""
    # Suppose data = [(1, "Alice", 50), (2, "Bob", 70)]
    rows = [dict(id=row[0], name=row[1], score=row[2]) for row in data]
    return {"type": "stats", "players": rows}


def postprocess_tournament(data: Any) -> Dict:
    """Clean/reshape tournament DB output into structured object."""
    # Suppose data = [(10, "NYC", 5000)]
    rows = [dict(id=row[0], location=row[1], prize=row[2]) for row in data]
    return {"type": "tournament", "tournaments": rows}


def postprocess_api(data: Any) -> Dict:
    """Clean API result."""
    return {"type": "api", "raw": data}


@dataclass
class AIConversationCustomContext:
    """
    Define the runtime context
    """

    provider: Literal["openai", "google"] = "openai"

    user_discord_id: int = 0
    user_discord_display_name: str

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
        graph_builder.add_node("postprocess", self.postprocess_step)
        graph_builder.add_node("message_gen", self.message_gen_step)
        graph_builder.add_node("personalize", self.personalize_step)

        # Order of execution
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", "postprocess")
        graph_builder.add_edge("postprocess", "message_gen")
        graph_builder.add_edge("message_gen", "personalize")
        graph_builder.add_edge("personalize", END)

        self.graph = graph_builder.compile()

    async def chatbot(self, state: State):
        try:
            # Keyword-based routing
            tool_hint = None
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
                tool_hint = "stats_database"

            keywords_tournament = ["tournament", "bet"]
            if any(keyword in user_msg_lower for keyword in keywords_tournament):
                tool_hint = "tournament_database"

            keywords_schedule = ["time", "date", "schedule"]
            if any(keyword in user_msg_lower for keyword in keywords_schedule):
                tool_hint = "activity_database"

            # Pass hint to the LLM so it picks the right tool
            output = await execute_with_retry(self.agent, user_msg, tool_hint)
            if tool_hint:
                output = await self.agent.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=f"{user_msg}. Use {tool_hint} for data."
                            )
                        ]
                    },
                )
            else:
                output = await self.agent.ainvoke(
                    {"messages": [HumanMessage(content=f"{user_msg}.")]},
                )
            return output

        except GraphRecursionError:
            print_error_log("Agent stopped due to max iterations.")

    async def postprocess_step(self, state: State):
        """Each DB result gets custom formatting."""
        last_msg = state["messages"][-1]
        content = last_msg.content

        if "stats" in content.lower():
            structured = postprocess_stats(content)
        elif "tournament" in content.lower():
            structured = postprocess_tournament(content)
        else:
            structured = postprocess_api(content)

        return {
            "messages": state["messages"]
            + [AIMessage(content=f"POSTPROCESSED: {structured}")]
        }

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

    async def personalize_step(self, state: State, runtime: Runtime[CustomContext]):
        """Add personal tone based on the user name from runtime context."""
        last_msg = state["messages"][-1]
        user_name = runtime.context.user_name or "friend"

        personalized = await openai_model.ainvoke(
            [
                HumanMessage(
                    content=f"Take this message:\n'{last_msg.content}'\n"
                    f"And make it more personal by addressing the user named {user_name}."
                )
            ]
        )
        return {"messages": state["messages"] + [personalized]}

