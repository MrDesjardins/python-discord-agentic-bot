"""
Agent Workflow for the Discord bot
"""

from typing import Any, Literal, TypedDict, Annotated
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain.output_parsers import PydanticOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.runtime import Runtime
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from deps.database.utils_database import get_table_schema
from deps.database.system_database import DBName, DatabaseManager
from deps.log import print_error_log
from deps.models.agent_llm_model import SQLQuery

MAX_RETRIES = 5
MAX_ITERATIONS = 3
RECURSION_LIMIT = 2 * MAX_ITERATIONS + 1

openai_model = init_chat_model("openai:gpt-4.1")
google_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


@dataclass
class AIConversationCustomContext:
    """
    Define the runtime context
    """

    provider: Literal["openai", "google"] = "openai"
    message_history: list = field(default_factory=list)  # Unique list per context
    user_discord_id: int = 0
    user_discord_display_name: str = ""
    user_rank: str = ""


def get_user_schema() -> str:
    """Get the schema for the user database."""
    return f"""{get_table_schema('user_info')}"""


def get_stats_schema() -> str:
    """Get the schema for the Stats database."""
    return f"""{get_table_schema('user_full_match_info')}
\n{get_table_schema('user_full_stats_info')}"""


def get_tournament_schema() -> str:
    """Get the schema for the Tournament database."""
    return f"""{get_table_schema('tournament')}\n
{get_table_schema('tournament_guild')}\n
{get_table_schema('tournament_game')}\n
{get_table_schema('user_tournament')}\n
{get_table_schema('tournament_team_members')}"""


def get_activity_schema() -> str:
    """
    Get the schema for the Activity database.
    """
    return (
        f"""{get_table_schema('user_activity')}"""
        f"The field in the table user_activity can be `connect` or `disconnect` which can be used to know when someone was online and disconnect between a period of time. "
    )


def execute_database(query: str) -> Any:
    """
    Utility function to connect to a database
    """
    with DatabaseManager.get_database_manager() as db:
        try:
            cursor = db.get_cursor(DBName.SIEGE)
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Exception as e:
            return f"SQL_ERROR: {str(e)}"


@tool
def database_tool(query: str) -> Any:
    """Query the users database. Input should be a SQL query."""
    return execute_database(query)


def get_bot_role() -> SystemMessage:
    """Define the bot's role in the conversation."""
    return SystemMessage(
        content=(
            "You are a bot that is mentioned in a Discord server. You need to answer to the user who mentioned you. "
            "You should not mention anything about your name or your purpose, just answer the question. "
            "Keep responses concise. "
        )
    )


async def execute_plain(agent, prompt_msgs: list[BaseMessage]):
    """Run agent without SQL-specific retry logic."""
    return await agent.ainvoke({"messages": [get_bot_role()] + prompt_msgs})


sql_parser = PydanticOutputParser(pydantic_object=SQLQuery)


def sql_prompt_schema(user_id: int, schema_info: str, question: str) -> str:
    return f"""
You are a SQL assistant. If database data is needed, output **only** a JSON object 
matching this Pydantic schema:

{sql_parser.schema}

Constraints:
- Only SELECT queries
- Valid SQLite 3 syntax
- Use the provided schema: {schema_info}
- Use the exact user_id: {user_id} if needed
- Do not include raw SQL in your human-facing answer

Question: {question}
"""


async def execute_sql_with_structured(
    agent,
    ctx: AIConversationCustomContext,
    question: str,
    schema_info: str,
    user_id: int,
):
    prompt_text = sql_prompt_schema(user_id, schema_info, question)

    # Ask LLM to produce structured output
    # result = await agent.ainvoke(prompt_text)
    model = get_model(ctx.provider)
    result = await model.ainvoke([SystemMessage(content=prompt_text)])

    # Parse directly to typed object
    sql_query: SQLQuery = sql_parser.parse(str(result.content))

    # Call the tool using the validated output
    return execute_database(sql_query.query)


def select_model(
    state: AgentState,
    runtime: Runtime[AIConversationCustomContext],
    ctx: AIConversationCustomContext,
) -> BaseChatModel:
    """
    Return the right model depending of the provider
    """

    # With dynamic model selection, you must bind tools explicitly
    return get_model(ctx.provider)


def get_model(provider: str) -> BaseChatModel:
    if provider == "google":
        return google_model
    elif provider == "openai":
        return openai_model
    else:
        raise ValueError(f"Unsupported provider: {provider}")


class State(TypedDict):
    """
    Messages have the type "list". The `add_messages` function
    in the annotation defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """

    messages: Annotated[list, add_messages]


class AIConversationWorkflow:
    """
    Class that describe the workflow of a user and AI communicating back and forth (conversation)
    """

    def __init__(self, ctx: AIConversationCustomContext):
        self.agent = create_react_agent(
            lambda state, runtime: select_model(state, runtime, ctx),
            tools=[database_tool],
        )

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("message_gen", self.message_gen_step)

        # Order of execution
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", "message_gen")
        graph_builder.add_edge("message_gen", END)

        self.graph = graph_builder.compile()

    async def chatbot(self, state: State, config: RunnableConfig):
        try:
            # Take last N messages from context history (avoid exceeding token limits)
            ctx: AIConversationCustomContext = config["configurable"]["ctx"]
            history_to_include = ctx.message_history
            history_text = "\n".join(history_to_include)

            # Keyword-based routing
            user_msg: HumanMessage = (
                state["messages"][-1]
                if isinstance(state["messages"], list)
                else state["messages"]
            )

            user_msg_lower = (
                user_msg.content.lower() if isinstance(user_msg.content, str) else ""
            )

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
            schema = get_user_schema()  # Always

            if any(keyword in user_msg_lower for keyword in keywords_full_match_info):
                schema += get_stats_schema()

            keywords_tournament = ["tournament", "bet"]
            if any(keyword in user_msg_lower for keyword in keywords_tournament):
                schema += get_tournament_schema()

            keywords_schedule = ["time", "date", "schedule"]
            if any(keyword in user_msg_lower for keyword in keywords_schedule):
                schema += get_activity_schema()

            # Construct prompt
            tool_hint_str = "Use the tool database_tool to query the database using the provided schema."
            prompt_msgs: list[BaseMessage] = [
                SystemMessage(
                    content=f"Channel history:\n{history_text}\n\nCurrent message:\n{user_msg.content}\n{tool_hint_str}"
                )
            ]

            # Personalize
            context = ""
            if ctx.user_rank == "Champion":
                context += "In the message, call the user 'champion'. "
                context += "The user like sarcasm, so answer in a sarcastic tone. "
            else:
                context += "You are a bot that is friendly, helpful and professional. You should not be rude or sarcastic. "

            if schema:
                prompt_msgs.append(
                    SystemMessage(content=f"Here is the schema:\n{schema}")
                )
            prompt_msgs.append(SystemMessage(content=context))
            query_result = await execute_sql_with_structured(
                self.agent,
                ctx,
                question=str(user_msg.content),
                schema_info=schema,
                user_id=ctx.user_discord_id,
            )
            if isinstance(query_result, str) and query_result.startswith("SQL_ERROR:"):
                return {
                    "messages": state["messages"]
                    + [
                        AIMessage(
                            content="I ran into a database issue while handling your request. Please try again later."
                        )
                    ]
                }
            return query_result

        except GraphRecursionError as e:
            print_error_log(f"Agent stopped due to max iterations: {e}")

    async def message_gen_step(self, state: State, config: RunnableConfig):
        ctx: AIConversationCustomContext = config["configurable"]["ctx"]
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            structured_msg = last_msg.content
        else:
            structured_msg = str(last_msg)
        model = get_model(ctx.provider)
        final_output = await model.ainvoke(
            [
                HumanMessage(
                    content=f"Turn this into a concise, user-friendly message: {structured_msg}"
                )
            ]
        )
        return {"messages": state["messages"] + [final_output]}
