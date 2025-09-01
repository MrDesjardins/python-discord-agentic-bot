"""
Agent Workflow for the Discord bot
"""

from typing import Any, Literal, Optional, TypedDict, Annotated, Union
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
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
from deps.rules.system_instructions import system_instruction_when_bot_mentioned

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
    user_question: str = ""
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


def execute_database(query: str) -> Union[str, list[Any]]:
    """
    Utility function to connect to a database
    """
    with DatabaseManager.get_database_manager() as db:
        cursor = db.get_cursor(DBName.SIEGE)
        cursor.execute(query)
        results = cursor.fetchall()
        return results


@tool
def database_tool(query: str) -> Union[str, list[Any]]:
    """Query the users database. Input should be a SQL query."""
    return execute_database(query)


def get_bot_role() -> SystemMessage:
    """Define the bot's role in the conversation."""
    return SystemMessage(content=(system_instruction_when_bot_mentioned))


async def execute_plain(agent, prompt_msgs: list[BaseMessage]):
    """Run agent without SQL-specific retry logic."""
    return await agent.ainvoke({"messages": [get_bot_role()] + prompt_msgs})


sql_parser = PydanticOutputParser(pydantic_object=SQLQuery)

sql_prompt = PromptTemplate(
    template=(
        "You are an expert SQL assistant.\n"
        "Given an input question, first create a syntactically correct SQLite 3 query to run, "
        "Use the exact user_id: {user_id} when you need to query data for the user who is asking the questions,\n"
        "Database tables and schema: {schema}\n"
        "Past conversation that might or not be related to the question: {history}\n"
        "User question: {question}\n"
        "Return a SQL query wrapped in this JSON schema:\n"
        "{format_instructions}"
        ".\n You already tried with this query {last_query} and for this error {last_error}"
    ),
    input_variables=["question", "user_id", "last_query", "last_error"],
    partial_variables={"format_instructions": sql_parser.get_format_instructions()},
)


async def execute_sql_with_structured(
    model: BaseChatModel,
    question: str,
    history: str,
    schema: str,
    user_id: int,
) -> dict[str, list[BaseMessage]]:
    """
    Execute SQL with retries and error feedback to the LLM.
    """
    last_error = None
    sql_query: Optional[SQLQuery] = None
    rows = None

    for attempt in range(MAX_RETRIES):
        # --- Step 1: Generate SQL ---
        sql_input = sql_prompt.format_prompt(
            question=question,
            history=history,
            schema=schema,
            user_id=user_id,
            last_query=sql_query.query if sql_query else "",
            last_error=last_error or "",
        )
        sql_output = await model.ainvoke(sql_input.to_string())

        try:
            sql_query = sql_parser.parse(str(sql_output.content))

            # --- Step 2: Run the query ---
            if sql_query:
                rows = execute_database(sql_query.query)
                if len(rows) == 0:
                    raise ValueError(
                        "SQL query returned no results. Try a different query."
                    )
                else:
                    break

        except Exception as e:
            last_error = str(e)
            if attempt == MAX_RETRIES - 1:
                return {
                    "messages": [
                        AIMessage(
                            content=f"Query failed after {MAX_RETRIES} retries. Error: {last_error}"
                        )
                    ]
                }
            # continue loop, LLM will try again with error feedback
    if sql_query is None:
        return {
            "messages": [
                AIMessage(
                    content=f"Failed to generate a valid SQL query after {MAX_RETRIES} retries."
                )
            ]
        }
    # --- Step 3: Interpret results ---
    interpretation_prompt = PromptTemplate(
        template=(
            "The user asked: {question}\n\n"
            "The SQL query executed was:\n{query}\n\n"
            "The database returned:\n{rows}\n\n"
            "Explain this result clearly for the user."
        ),
        input_variables=["question", "query", "rows"],
    )

    interp_input = interpretation_prompt.format_prompt(
        question=question,
        query=sql_query.query,
        rows=rows,
    )
    interp_output = await model.ainvoke(interp_input.to_string())

    return {"messages": [AIMessage(content=interp_output.content)]}


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

            # Keyword-based routing
            user_original_msg = ctx.user_question
            user_msg_lower = user_original_msg.lower()

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
                "clutch",
                "rank",
            ]
            schema = get_user_schema()  # Always

            if any(keyword in user_msg_lower for keyword in keywords_full_match_info):
                schema += get_stats_schema()

            keywords_tournament = ["tournament", "bet"]
            if any(keyword in user_msg_lower for keyword in keywords_tournament):
                schema += get_tournament_schema()

            keywords_schedule = [
                "time",
                "date",
                "schedule",
                "activity",
            ]
            if any(keyword in user_msg_lower for keyword in keywords_schedule):
                schema += get_activity_schema()

            user_original_msg = ctx.user_question
            history_to_include = ctx.message_history
            history_text = "\n".join(history_to_include)

            model = get_model(ctx.provider)
            query_result = await execute_sql_with_structured(
                model=model,
                schema=schema,
                question=str(user_original_msg),
                history=history_text,
                user_id=ctx.user_discord_id,
            )
            if isinstance(query_result, str) and query_result.startswith("SQL_ERROR:"):
                return {
                    "messages": state["messages"]
                    + [
                        AIMessage(
                            content="I ran into a database issue while handling your request. Please try again later."
                        )
                    ]Z
                }
            return query_result

        except GraphRecursionError as e:
            print_error_log(f"Agent stopped due to max iterations: {e}")

    async def message_gen_step(self, state: State, config: RunnableConfig):
        """
        Craft the answer with the channel history, the user question and potentially information from the database
        """
        ctx: AIConversationCustomContext = config["configurable"]["ctx"]
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            structured_msg = last_msg.content
        else:
            structured_msg = str(last_msg)

        user_original_msg = ctx.user_question
        history_to_include = ctx.message_history
        history_text = "\n".join(history_to_include)

        prompt_msgs: list[BaseMessage] = [
            SystemMessage(content=system_instruction_when_bot_mentioned),
            SystemMessage(
                content=f"Channel history:\n{history_text}\n\nUser question to answer:\n{user_original_msg}\n"
            ),
        ]

        # Personalize
        context = ""
        if ctx.user_rank == "Champion":
            context += "In the message, call the user 'champion'. "
            context += "The user like sarcasm, so answer in a sarcastic tone. "
        else:
            context += "You are a bot that is friendly, helpful and professional. You should not be rude or sarcastic. "

        prompt_msgs.append(SystemMessage(content=context))
        prompt_msgs.append(
            HumanMessage(
                content=(
                    f"Turn this into a concise message that is well formatted for Discord (use triple tick for table): {structured_msg}. \n "
                    "Never mention anything about database, or user id, or guid or SQL or internal details. \n"
                    "Only show the display name when you refer to the user.\n"
                )
            )
        )
        model = get_model(ctx.provider)
        final_output = await model.ainvoke(prompt_msgs)
        return {"messages": state["messages"] + [final_output]}
