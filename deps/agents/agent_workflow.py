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

MAX_RETRIES_SQL_VALID = 5
MAX_RETRIES_ANSWERING_JUDGE = 5
MAX_ITERATIONS = 3
RECURSION_LIMIT = 2 * MAX_ITERATIONS + 1
MAX_HISTORY_MESSAGES = 20

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


def execute_database(query: str) -> list[Any]:
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
        "Pass SQL queries that you tried and failed: {sql_history}\n"
        "User question: {question}\n"
        "Return a SQL query wrapped in this JSON schema:\n"
        "{format_instructions}"
        ".\n You already tried with this query {last_query} and for this error {last_error}"
    ),
    input_variables=[
        "question",
        "user_id",
        "last_query",
        "last_error",
        "history",
        "schema",
        "sql_history",
    ],
    partial_variables={"format_instructions": sql_parser.get_format_instructions()},
)

interpretation_prompt = PromptTemplate(
    template=(
        "The user asked: {question}\n\n"
        "The SQL query executed was:\n{query}\n\n"
        "The database returned:\n{rows}\n\n"
        "I want you to determine if the rows contains the information needed to answer the user's question. If yes, return only 'yes'. If no, return only 'no'."
    ),
    input_variables=["question", "query", "rows"],
)

judge_answer_prompt = PromptTemplate(
    template=(
        "You are an expert assistant that judges if the answer provided is sufficient to answer the user's question.\n"
        "User question: {question}\n"
        "Answer provided: {answer}\n"
        "If the answer is sufficient, respond with 'yes'. If more information is needed, respond with 'no'."
    ),
    input_variables=["question", "answer"],
)


def select_model(
    state: AgentState,
    runtime: Runtime[AIConversationCustomContext],
    ctx: AIConversationCustomContext,
) -> BaseChatModel:
    """
    Return the right model depending of the provider
    """

    # With dynamic model selection, you must bind tools explicitly
    del state, runtime  # Unused parameters
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
    """
    Keep track if the answer from the chatbot is sufficient or if more info is needed
    """
    is_answer_sufficient: bool

    """
    Counter to track how many times more info was requested in a workflow
    """
    need_more_info_counter: int


class AIConversationWorkflow:
    """
    Class that describe the workflow of a user and AI communicating back and forth (conversation)
    """

    def __init__(self, ctx: AIConversationCustomContext):
        self.agent = create_react_agent(
            lambda state, runtime: select_model(state, runtime, ctx),
            tools=[database_tool],
        )
        self.llm = get_model(ctx.provider)
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("needs_more_info", self.needs_more_info_step)
        graph_builder.add_node("gather_more_info", self.gather_more_info_step)
        graph_builder.add_node("message_gen", self.message_gen_step)
        graph_builder.add_node("judge_answer", self.judge_answer_step)

        # Order of execution
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", "needs_more_info")
        graph_builder.add_edge("gather_more_info", "chatbot")
        graph_builder.add_conditional_edges(
            "needs_more_info",
            self.needs_more_info_condition,
            path_map={
                "yes": "gather_more_info",
                "no": "message_gen",
            },
        )
        graph_builder.add_edge("message_gen", "judge_answer")
        graph_builder.add_conditional_edges(
            "judge_answer",
            lambda state: "yes" if state.get("is_answer_sufficient") else "no",
            path_map={
                "yes": END,
                "no": "gather_more_info",
            },
        )

        self.graph = graph_builder.compile()

    async def execute_sql_with_structured(
        self,
        question: str,
        history: str,
        schema: str,
        user_id: int,
    ) -> tuple[Optional[SQLQuery], list[Any]]:
        """
        Execute SQL
        """
        last_error = None
        ai_sql_query: Optional[SQLQuery] = None
        rows = None

        sql_history = ""
        last_generated_sql_query_by_ai = ""
        for attempt in range(MAX_RETRIES_SQL_VALID):
            # --- Step 1: Generate SQL ---
            sql_input = sql_prompt.format_prompt(
                question=question,
                history=history,
                sql_history=sql_history,
                schema=schema,
                user_id=user_id,
                last_query=last_generated_sql_query_by_ai,
                last_error=last_error or "",
            )
            response = await self.llm.ainvoke(
                [HumanMessage(content=sql_input.to_string())]
            )

            try:
                ai_sql_query = sql_parser.parse(str(response.content))
                last_generated_sql_query_by_ai = (
                    ai_sql_query.query if ai_sql_query else ""
                )
                # --- Step 2: Run the query ---
                if ai_sql_query:
                    rows = execute_database(ai_sql_query.query)
                    return ai_sql_query, rows
            except ValueError as e:
                last_error = str(e)
                if attempt == MAX_RETRIES_SQL_VALID - 1:
                    return ai_sql_query, []
            finally:
                sql_history += f"Attempt {attempt + 1}:\nQuery: {last_generated_sql_query_by_ai}\nError: {last_error}\n"
                # continue loop, LLM will try again with error feedback
        return ai_sql_query, []

    def needs_more_info_condition(self, state: State) -> str:
        # Example: chatbot put its evaluation in the state
        if state.get("is_answer_sufficient"):
            return "no"
        else:
            return "yes"

    async def chatbot(self, state: State, config: RunnableConfig):
        try:
            ctx: AIConversationCustomContext = config["configurable"]["ctx"]

            user_original_msg = ctx.user_question
            user_msg_lower = user_original_msg.lower()

            # --- Keyword Routing ---
            schema = get_user_schema()

            if any(
                k in user_msg_lower
                for k in [
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
            ):
                schema += get_stats_schema()

            if any(k in user_msg_lower for k in ["tournament", "bet"]):
                schema += get_tournament_schema()

            if any(
                k in user_msg_lower
                for k in [
                    "time",
                    "hour",
                    "minute",
                    "second",
                    "when",
                    "date",
                    "schedule",
                    "activity",
                ]
            ):
                schema += get_activity_schema()

            # --- Include chat history ---
            history_text = "\n".join(ctx.message_history[:MAX_HISTORY_MESSAGES])

            # --- Execute SQL ---
            sql_query, rows = await self.execute_sql_with_structured(
                schema=schema,
                question=str(user_original_msg),
                history=history_text,
                user_id=ctx.user_discord_id,
            )

            # Format content properly
            result_payload = {
                "type": "sql_result",
                "sql_query": sql_query.query if sql_query else None,
                "rows": rows,
            }

            new_message = AIMessage(content=[result_payload])

            return State(
                messages=state["messages"] + [new_message],
                is_answer_sufficient=state.get("is_answer_sufficient", False),
                need_more_info_counter=state.get("need_more_info_counter", 0),
            )

        except GraphRecursionError as e:
            print_error_log(f"Agent stopped due to max iterations: {e}")
            raise e

    async def needs_more_info_step(self, state: State, config: RunnableConfig):
        """
        Determine if the answer from the chatbot is sufficient or if more info is needed
        We should have a query that returned rows to analyze
        If the LLM determines that more info is needed, we set is_answer_sufficient to False
        """
        last_msg = state["messages"][-1]
        sql_query = last_msg.content[0]["sql_query"]
        rows = last_msg.content[0]["rows"]
        new_state = state.copy()

        # Check if the SQL response was reasonable
        question = config["configurable"]["ctx"].user_question
        prompt_msgs = interpretation_prompt.format_prompt(
            question=question, query=sql_query, rows=rows
        )
        final_output = await self.llm.ainvoke(
            [HumanMessage(content=prompt_msgs.to_string())]
        )

        answer_text = str(final_output.content).strip().lower()

        new_state["is_answer_sufficient"] = (
            True if str(answer_text).lower() == "yes" else False
        )

        return new_state

    def gather_more_info_step(self, state: State) -> State:
        """
        Ask the user for more information to clarify their request or re-perform the AI with more context
        """
        if state["need_more_info_counter"] >= MAX_RETRIES_ANSWERING_JUDGE:
            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="The previous answer was insufficient. Let me ask for more details. Could you provide more information to help me better assist you?"
                    )
                ],
                "is_answer_sufficient": False,
                "need_more_info_counter": state["need_more_info_counter"],
            }
        else:

            return {
                "messages": state["messages"]
                + [
                    AIMessage(
                        content="The previous answer was insufficient. Let's try again by adding the previous answer to the original question for more context."
                    )
                ],
                "is_answer_sufficient": state[
                    "is_answer_sufficient"
                ],  # That was set to false in the needs_more_info_step step (previous step)
                "need_more_info_counter": state["need_more_info_counter"] + 1,
            }

    async def message_gen_step(self, state: State, config: RunnableConfig):
        """
        Craft the answer with the channel history, the user question and potentially information from the database
        """
        ctx: AIConversationCustomContext = config["configurable"]["ctx"]
        last_msg = state["messages"][-1]
        if (
            isinstance(last_msg, AIMessage)
            or isinstance(last_msg, HumanMessage)
            or isinstance(last_msg, SystemMessage)
        ):
            structured_msg = last_msg.content
        else:
            structured_msg = str(last_msg)

        user_original_msg = ctx.user_question
        history_to_include = ctx.message_history
        history_text = "\n".join(history_to_include[:MAX_HISTORY_MESSAGES])

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
        final_output = await self.llm.ainvoke(prompt_msgs)
        return {"messages": state["messages"] + [final_output]}

    async def judge_answer_step(self, state: State, config: RunnableConfig):
        """
        Judge if the answer generated is sufficient to answer the user's question
        """
        last_msg = state["messages"][-1]
        new_state = state.copy()

        # Judge
        ctx: AIConversationCustomContext = config["configurable"]["ctx"]
        user_original_msg = ctx.user_question
        message_generated_previous_step = last_msg.content
        prompt_msgs = judge_answer_prompt.format_prompt(
            question=user_original_msg, answer=message_generated_previous_step
        )
        final_output = await self.llm.ainvoke(
            [HumanMessage(content=prompt_msgs.to_string())]
        )

        answer_text = str(final_output.content).strip().lower()
        if answer_text == "yes":
            new_state["is_answer_sufficient"] = True
        else:
            new_state["is_answer_sufficient"] = False
            new_state["need_more_info_counter"] += 1
