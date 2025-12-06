from typing import Any, List, Optional, Dict, TypedDict
from dataclasses import dataclass, field
import sqlite3
import contextvars
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from deps.database.utils_database import get_table_schema
from deps.database.system_database import DBName, DatabaseManager
from deps.rules.system_instructions import (
    system_instruction_when_bot_mentioned,
    tool_get_schema_description,
    tool_get_all_schema_tool_description,
    tool_sql_query_description,
    tool_discord_format_message_description,
)

# ---- Constants ----
MAX_AGENT_STEPS = 30  # prevent infinite loops
MAX_SQL_RETRIES = 20
MAX_HISTORY_MESSAGES = 20

# ---- Initialize chat models (adjust names/providers if needed) ----
openai_model = init_chat_model("openai:gpt-5-mini")
google_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


# ---- Pydantic model representing the structured SQL the LLM should return ----
class SQLQuery(BaseModel):
    query: str
    explain: Optional[str] = None


# ---- Runtime context ----
@dataclass
class AIConversationCustomContext:
    provider: str = "openai"  # "openai" or "google"
    message_history: List[str] = field(default_factory=list)
    user_question: str = ""
    user_discord_id: int = 0
    user_discord_display_name: str = ""
    user_rank: str = ""

# ---- Thread-local context storage for tools ----
_current_context: contextvars.ContextVar[Optional[AIConversationCustomContext]] = contextvars.ContextVar(
    'current_context', default=None
)


# --- Define the graph state ---
class WorkflowState(TypedDict, total=False):
    raw_message: Optional[str]
    formatted_message: Optional[str]
    needs_retry: Optional[bool]
    skip_db: Optional[bool]


# ---- Database helpers ----
def execute_database(query: str) -> List[Any]:
    """Execute query against your SIEGE DB (synchronous)."""
    with DatabaseManager.get_database_manager() as db:
        cursor = db.get_cursor(DBName.SIEGE)
        cursor.execute(query)
        results = cursor.fetchall()
        return results


# ---- Schema helpers ----
def get_user_schema() -> str:
    return get_table_schema("user_info")


def get_stats_schema() -> str:
    return (
        f"{get_table_schema('user_full_match_info')}\n"
        f"{get_table_schema('user_full_stats_info')}\n"
    )


def get_tournament_schema() -> str:
    return (
        f"{get_table_schema('tournament')}\n"
        f"{get_table_schema('tournament_guild')}\n"
        f"{get_table_schema('tournament_game')}\n"
        f"{get_table_schema('user_tournament')}\n"
        f"{get_table_schema('tournament_team_members')}\n"
        f"{get_table_schema('bet_user_tournament')}\n"
        f"{get_table_schema('bet_game')}\n"
        f"{get_table_schema('bet_ledger_entry')}\n"
        f"{get_table_schema('bet_user_game')}\n"
    )


def get_activity_schema() -> str:
    return f"{get_table_schema('user_activity')}\nThe field `event` in the table `user_activity` can be `connect` or `disconnect`.\n"


@tool("get_schema", description=tool_get_schema_description)
async def get_schema_tool() -> str:
    """Get relevant database schema based on the user question."""
    ctx = _current_context.get()
    if not ctx:
        return "Error: No context available"
    
    user_question = ctx.user_question
    name = (user_question or "").lower()
    schema = get_user_schema()
    if any(
        k in name
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
        schema += "\n" + get_stats_schema()
    if "tourn" in name or "tournament" in name or "bet" in name:
        schema += "\n" + get_tournament_schema()
    if any(
        k in name
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
        schema += "\n" + get_activity_schema()
    return schema


@tool("get_all_schema", description=tool_get_all_schema_tool_description)
async def get_all_schema() -> str:
    return (
        f"{get_user_schema()}\n"
        f"{get_stats_schema()}\n"
        f"{get_tournament_schema()}\n"
        f"{get_activity_schema()}\n"
    )


# Prompt template for SQL generation; format_instructions will be injected
SQL_PROMPT_TEMPLATE = (
    "You are an expert SQL assistant for SQLite 3.\n"
    "When appropriate, use the exact user_id: {user_discord_id}.\n"
    "Database schema provided:\n{schema}\n\n"
    "Past conversation / history:\n{history}\n\n"
    "Previous SQL attempts and errors (if any):\n{sql_history}\n\n"
    "User question: {user_question}\n\n"
    "Return ONLY a JSON object that matches this pydantic schema:\n{format_instructions}\n\n"
    "Important safety notes:\n"
    "- Only produce read-only SQL (SELECT / WITH). Do NOT produce INSERT/UPDATE/DELETE/ALTER/DROP/ATTACH/DETACH.\n"
    "- Do not include semicolons.\n"
    "- Keep queries reasonably bounded (use LIMIT where appropriate).\n"
)


@tool("query_database_tool", description=tool_sql_query_description)
async def query_database_tool(schema: str) -> Dict[str, Any]:
    """
    Generate SQL using the LLM with retries. On each retry, feed back sql_history + last error so the LLM can improve.
    If a "no such table" error is detected, immediately triggers a suggestion to use get_all_schema.
    Returns:
        {
          "query": str | None,
          "rows": list | None,
          "error": str | None,
          "attempts": int,
          "needs_full_schema": bool  # True if get_all_schema should be called
        }
    """
    ctx = _current_context.get()
    if not ctx:
        return {"query": None, "rows": [], "error": "No context available", "attempts": 0, "needs_full_schema": False}
    
    user_question = ctx.user_question
    user_discord_id = ctx.user_discord_id
    discord_message_history = "\n".join(ctx.message_history[:MAX_HISTORY_MESSAGES])
    provider = ctx.provider
    model: BaseChatModel = openai_model if provider == "openai" else google_model

    sql_history: List[Dict[str, str]] = []

    last_query = None
    last_error = None
    no_such_table_found = False

    for attempt in range(1, MAX_SQL_RETRIES + 1):
        # Build a readable sql_history string for the prompt
        sql_history_text = ""
        if sql_history:
            parts = []
            for i, entry in enumerate(sql_history):
                q = entry.get("sql", "")
                e = entry.get("error", "")
                parts.append(f"Attempt {i+1}:\nQuery:\n{q}\nError:\n{e}\n")
            sql_history_text = "\n".join(parts)

        prompt_str = SQL_PROMPT_TEMPLATE.format(
            user_question=user_question,
            user_discord_id=user_discord_id,
            schema=schema,
            history=discord_message_history,
            sql_history=sql_history_text,
            format_instructions=SQLQuery.model_json_schema(),
        )

        # Ask the model to produce structured JSON that matches SQLQuery
        try:
            response = await model.ainvoke([HumanMessage(content=prompt_str)])
        except (ValueError, RuntimeError) as e:
            last_error = f"LLM call failed: {e}"
            sql_history.append({"sql": "", "error": last_error})
            continue

        # Parse the structured JSON returned by the model (pydantic v2)
        try:
            parsed: SQLQuery = SQLQuery.model_validate_json(response.content)
            last_query = parsed.query.strip()
        except Exception as e:
            last_error = f"Parse error: {e}"
            sql_history.append({"sql": str(response.content), "error": last_error})
            continue

        # Basic safety checks before executing
        lowered = last_query.lower()
        # Forbid destructive statements
        destructive_keywords = [
            "delete",
            "update",
            "drop",
            "alter",
            "insert",
            "attach",
            "detach",
        ]
        if any(k in lowered.split() for k in destructive_keywords):
            last_error = "Refused to execute destructive SQL."
            sql_history.append({"sql": last_query, "error": last_error})
            return {
                "query": last_query,
                "rows": [],
                "error": last_error,
                "attempts": attempt,
                "needs_full_schema": False,
            }

        # Disallow semicolons (prevent multi-statement)
        if ";" in last_query:
            last_error = "Semicolons are not allowed in SQL."
            sql_history.append({"sql": last_query, "error": last_error})
            continue

        # Ensure it's a SELECT or WITH query (simple heuristic)
        if not (lowered.startswith("select") or lowered.startswith("with")):
            last_error = "Only SELECT or WITH queries are allowed."
            sql_history.append({"sql": last_query, "error": last_error})
            continue

        # Execute the query
        try:
            rows = execute_database(last_query)
            return {
                "query": last_query,
                "rows": rows,
                "error": None,
                "attempts": attempt,
                "needs_full_schema": False,
            }
        except sqlite3.OperationalError as exec_err:
            error_str = str(exec_err).lower()
            # Check if this is a "no such table" error
            if "no such table" in error_str:
                no_such_table_found = True
                last_error = f"Execution error: {exec_err}"
                sql_history.append({"sql": last_query, "error": last_error})
                # Immediately signal that full schema is needed instead of retrying
                return {
                    "query": last_query,
                    "rows": [],
                    "error": last_error,
                    "attempts": attempt,
                    "needs_full_schema": True,
                }
            else:
                last_error = f"Execution error: {exec_err}"
                sql_history.append({"sql": last_query, "error": last_error})
                # Continue loop so model can see this execution error in next attempt
                continue

    # If we exhausted attempts
    return {
        "query": last_query,
        "rows": [],
        "error": last_error or "Failed to produce valid SQL.",
        "attempts": MAX_SQL_RETRIES,
        "needs_full_schema": no_such_table_found,
    }


# Format prompt for final message generation
FORMAT_PROMPT_TEMPLATE = (
    "You are a friendly assistant formatting a Discord message. Do NOT mention SQL, DB internals, user ids, or internal IDs.\n"
    "Channel history:\n{history}\n\n"
    "User question:\n{question}\n\n"
    "SQL used (for reference only):\n{sql}\n\n"
    "Database rows (for reference only):\n{rows}\n\n"
    "Produce a concise Markdown reply suitable for Discord. If there is tabular data, use triple-backtick blocks for tables.\n"
)


async def format_message_for_discord(
    sql: Optional[str],
    rows: List[Any],
    runtime: ToolRuntime[AIConversationCustomContext],
) -> str:
    """
    Formats a final message for Discord. Calls the LLM to produce the textual reply.
    """
    ctx = runtime.context
    provider = ctx.provider
    user_question = ctx.user_question
    user_rank = ctx.user_rank
    discord_message_history = "\n".join(ctx.message_history[:MAX_HISTORY_MESSAGES])
    model = openai_model if provider == "openai" else google_model

    # Slight tone modification for rank
    system_text = system_instruction_when_bot_mentioned
    if user_rank == "Champion":
        # Keep it friendly but slightly different if you want
        system_text += " Address the user as 'champion' and keep a light witty tone."

    prompt_str = FORMAT_PROMPT_TEMPLATE.format(
        history=discord_message_history or "",
        question=user_question,
        sql=sql or "",
        rows=str(rows or []),
    )

    final_output = await model.ainvoke(
        [SystemMessage(content=system_text), HumanMessage(content=prompt_str)]
    )
    # Ensure we return a string
    return str(final_output.content)


# --- Node definitions ---
def interpret_question_node(
    state: WorkflowState, runtime: ToolRuntime[AIConversationCustomContext]
) -> Dict[str, Any]:
    """
    Decide whether a DB query is needed. If not, we can skip DB and return a default message.
    """
    question = runtime.context.user_question
    llm: BaseChatModel = (
        openai_model if runtime.context.provider == "openai" else google_model
    )
    resp = llm.invoke(
        [
            HumanMessage(
                content=f"User asked: {question}\nDo you need to query thre system private database to answer or you can answer with your general knowledge? Answer yes to access the database or no to directly answer."
            )
        ]
    )
    needs = "yes" in resp.content.lower()
    return {"needs_retry": False, "schema": None, "skip_db": not needs}


async def get_access_database_knowledge_node(
    state: WorkflowState, runtime: ToolRuntime[AIConversationCustomContext]
) -> Dict[str, Any]:
    question = runtime.context.user_question
    llm: BaseChatModel = (
        openai_model if runtime.context.provider == "openai" else google_model
    )
    
    # Set the context for tools to access
    _current_context.set(runtime.context)
    
    try:
        # Bind the tools to the LLM
        tools = [get_schema_tool, get_all_schema, query_database_tool]
        llm_with_tools = llm.bind_tools(tools)
        
        # Create a mapping of tool names to tool objects for execution
        tool_map = {tool.name: tool for tool in tools}
        
        # Agentic loop: keep invoking until we get a final response (not a tool call)
        messages = [
            HumanMessage(
                content=f"User asked: {question}\nUse the tools to get the database schema and query to get the data you need to answer the question."
            )
        ]
        
        for _ in range(MAX_AGENT_STEPS):
            resp = await llm_with_tools.ainvoke(messages)
            messages.append(resp)
            
            # Check if the response contains tool calls
            if not hasattr(resp, "tool_calls") or not resp.tool_calls:
                # No more tool calls, return final response
                return Command(update={"raw_message": resp.content})
            
            # Process tool calls
            for tool_call in resp.tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call.get("args", {})
                
                # Execute the tool using ainvoke
                if tool_name in tool_map:
                    tool_obj = tool_map[tool_name]
                    result = await tool_obj.ainvoke(tool_input)
                else:
                    result = f"Unknown tool: {tool_name}"
                
                # Add tool result to messages
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        
        # If we exhausted steps, return what we have
        return Command(update={"raw_message": resp.content})
    finally:
        # Clear the context when done
        _current_context.set(None)


async def format_output_node(
    state: WorkflowState, runtime: ToolRuntime[AIConversationCustomContext]
) -> Dict[str, Any]:
    formatted = await format_message_for_discord(
        sql=state.get("query"), rows=state.get("rows", []), runtime=runtime
    )
    return Command(update={"formatted_message": formatted})


def evaluate_node(
    state: WorkflowState, runtime: ToolRuntime[AIConversationCustomContext]
) -> Dict[str, Any]:
    question = runtime.context.user_question
    formatted = state.get("formatted_message", "")
    llm: BaseChatModel = (
        openai_model if runtime.context.provider == "openai" else google_model
    )
    resp = llm.invoke(
        [
            HumanMessage(
                content=f"User asked: {question}\nResult:\n{formatted}\nDoes this answer the question? Answer only with 'yes' or 'no'."
            )
        ]
    )
    needs = not ("yes" in resp.content.lower())
    return Command(update={"needs_retry": needs})


# --- Build the graph ---
builder = StateGraph(WorkflowState, AIConversationCustomContext)

builder.add_node("interpret_question", interpret_question_node)
builder.add_node(
    "get_access_database_knowledge_node", get_access_database_knowledge_node
)
builder.add_node("format_output", format_output_node)
builder.add_node("evaluate", evaluate_node)

# Entry edge
builder.add_edge(START, "interpret_question")


def route_after_interpret(state: WorkflowState) -> str:
    return (
        "get_access_database_knowledge_node"
        if not state.get("skip_db", False)
        else "format_output"
    )


builder.add_conditional_edges("interpret_question", route_after_interpret)
builder.add_edge("get_access_database_knowledge_node", "format_output")
builder.add_edge("format_output", "evaluate")


def route_after_eval(state: WorkflowState) -> str:
    return (
        "get_access_database_knowledge_node" if state.get("needs_retry", False) else END
    )


builder.add_conditional_edges("evaluate", route_after_eval)

graph = builder.compile()


async def run_workflow(ctx: AIConversationCustomContext) -> str:
    final_state = await graph.ainvoke({}, context=ctx)
    return final_state.get(
        "formatted_message", "Sorry — I could not answer your question."
    )
