from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

from deps.database.utils_database import get_table_schema
from deps.database.system_database import DBName, DatabaseManager
from deps.rules.system_instructions import system_instruction_when_bot_mentioned

# ---- Constants ----
MAX_AGENT_STEPS = 20  # prevent infinite loops
MAX_SQL_RETRIES = 8
MAX_HISTORY_MESSAGES = 20

# ---- Initialize chat models (adjust names/providers if needed) ----
openai_model = init_chat_model("openai:gpt-4.1")
google_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


# ---- Pydantic model representing the structured SQL the LLM should return ----
class SQLQuery(BaseModel):
    query: str
    explain: Optional[str] = None


sql_parser = PydanticOutputParser(pydantic_object=SQLQuery)


# ---- Runtime context ----
@dataclass
class AIConversationCustomContext:
    provider: str = "openai"  # "openai" or "google"
    message_history: List[str] = field(default_factory=list)
    user_question: str = ""
    user_discord_id: int = 0
    user_discord_display_name: str = ""
    user_rank: str = ""


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
    return f"{get_table_schema('user_full_match_info')}\n{get_table_schema('user_full_stats_info')}"


def get_tournament_schema() -> str:
    return (
        f"{get_table_schema('tournament')}\n"
        f"{get_table_schema('tournament_guild')}\n"
        f"{get_table_schema('tournament_game')}\n"
        f"{get_table_schema('user_tournament')}\n"
        f"{get_table_schema('tournament_team_members')}"
    )


def get_activity_schema() -> str:
    return f"{get_table_schema('user_activity')}\nThe field in the table user_activity can be `connect` or `disconnect`."


# Create wrapped tool functions outside the class for proper LangGraph integration
def _create_context_tools(ctx: AIConversationCustomContext):
    """Create tool functions with context baked in."""
    
    @tool("get_schema")
    async def get_schema_tool_wrapped() -> str:
        """
        Select and return the relevant database schema(s) based on the content of the user's question.

        🛠 Purpose:
        - Dynamically determine which database tables are relevant for a given query.
        - Provide the agent with textual schema information that can be used to generate
            correct SQL queries.
        - Ensures the LLM only sees the schema it needs for the current question.

        📥 Input:
        - user_question: The former question or message from the user that must be answered.

        📤 Output:
        - A string containing the concatenated schema text for the relevant tables.
        - Includes user, stats, tournament, and activity tables as needed.

        🧠 When to use:
        - Call this tool **before generating SQL queries**.
        - The agent should use this tool when the user question involves:
            • Player statistics, matches, or general data queries → include stats schema
            • Tournaments, bets, or team info → include tournament schema
            • Timing, schedule, or activity tracking → include activity schema
        - Use this tool whenever the LLM needs structured table information
            to generate valid queries.

        🚫 Notes:
        - Only the user_question is needed; other context like user_id or rank is not required.
        - Returns plain text; the LLM should not attempt to interpret or modify the schema.
        """
        return await get_schema_tool(user_question=ctx.user_question)

    @tool("query_sql_query_database")
    async def query_database_tool_wrapped(schema: str) -> Dict[str, Any]:
        """
        Generate and execute a SQL query based on the user's question and the provided database schema.

        🛠 Purpose:
        - Create a SELECT query that accurately retrieves data relevant to the user's request.
        - Execute the generated SQL against the SIEGE database and return the results.
        - Handle retries and errors in SQL generation and execution.

        📥 Required arguments (must be provided by the agent):
        - schema: The database schema text relevant to the user's question.

        📤 Output:
        - A dictionary containing:
            • "query": The generated SQL query string (or None if failed).
            • "rows": The list of result rows returned from the database (or empty list).
            • "error": Any error message encountered during generation/execution (or None).
            • "attempts": The number of attempts taken to generate/execute valid SQL.

        🧠 When to use:
        - Call this tool after obtaining the relevant schema using the get_schema tool.
        - Use this tool when the user question requires data retrieval from the database.
        - The agent should invoke this tool whenever it needs to fetch data to answer the user's query

        🚫 Notes:
        - The tool automatically handles user context (question, ID, history); do not pass these manually.
        - Only read-only SELECT queries are allowed; destructive operations are blocked.
        """
        return await query_database_tool(
            user_question=ctx.user_question,
            user_discord_id=ctx.user_discord_id,
            discord_message_history="\n".join(
                ctx.message_history[:MAX_HISTORY_MESSAGES]
            ),
            provider=ctx.provider,
            schema=schema,
        )

    @tool("format_message_for_discord")
    async def format_discord_message_tool_wrapped(sql: Optional[str], rows: List[Any]) -> str:
        """
        Use this tool ONLY to generate the final textual response that will be sent back to the Discord user.

        🛠 Purpose:
        - Format a complete and human-readable reply from the bot.
        - Combine the user's original question, relevant Discord message history, executed SQL (if any),
            and the retrieved query result rows.
        - Optionally adjust tone based on the user's rank (e.g., "Champion").

        📥 Required arguments (must be provided by the agent):
        - discord_message_history: Recent messages from the current Discord conversation.
        - user_question: The user's most recent message or query.
        - sql: The SQL statement used to fetch data (if any) or an empty string.
        - rows: The result returned from the database query.

        🚫 DO NOT provide the following arguments in tool calls (they are injected automatically):
        - user_discord_id
        - provider
        - user_rank

        📤 Output:
        - A final, properly formatted text response for Discord.
        - Must NOT include markdown code fences or system-like formatting—return plain text only.

        🔁 When to call:
        - After database query results are successfully retrieved.
        - As the final step before returning the final answer to the user.
        """
        return await format_discord_message_tool(
            sql=sql,
            rows=rows,
            user_discord_id=ctx.user_discord_id,
            user_rank=ctx.user_rank,
            discord_message_history="\n".join(
                ctx.message_history[:MAX_HISTORY_MESSAGES]
            ),
            user_question=ctx.user_question,
            provider=ctx.provider,
        )

    return [get_schema_tool_wrapped, query_database_tool_wrapped, format_discord_message_tool_wrapped]


class ContextTools:
    def __init__(self, ctx: AIConversationCustomContext):
        self.ctx = ctx


async def get_schema_tool(user_question: str) -> str:
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


# Prompt template for SQL generation; format_instructions will be injected
SQL_PROMPT = PromptTemplate(
    template=(
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
    ),
    input_variables=["user_question", "user_discord_id", "schema", "history", "sql_history"],
    partial_variables={"format_instructions": sql_parser.get_format_instructions()},
)


async def query_database_tool(
    user_question: str,
    user_discord_id: int,
    schema: str,
    discord_message_history: str = "",
    provider: str = "openai",
) -> Dict[str, Any]:
    """
    Generate SQL using the LLM with retries. On each retry, feed back sql_history + last error so the LLM can improve.
    Returns:
        {
          "query": str | None,
          "rows": list | None,
          "error": str | None,
          "attempts": int
        }
    """

    model: BaseChatModel = openai_model if provider == "openai" else google_model

    sql_history: List[Dict[str, str]] = []

    last_query = None
    last_error = None

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

        prompt = SQL_PROMPT.format_prompt(
            user_question=user_question,
            user_discord_id=user_discord_id,
            schema=schema,
            history=discord_message_history,
            sql_history=sql_history_text,
        )

        # Ask the model to produce structured JSON that matches SQLQuery
        try:
            response = await model.ainvoke([HumanMessage(content=prompt.to_string())])
        except Exception as e:
            last_error = f"LLM call failed: {e}"
            sql_history.append({"sql": "", "error": last_error})
            continue

        # Parse the structured JSON returned by the model
        try:
            parsed: SQLQuery = sql_parser.parse(str(response.content))
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
            }
        except Exception as exec_err:
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
    }


# Format prompt for final message generation
FORMAT_PROMPT = PromptTemplate(
    template=(
        "You are a friendly assistant formatting a Discord message. Do NOT mention SQL, DB internals, user ids, or internal IDs.\n"
        "Channel history:\n{history}\n\n"
        "User question:\n{question}\n\n"
        "SQL used (for reference only):\n{sql}\n\n"
        "Database rows (for reference only):\n{rows}\n\n"
        "Produce a concise Markdown reply suitable for Discord. If there is tabular data, use triple-backtick blocks for tables.\n"
    ),
    input_variables=["history", "question", "sql", "rows"],
)


async def format_discord_message_tool(
    discord_message_history: str,
    user_question: str,
    sql: Optional[str],
    rows: List[Any],
    user_discord_id: int,
    user_rank: str = "",
    provider: str = "openai",
) -> str:
    """
    Formats a final message for Discord. Calls the LLM to produce the textual reply.
    """
    model = openai_model if provider == "openai" else google_model

    # Slight tone modification for rank
    system_text = system_instruction_when_bot_mentioned
    if user_rank == "Champion":
        # Keep it friendly but slightly different if you want
        system_text += " Address the user as 'champion' and keep a light witty tone."

    prompt = FORMAT_PROMPT.format_prompt(
        history=discord_message_history or "",
        question=user_question,
        sql=sql or "",
        rows=str(rows or []),
    )

    final_output = await model.ainvoke(
        [SystemMessage(content=system_text), HumanMessage(content=prompt.to_string())]
    )
    # Ensure we return a string
    return str(final_output.content)


class AIConversationWorkflow:
    def __init__(self, ctx: AIConversationCustomContext):
        self.ctx = ctx
        
        # Create the tools using the factory function
        tools = _create_context_tools(ctx)
        
        # Select the appropriate model
        model = openai_model if ctx.provider == "openai" else google_model
        
        # Create the ReAct agent - it will handle tool calling automatically
        self.agent = create_react_agent(model, tools)

    async def run(self) -> str:
        """
        Run the agent workflow for the user question.
        Returns the final formatted Discord message.
        """
        ctx = self.ctx
        
        agent_system = SystemMessage(content=system_instruction_when_bot_mentioned)
        agent_human = HumanMessage(
            content=(
                "You are a Discord assistant. The user asked:\n\n"
                f"{ctx.user_question}\n\n"
                "Available tools:\n"
                "1. get_schema - Retrieve the database schema for the user's question\n"
                "2. query_database - Generate SQL and execute it against the database\n"
                "3. format_discord_message - Format the final response for Discord\n\n"
                "Follow this workflow:\n"
                "1. First, call get_schema to understand what tables are relevant\n"
                "2. Then, call query_database with the schema to get results\n"
                "3. Finally, call format_discord_message with the SQL results to produce the final response\n\n"
                "Do not include internal details like SQL, user id, or database internals in the final output."
            )
        )
        
        # Run the agent - it will automatically:
        # 1. Detect when tools should be called
        # 2. Execute the tools
        # 3. Pass results back to the model
        # 4. Continue until the model produces a final answer
        response = await self.agent.ainvoke(
            {"messages": [agent_system, agent_human]},
            config={"recursion_limit": MAX_AGENT_STEPS}
        )
        
        # Extract the final message content
        if response and "messages" in response:
            messages = response["messages"]
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    return str(last_message.content)
                else:
                    return str(last_message)
        
        return "Sorry, I couldn't complete your request."
