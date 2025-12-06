"""
Contains all system instruction that can be reused in different workflow
"""

system_instruction_when_bot_mentioned = (
    "You are a bot that is mentioned in a Discord server. You must answer the user. "
    "Do not mention your name or internal purpose. Keep responses concise. "
)


tool_get_schema_description =  """
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
tool_get_all_schema_tool_description =  """
        Select and return all the database schemas

        🛠 Purpose:
        - Ensures the LLM sees all the schema in case the situational tool did not work out.

        📥 Input:
        - None

        📤 Output:
        - A string containing the concatenated schema text for the relevant tables.
        - Includes user, stats, tournament, and activity tables as needed.

        🧠 When to use:
        - Call this tool **before generating SQL queries** only when the get_schema did not provide sufficient information.
        - Use this tool whenever the LLM needs structured table information
            to generate valid queries.

        🚫 Notes:
        - Returns plain text; the LLM should not attempt to interpret or modify the schema.
        """
tool_sql_query_description =  """
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
        
tool_discord_format_message_description =  """
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