from deps.database.system_database import DBName, DatabaseManager
import sqlite3


def get_table_schema(table_name: str) -> str:
    """
    Get schema for a specific table using PRAGMA table_info.
    Returns formatted schema string or empty string if table doesn't exist.
    """
    try:
        schema_info = []
        with DatabaseManager.get_database_manager() as db:
            cursor = db.get_cursor(DBName.SIEGE)
            cursor.execute(f"PRAGMA table_info({table_name});")
            cols = cursor.fetchall()
            if not cols:
                # Table doesn't exist or has no columns
                return f"-- Table '{table_name}' not found or is empty\n"
            col_defs = ", ".join([f"{c[1]} {c[2]}" for c in cols])
            schema_info.append(f"{table_name}({col_defs})")
            return "\n".join(schema_info)
    except sqlite3.DatabaseError as e:
        return f"-- Error retrieving schema for '{table_name}': {e}\n"
