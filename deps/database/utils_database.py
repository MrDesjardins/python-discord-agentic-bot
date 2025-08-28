from deps.database.system_database import DBName, DatabaseManager


def get_table_schema(table_name: str) -> str:
    schema_info = []
    with DatabaseManager.get_database_manager() as db:
        cursor = db.get_cursor(DBName.SIEGE)
        cursor.execute(f"PRAGMA table_info({table_name});")
        cols = cursor.fetchall()
        col_defs = ", ".join([f"{c[1]} {c[2]}" for c in cols])
        schema_info.append(f"{table_name}({col_defs})")
        return "\n".join(schema_info)
