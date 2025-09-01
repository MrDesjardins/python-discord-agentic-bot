from pydantic import BaseModel


class SQLQuery(BaseModel):
    tool: str  # One of: stats_database, tournament_database, activity_database
    query: str
