from pydantic import BaseModel, Field

class SQLQuery(BaseModel):
    tool: str = Field(description="The tool to use, always 'database_tool'")
    query: str = Field(description="The SQL query to execute")
