from pydantic import BaseModel
from langchain_core.tools import BaseTool

def result_tool_generator(
    outkey: str,
    result_schema: type[BaseModel] | tuple[type, str],
    doc: str,
) -> BaseTool:
    ...
