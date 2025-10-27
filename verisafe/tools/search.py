from graphcore.graph import WithToolCallId
from pydantic import Field
from typing import List, Annotated, cast, Literal, TypedDict, Protocol, ClassVar, Any
from langchain_core.tools import tool, InjectedToolCallId
from verisafe.core.context import CryptoContext
from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime
from verisafe.diagnostics.stream import ManualSearchResult
from verisafe.rag.db import PostgreSQLRAGDatabase
from dataclasses import Field as DField

class RAGDBContext(Protocol):
    __dataclass_fields__: ClassVar[dict[str, DField[Any]]]

    @property
    def rag_db(self) -> PostgreSQLRAGDatabase:
        ...

class SearchResultText(TypedDict):
    """
    Encoding of text search result from: https://docs.anthropic.com/en/api/messages#body-messages-content-content-content-text
    """
    type: Literal["text"]
    text: str

class SearchResultSchema(TypedDict):
    """
    Encoding of the search result tool result from: https://docs.anthropic.com/en/api/messages#body-messages-content-content-content
    """
    type: Literal["search_result"]
    title: str
    source: str
    content: List[SearchResultText]

class CVLManualSearchSchema(WithToolCallId):
    """
    Search the CVL manual database for information relevant to a question about CVL.

    This tool uses semantic similarity search to find the most relevant documentation
    sections from the CVL manual that can help answer questions about CVL syntax,
    semantics, and best practices.

    The result is a list of quotes from the manual, identified with the name of the relevant section.
    """
    question: str = Field(description="A single, self-contained question about CVL. Avoid open-ended 'how do I...?' questions in favor of 'What is the syntax for ...?' style questions.")
    similarity_cutoff: float = Field(default=0.5, description="Minimum cosine similarity threshold for results (default: 0.7)")
    max_results: int = Field(default=10, description="Maximum number of search results to return (default: 10)")
    manual_section: List[str] = \
        Field(default=[], description="A list of manual sections to search. "
              "If specified, at least one section heading must match at least one of the values provided here")


@tool(args_schema=CVLManualSearchSchema)
def cvl_manual_search(
    question: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    similarity_cutoff: float = 0.5,
    max_results: int = 10,
    manual_section: List[str] = []
) -> str | List[dict]:
    """Search the CVL manual database for relevant documentation."""
    runtime = get_runtime(RAGDBContext)
    writer = get_stream_writer()

    try:
        to_ret: List[SearchResultSchema] = []
        for t in runtime.context.rag_db.find_refs(query=question, similarity_cutoff=similarity_cutoff, top_k=max_results, manual_section=manual_section):
            upd : ManualSearchResult = {
                "type": "manual_search",
                "tool_id": tool_call_id,
                "ref": t
            }
            writer(upd)
            to_ret.append({
                "type": "search_result",
                "source": "CVL Manual",
                "title": " / ".join(t.headers),
                "content": [
                    {"type": "text", "text": t.content + f"\n (Similarity: {t.similarity})"}
                ]
            })
        return cast(List[dict], to_ret)
    except Exception as e:
        return f"Failed to search CVL manual: {str(e)}"
