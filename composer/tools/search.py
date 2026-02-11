from graphcore.graph import WithToolCallId
from pydantic import BaseModel, Field
from typing import Annotated, cast, Literal, TypedDict, Protocol, ClassVar, Any, Callable, overload

from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langgraph.config import get_stream_writer
from langgraph.runtime import get_runtime
from composer.diagnostics.stream import ManualSearchResult
from composer.rag.db import PostgreSQLRAGDatabase
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
    content: list[SearchResultText]

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
    manual_section: list[str] = \
        Field(default=[], description="A list of manual sections to search. "
              "If specified, at least one section heading must match at least one of the values provided here")

def _cvl_manual_search_factory(
    db_provider: Callable[[], PostgreSQLRAGDatabase]
) -> BaseTool:
    @tool("cvl_manual_search", args_schema=CVLManualSearchSchema)
    def _cvl_manual_search(
        question: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        similarity_cutoff: float = 0.5,
        max_results: int = 10,
        manual_section: list[str] = []
    ) -> str | list[dict]:
        """Search the CVL manual database for relevant documentation."""
        rag_db = db_provider()
        writer = get_stream_writer()

        try:
            to_ret: list[SearchResultSchema] = []
            for t in rag_db.find_refs(query=question, similarity_cutoff=similarity_cutoff, top_k=max_results, manual_section=manual_section):
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
            return cast(list[dict], to_ret)
        except Exception as e:
            return f"Failed to search CVL manual: {str(e)}"
    return _cvl_manual_search

class CVLKeywordSearchSchema(BaseModel):
    """
    Search the CVL manual for sections matching keywords using full-text search.

    Returns the headers of matching sections ranked by relevance. Use get_cvl_manual_section
    to retrieve the full content of a section returned by this tool.
    """
    query: str = Field(description=(
        "A websearch-style query string. Unquoted terms are combined with AND. "
        "Use 'OR' between terms for alternatives, quotes for exact phrases, "
        "and '-' to exclude terms. Example: '\"ghost variable\" OR storage -mapping'"
    ))
    min_depth: int = Field(default=0, description="Minimum section depth (0-6). Only return sections where at least this many header levels (h1..hN) are present. 0 means no filtering.")
    limit: int = Field(default=10, description="Maximum number of results to return.")

class CVLGetSectionSchema(BaseModel):
    """
    Retrieve the full content of a CVL manual section by its exact headers.

    Use cvl_keyword_search to discover section headers first, then use this tool
    to fetch the complete text of a specific section.
    """
    headers: list[str] = Field(description="The section header path, e.g. ['Types', 'Integer Types']. Must match exactly.")

def _cvl_keyword_search_factory(
    db_provider: Callable[[], PostgreSQLRAGDatabase]
) -> BaseTool:
    @tool("cvl_keyword_search", args_schema=CVLKeywordSearchSchema)
    def _cvl_keyword_search(
        query: str,
        min_depth: int = 0,
        limit: int = 10,
    ) -> str:
        """Search the CVL manual for sections matching keywords."""
        rag_db = db_provider()
        try:
            hits = rag_db.search_manual_keywords(query, min_depth=min_depth, limit=limit)
            if not hits:
                return "No matching sections found."
            lines = []
            for h in hits:
                section_path = " > ".join(h.headers)
                lines.append(f"[{h.relevance:.4f}] {section_path}")
            return "\n".join(lines)
        except Exception as e:
            return f"Failed to search CVL manual: {str(e)}"
    return _cvl_keyword_search

def _cvl_get_section_factory(
    db_provider: Callable[[], PostgreSQLRAGDatabase]
) -> BaseTool:
    @tool("get_cvl_manual_section", args_schema=CVLGetSectionSchema)
    def _get_cvl_manual_section(
        headers: list[str],
    ) -> str:
        """Retrieve the full content of a CVL manual section by its headers."""
        rag_db = db_provider()
        try:
            content = rag_db.get_manual_section(headers)
            if content is None:
                return f"No section found matching headers: {headers}"
            return content
        except Exception as e:
            return f"Failed to retrieve section: {str(e)}"
    return _get_cvl_manual_section

def _get_provider(ctxt: "type[RAGDBContext] | PostgreSQLRAGDatabase") -> Callable[[], PostgreSQLRAGDatabase]:
    if isinstance(ctxt, PostgreSQLRAGDatabase):
        return lambda: ctxt
    else:
        return lambda: get_runtime(ctxt).context.rag_db

@overload
def cvl_manual_search(ctxt: PostgreSQLRAGDatabase) -> BaseTool:
    ...

@overload
def cvl_manual_search(ctxt: type[RAGDBContext]) -> BaseTool:
    ...

def cvl_manual_search(ctxt: type[RAGDBContext] | PostgreSQLRAGDatabase) -> BaseTool:
    return _cvl_manual_search_factory(_get_provider(ctxt))

@overload
def cvl_manual_tools(ctxt: PostgreSQLRAGDatabase) -> list[BaseTool]:
    ...

@overload
def cvl_manual_tools(ctxt: type[RAGDBContext]) -> list[BaseTool]:
    ...

def cvl_manual_tools(ctxt: type[RAGDBContext] | PostgreSQLRAGDatabase) -> list[BaseTool]:
    provider = _get_provider(ctxt)
    return [
        _cvl_manual_search_factory(provider),
        _cvl_keyword_search_factory(provider),
        _cvl_get_section_factory(provider),
    ]
