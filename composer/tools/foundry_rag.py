from typing import Iterable
from langchain_core.tools import BaseTool
from pydantic import Field

from composer.rag.db import ComposerRAGDB

from graphcore.tools.schemas import WithAsyncDependencies

def _header_string(s: list[str]) -> str:
    return " > ".join(i for i in s if i)

class FoundryKeywordSearch(WithAsyncDependencies[str, ComposerRAGDB]):
    """
    Search for foundry cheatcodes using full text search.

    Returns the section titles that match in order of relevance. Use `get_foundry_cheatcode` to access
    the cheatcode documentation.
    """
    query: str = Field(description=(
        "A websearch-style query string. Unquoted terms are combined with AND. "
        "Use 'OR' between terms for alternatives, quotes for exact phrases, "
        "and '-' to exclude terms. Example: '\"mock calls\" OR prank -logs'"
    ))
    limit: int = Field(default=10, description="Maximum number of results to return.")

    async def run(self) -> str:
        with self.tool_deps() as db:
            res = await db.search_manual_keywords(
                self.query, limit=self.limit
            )
            to_ret = []
            for r in res:
                to_ret.append(f"{_header_string(r.headers)} [relevance: {r.relevance:.4f}]")
            if not to_ret:
                return "No results found"
            return "\n".join(to_ret)

class FoundryVectorSearch(WithAsyncDependencies[str, ComposerRAGDB]):
    """
    Search the foundry cheatcode manual for sections which match a natural language
    question. Returns the manual section title, the relevant text, and its relevance score.
    """
    query: str = Field(description=(
        "A single, natural language question to use to search for foundry cheatcodes. For example, "
        "'how do I mock an external call?' or 'what is the API of prank?'"
    ))
    similarity_cutoff: float = Field(default=0.5, description="Minimum cosine similarity threshold for results (default: 0.7)")
    max_results: int = Field(default=10, description="Maximum number of search results to return (default: 10)")

    async def run(self) -> str:
        with self.tool_deps() as db:
            res = await db.find_refs(
                self.query, similarity_cutoff=self.similarity_cutoff, top_k=self.max_results
            )

            to_ret = []
            for r in res:
                to_ret.append(f"----\nSection: {_header_string(r.headers)}\n\n{r.content}\nSimilarity: {r.similarity:.4f}")
            if not to_ret:
                return "(No results found)"
            return "\n".join(to_ret)

class FoundrySectionGet(WithAsyncDependencies[str, ComposerRAGDB]):
    """
    Retrieve the contents of a section of the foundry cheatcodes manual by name.
    """
    section_names: list[str] = Field(description=(
        "A list of section headings identifying the portion of the cheatcodes manual to read. "
        "By convention, the `foundry_cheatcodes_keyword_search` and `foundry_cheatcodes_manual_search` separates "
        "section headers with `>`. For example, to retrieve the 'Cheatcodes > prank > Examples' section, pass ['Cheatcodes', 'prank', 'Examples']."
    ))

    async def run(self) -> str:
        with self.tool_deps() as db:
            content = await db.get_manual_section(self.section_names)
            if content is None:
                return f"No section found for {' > '.join(self.section_names)!r}"
            return content


def get_tools(
    db: ComposerRAGDB
) -> Iterable[BaseTool]:
    return [
        FoundrySectionGet.bind(db).as_tool("foundry_cheatcodes_get_section"),
        FoundryVectorSearch.bind(db).as_tool("foundry_cheatcodes_manual_search"),
        FoundryKeywordSearch.bind(db).as_tool("foundry_cheatcodes_keyword_search"),
    ]
