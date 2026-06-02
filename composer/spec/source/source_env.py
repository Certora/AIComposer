from typing import Unpack
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langchain_core.language_models.chat_models import BaseChatModel

from graphcore.graph import Builder
from graphcore.tools.vfs import fs_tools

from composer.spec.tool_env import BaseSourceTools, BasicAgentTools
from composer.spec.services import build_rag_tool_env, RAGInputs
from composer.spec.service_host import ServiceHost, Sort
from composer.spec.code_explorer import indexed_code_explorer_tool
from composer.spec.agent_index import AgentIndex, AgentIndexConfig, RetrieveDocumentTool


@dataclass(frozen=True)
class _BaseSourceTools():
    base_source_tools: tuple[BaseTool, ...]


def build_basic_source_tools(
    root: str,
    forbidden_read: str,
) -> BaseSourceTools:
    return _BaseSourceTools(
        tuple(fs_tools(fs_layer=root, forbidden_read=forbidden_read, cache_listing=False))
    )


def build_source_tools(
    s: BaseSourceTools,
    llm: BasicAgentTools,
    store: BaseStore,
    cache_ns: tuple[str, ...],
    recursion_limit: int,
) -> tuple[BaseTool, ...]:
    """Wrap the base source tools with the indexed code_explorer sub-agent
    + the document-ref retrieval tool. Returns the full source tool tuple."""

    @dataclass(frozen=True)
    class _ExplorerEnv:
        builder: Builder[None, None, None]
        llm: BaseChatModel
        base_source_tools: tuple[BaseTool, ...]
        index: AgentIndex

    # Source-code agent caches are always per-user (no shared base);
    # the caller is responsible for passing a ``cache_ns`` that's
    # already user-scoped via ``user_data_ns(uid)``. The index runs
    # single-pool: no overlay, base_layer == cache_ns.
    ind = AgentIndex(
        store=store,
        config=AgentIndexConfig(base_layer=cache_ns),
    )

    explorer_tool = indexed_code_explorer_tool(
        _ExplorerEnv(
            builder=llm.builder,
            base_source_tools=s.base_source_tools,
            index=ind,
            llm=llm.llm,
        ),
        recursion_limit=recursion_limit,
    )
    return s.base_source_tools + (
        explorer_tool,
        RetrieveDocumentTool.bind(ind).as_tool("code_document_ref"),
    )


class SourceParams(RAGInputs):
    root: str
    forbidden_read: str
    source_question_ns: tuple[str, ...]


def build_source_env(
    *,
    sort: Sort = "existing",
    **params: Unpack[SourceParams],
) -> ServiceHost:
    """Build a fully-bound ``ServiceHost`` with both RAG and source tool
    suites. Source-mode autoprover defaults to ``sort="existing"``."""
    rag_env = build_rag_tool_env(sort=sort, **params)

    basic_source = build_basic_source_tools(
        root=params["root"],
        forbidden_read=params["forbidden_read"],
    )
    full_source = build_source_tools(
        basic_source,
        rag_env,
        params["store"],
        params["source_question_ns"],
        recursion_limit=params["recursion_limit"],
    )
    return ServiceHost(
        llm=rag_env.llm,
        builder=rag_env.builder,
        rag_tools=rag_env.rag_tools,
        source_tools=full_source,
        sort=rag_env.sort,
    )
