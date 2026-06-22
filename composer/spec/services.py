from typing import TypedDict, Unpack
from dataclasses import dataclass
from composer.rag.db import PostgreSQLRAGDatabase
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from graphcore.graph import Builder
from composer.spec.tool_env import BaseRAGTools
from composer.spec.service_host import ModelProvider, PureServiceHost, Sort
from composer.spec.cvl_research import indexed_cvl_research_tool, CVL_RESEARCH_BASE_DOC
from composer.tools.search import cvl_manual_tools
from composer.kb.knowledge_base import kb_tools
from composer.spec.agent_index import AgentIndex, AgentIndexConfig, RetrieveDocumentTool


@dataclass(frozen=True)
class _BaseRAGTools():
    base_rag_tools: tuple[BaseTool, ...]


def build_rag_tools(
    s: BaseRAGTools,
    models: ModelProvider,
    store: BaseStore,
    recursion_limit: int,
    index_config: AgentIndexConfig,
) -> tuple[BaseTool, ...]:
    """Wrap the base RAG tools with the indexed cvl_researcher sub-agent
    + the document-ref retrieval tool. Returns the full RAG tool tuple.

    The cvl_researcher is a support sub-agent, so it runs on the lite tier."""
    ind = AgentIndex(store=store, config=index_config)

    @dataclass(frozen=True)
    class _CVLResearchEnv:
        builder: Builder[None, None, None]
        base_rag_tools: tuple[BaseTool, ...]
        agent_index: AgentIndex

    cvl_researcher = indexed_cvl_research_tool(
        _CVLResearchEnv(
            builder=models.builder_lite(),
            base_rag_tools=s.base_rag_tools,
            agent_index=ind,
        ),
        CVL_RESEARCH_BASE_DOC,
        recursion_limit=recursion_limit,
    )
    return s.base_rag_tools + (
        cvl_researcher,
        RetrieveDocumentTool.bind(ind).as_tool("cvl_document_ref"),
    )


def build_basic_rag_tools(
    db: PostgreSQLRAGDatabase,
    store: BaseStore,
    kb_ns: tuple[str, ...]
) -> BaseRAGTools:
    return _BaseRAGTools(
        tuple(cvl_manual_tools(db)) + tuple(kb_tools(
            store, kb_ns, read_only=True
        ))
    )


class LLMInputs(TypedDict):
    models: ModelProvider


class RAGInputs(LLMInputs):
    db: PostgreSQLRAGDatabase
    store: BaseStore
    kb_ns: tuple[str, ...]
    cvl_index_config: AgentIndexConfig
    recursion_limit: int


def build_rag_tool_env(
    *,
    sort: Sort = "greenfield",
    **params: Unpack[RAGInputs],
) -> PureServiceHost:
    """Build a source-less ``PureServiceHost`` carrying the RAG tool
    suite. The natspec greenfield path uses this directly; the source
    path layers source tools on top via :func:`build_source_env`."""
    base_rag = build_basic_rag_tools(
        db=params["db"],
        kb_ns=params["kb_ns"],
        store=params["store"],
    )
    full_rag = build_rag_tools(
        models=params["models"],
        s=base_rag,
        store=params["store"],
        index_config=params["cvl_index_config"],
        recursion_limit=params["recursion_limit"],
    )
    return PureServiceHost(
        models=params["models"],
        rag_tools=full_rag,
        sort=sort,
    )
