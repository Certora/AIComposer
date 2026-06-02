from typing import TypedDict, Unpack
from dataclasses import dataclass
from composer.rag.db import PostgreSQLRAGDatabase
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from composer.templates.loader import load_jinja_template
from graphcore.graph import Builder
from composer.spec.tool_env import BaseRAGTools, BasicAgentTools
from composer.spec.service_host import PureServiceHost, Sort
from composer.spec.cvl_research import indexed_cvl_research_tool, CVL_RESEARCH_BASE_DOC
from composer.tools.search import cvl_manual_tools
from composer.kb.knowledge_base import kb_tools
from composer.spec.agent_index import AgentIndex, AgentIndexConfig, RetrieveDocumentTool


@dataclass(frozen=True)
class _BasicLLM:
    """Minimal ``BasicAgentTools`` satisfier — ``llm`` + ``builder``.

    Internal to the env builders here. ``sort`` lives on ``ServiceHost``,
    not on this struct: the structural interfaces for sub-agent tool
    implementations (``CVLResearchEnv``, ``CodeExplorerEnv``) don't need
    the workflow ``sort`` to do their work, and dragging it through their
    construction made the call sites lie about what they were configuring.
    """
    llm: BaseChatModel
    _checkpointer: Checkpointer

    @property
    def builder(self) -> Builder[None, None, None]:
        return Builder[None, None, None]().with_llm(
            self.llm
        ).with_loader(
            load_jinja_template
        ).with_checkpointer(self._checkpointer)


@dataclass(frozen=True)
class _BaseRAGTools():
    base_rag_tools: tuple[BaseTool, ...]


def build_rag_tools(
    s: BaseRAGTools,
    llm: BasicAgentTools,
    store: BaseStore,
    recursion_limit: int,
    index_config: AgentIndexConfig,
) -> tuple[BaseTool, ...]:
    """Wrap the base RAG tools with the indexed cvl_researcher sub-agent
    + the document-ref retrieval tool. Returns the full RAG tool tuple."""
    ind = AgentIndex(store=store, config=index_config)

    @dataclass(frozen=True)
    class _CVLResearchEnv:
        builder: Builder[None, None, None]
        llm: BaseChatModel
        base_rag_tools: tuple[BaseTool, ...]
        agent_index: AgentIndex

    cvl_researcher = indexed_cvl_research_tool(
        _CVLResearchEnv(
            builder=llm.builder,
            base_rag_tools=s.base_rag_tools,
            agent_index=ind,
            llm=llm.llm,
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
    llm: BaseChatModel
    checkpoint: Checkpointer


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
    llm = _BasicLLM(
        llm=params["llm"],
        _checkpointer=params["checkpoint"],
    )
    base_rag = build_basic_rag_tools(
        db=params["db"],
        kb_ns=params["kb_ns"],
        store=params["store"],
    )
    full_rag = build_rag_tools(
        llm=llm,
        s=base_rag,
        store=params["store"],
        index_config=params["cvl_index_config"],
        recursion_limit=params["recursion_limit"],
    )
    return PureServiceHost(
        llm=llm.llm,
        builder=llm.builder,
        rag_tools=full_rag,
        sort=sort,
    )
