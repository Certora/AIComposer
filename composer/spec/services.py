from typing import TypedDict, Unpack, Protocol
from dataclasses import dataclass
from composer.rag.db import PostgreSQLRAGDatabase
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from composer.templates.loader import load_jinja_template
from graphcore.graph import Builder
from composer.spec.tool_env import (
    RAGTools, BaseRAGTools, BasicAgentTools, ToolEnvironment
)
from composer.spec.cvl_research import cvl_research_tool, CVL_RESEARCH_BASE_DOC
from composer.tools.search import cvl_manual_tools
from composer.kb.knowledge_base import kb_tools

@dataclass(frozen=True)
class _BasicLLM:
    _llm: BaseChatModel
    has_source: bool
    _checkpointer: Checkpointer

    @property
    def builder(self) -> Builder[None, None, None]:
        return Builder[None, None, None]().with_llm(
            self._llm
        ).with_loader(
            load_jinja_template
        ).with_checkpointer(self._checkpointer)

@dataclass(frozen=True)
class _BaseTools:
    builder: Builder[None, None, None]
    has_source: bool


@dataclass(frozen=True)
class _BaseRAGTools():
    base_rag_tools: tuple[BaseTool,...]

def build_rag_tools(
    s: BaseRAGTools,
    llm: BasicAgentTools
) -> RAGTools:
    
    @dataclass(frozen=True)
    class _CVLResearchEnv(_BaseTools):
        base_rag_tools: tuple[BaseTool, ...]
    
    @dataclass(frozen=True)
    class _RAGTools:
        rag_tools: tuple[BaseTool, ...]


    cvl_researcher = cvl_research_tool(
        _CVLResearchEnv(
            builder=llm.builder,
            has_source=llm.has_source,
            base_rag_tools=s.base_rag_tools
        ),
        CVL_RESEARCH_BASE_DOC
    )
    return _RAGTools(s.base_rag_tools + (cvl_researcher,))

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

class RagToolEnv(BasicAgentTools, RAGTools, BaseRAGTools, Protocol):
    pass

def build_rag_tool_env(
    **params: Unpack[RAGInputs],
) -> RagToolEnv:
    llm = _BasicLLM(
        _llm=params["llm"],
        has_source=False,
        _checkpointer=params["checkpoint"]
    )
    rag_tools = build_basic_rag_tools(
        db=params["db"],
        kb_ns=params["kb_ns"],
        store=params["store"]
    )

    full_rag_tools = build_rag_tools(
        llm=llm,
        s=rag_tools
    )

    @dataclass(frozen=True)
    class ToRet(_BaseRAGTools, _BaseTools):
        rag_tools: tuple[BaseTool, ...]

    return ToRet(
        builder=llm.builder,
        has_source=llm.has_source,
        base_rag_tools=rag_tools.base_rag_tools,
        rag_tools=full_rag_tools.rag_tools
    )


def build_natspec_env(
    **params: Unpack[RAGInputs]
) -> ToolEnvironment:
    common_rag = build_rag_tool_env(
        **params
    )

    class NatspecEnv:
        @property
        def builder(self) -> Builder[None, None, None]:
            return common_rag.builder

        @property
        def cvl_authorship_tools(self) -> tuple[BaseTool, ...]:
            return self.rag_tools

        @property
        def feedback_tools(self) -> tuple[BaseTool, ...]:
            return self.cvl_authorship_tools
        
        @property
        def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
            return tuple()
        
        @property
        def rag_tools(self) -> tuple[BaseTool, ...]:
            return common_rag.rag_tools

        @property
        def has_source(self) -> bool:
            return False
        
        @property
        def system_analysis_tools(self) -> tuple[BaseTool, ...]:
            return tuple()

    return NatspecEnv()