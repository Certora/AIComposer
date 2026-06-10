"""Env construction for the foundry test author.

Builds an env that swaps the CVL-specific RAG surface (``cvl_research``,
``cvl_manual_*``, ``scan_knowledge_base``, etc.) for the foundry cheatcode
RAG tools, but otherwise reuses the same source-tools machinery the
autoprove workflow uses — including the indexed ``code_explorer``
sub-agent — so the analysis and authoring agents can navigate and ask
questions about the existing solidity project.

The resulting env satisfies four protocols the foundry workflow consumes:

* ``BasicAgentTools`` — ``builder`` / ``llm`` / ``has_source``.
* ``RAGTools`` — populated with foundry cheatcode tools.
* ``SourceTools`` — base ``fs_tools`` plus the indexed ``code_explorer``
  + ``code_document_ref``. Exposed both directly (so the author can
  bind them) and as ``system_analysis_tools`` / ``bug_analysis_tools``
  (so the existing ``run_component_analysis`` and ``run_property_inference``
  pick them up via their protocol fields).
"""

from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from graphcore.graph import Builder

from composer.rag.db import ComposerRAGDB
from composer.spec.services import _BasicLLM
from composer.spec.source.source_env import (
    build_basic_source_tools, build_source_tools,
)
from composer.spec.tool_env import BasicAgentTools, RAGTools, SourceTools
from composer.tools.foundry_rag import get_tools as foundry_cheatcode_tools


class FoundryEnv(BasicAgentTools, RAGTools, SourceTools, Protocol):
    """The shape the foundry pipeline expects: builder + foundry RAG +
    source tools (fs + code explorer). ``system_analysis_tools`` and
    ``bug_analysis_tools`` are aliases for ``source_tools`` so the
    existing ``run_component_analysis`` / ``run_property_inference``
    helpers (which look those names up via their own protocols) find
    them."""

    @property
    def system_analysis_tools(self) -> tuple[BaseTool, ...]: ...

    @property
    def bug_analysis_tools(self) -> tuple[BaseTool, ...]: ...


def build_foundry_env(
    *,
    llm: BaseChatModel,
    checkpoint: Checkpointer,
    project_root: str,
    forbidden_read: str,
    rag_db: ComposerRAGDB,
    store: BaseStore,
    source_question_ns: tuple[str, ...],
    recursion_limit: int,
) -> FoundryEnv:
    """Construct a foundry-workflow env.

    ``rag_db`` is the foundry cheatcodes RAG database (distinct from the
    CVL manual DB — they live in different postgres databases per the
    rag-build separation).

    ``store`` + ``source_question_ns`` are needed by the indexed
    ``code_explorer`` sub-agent for its per-user query cache (same wiring
    autoprove uses; see ``build_source_env``).
    """
    base_llm = _BasicLLM(llm=llm, has_source=True, _checkpointer=checkpoint)

    basic_source = build_basic_source_tools(
        root=project_root,
        forbidden_read=forbidden_read,
    )
    full_source = build_source_tools(
        basic_source,
        base_llm,
        store,
        source_question_ns,
        recursion_limit=recursion_limit,
    )

    rag = tuple(foundry_cheatcode_tools(rag_db))

    @dataclass(frozen=True)
    class _Env:
        builder: Builder[None, None, None]
        llm: BaseChatModel
        has_source: bool
        rag_tools: tuple[BaseTool, ...]
        source_tools: tuple[BaseTool, ...]

        @property
        def system_analysis_tools(self) -> tuple[BaseTool, ...]:
            return self.source_tools

        @property
        def bug_analysis_tools(self) -> tuple[BaseTool, ...]:
            return self.source_tools

    return _Env(
        builder=base_llm.builder,
        llm=base_llm.llm,
        has_source=True,
        rag_tools=rag,
        source_tools=full_source.source_tools,
    )
