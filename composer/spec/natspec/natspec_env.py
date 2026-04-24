"""
Natspec-specific tool environment.

Mirrors ``composer/spec/source/source_env.py``: owns the pipeline-level umbrella
``NatspecEnvironment`` Protocol (which extends the base ``ToolEnvironment`` with
the role-scoped tool sets the natspec pipeline needs) and the ``build_natspec_env``
factory that materializes it in either the no-source or source-aware flavor.

The individual role Protocols (``MergeEnv`` in merge.py, ``FeedbackEnv`` in
feedback.py, ``BugEnvironment`` in bug.py, ``AnalysisEnv`` in system_analysis.py,
``GenerationEnv`` in author.py) live alongside the agents that consume them —
this module just composes them into the pipeline-level umbrella.
"""

from dataclasses import dataclass
from typing import Protocol, Unpack

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from graphcore.graph import Builder

from composer.spec._env_common import RAGInputs, build_rag_tool_env
from composer.spec.source.source_env import SourceOnlyParams, build_source_env
from composer.spec.tool_env import ToolEnvironment


class NatspecEnvironment(ToolEnvironment, Protocol):
    """Pipeline-level umbrella for the natspec workflow.

    Declares every role-scoped property natspec sub-agents need, so a single
    ``NatspecEnvironment`` instance can be passed to the pipeline and flow
    down to each agent (merge, feedback, bug, analysis, generation) whose
    local Protocol requires a subset.
    """

    @property
    def merge_tools(self) -> tuple[BaseTool, ...]:
        ...

    @property
    def interface_gen_tools(self) -> tuple[BaseTool, ...]:
        ...

@dataclass(frozen=True)
class _NatspecEnvImpl:
    """Concrete ``NatspecEnvironment``. Built once per pipeline run — each
    role-scoped tuple is precomputed from the underlying source or RAG env.
    ``merge_tools`` is natspec-specific and composed here, not in
    ``SourceEnvironment``.
    """
    llm: BaseChatModel
    builder: Builder[None, None, None]
    rag_tools: tuple[BaseTool, ...]
    cvl_authorship_tools: tuple[BaseTool, ...]
    feedback_tools: tuple[BaseTool, ...]
    merge_tools: tuple[BaseTool, ...]
    interface_gen_tools: tuple[BaseTool, ...]
    bug_analysis_tools: tuple[BaseTool, ...]
    system_analysis_tools: tuple[BaseTool, ...]


def build_natspec_env(
    *,
    source: SourceOnlyParams | None = None,
    **params: Unpack[RAGInputs],
) -> NatspecEnvironment:
    """Build the tool environment for the natspec pipeline.

    If ``source`` is supplied, layers the source-aware VFS tools over the RAG
    env and composes natspec-specific ``merge_tools`` from source + RAG.
    Otherwise returns a RAG-only greenfield env where every role tool set
    collapses to RAG (or the empty tuple for bug/system analysis, which only
    make sense when source is available).
    """
    if source is not None:
        src = build_source_env(**source, **params)
        return _NatspecEnvImpl(
            llm=src.llm,
            builder=src.builder,
            rag_tools=src.rag_tools,
            cvl_authorship_tools=src.cvl_authorship_tools,
            feedback_tools=src.feedback_tools,
            merge_tools=src.source_tools + src.rag_tools,
            interface_gen_tools=src.source_tools,
            bug_analysis_tools=src.bug_analysis_tools,
            system_analysis_tools=src.system_analysis_tools,
        )

    rag = build_rag_tool_env(**params)
    return _NatspecEnvImpl(
        llm=rag.llm,
        builder=rag.builder,
        rag_tools=rag.rag_tools,
        cvl_authorship_tools=rag.rag_tools,
        feedback_tools=rag.rag_tools,
        merge_tools=rag.rag_tools,
        interface_gen_tools=(),
        bug_analysis_tools=(),
        system_analysis_tools=(),
    )
