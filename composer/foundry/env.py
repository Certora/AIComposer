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


from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer


from composer.rag.db import ComposerRAGDB
from composer.spec.source.source_env import (
    build_basic_source_tools, build_source_tools,
)
from composer.spec.service_host import ModelProvider
from composer.spec.service_host import ServiceHost, PureServiceHost
from composer.tools.foundry_rag import get_tools as foundry_cheatcode_tools


def build_foundry_env(
    *,
    model_provider: ModelProvider,
    project_root: str,
    forbidden_read: str,
    rag_db: ComposerRAGDB,
    store: BaseStore,
    source_question_ns: tuple[str, ...],
    recursion_limit: int,
) -> ServiceHost:
    """Construct a foundry-workflow env.

    ``rag_db`` is the foundry cheatcodes RAG database (distinct from the
    CVL manual DB — they live in different postgres databases per the
    rag-build separation).

    ``store`` + ``source_question_ns`` are needed by the indexed
    ``code_explorer`` sub-agent for its per-user query cache (same wiring
    autoprove uses; see ``build_source_env``).
    """

    basic_source = build_basic_source_tools(
        root=project_root,
        forbidden_read=forbidden_read,
    )
    full_source = build_source_tools(
        basic_source,
        model_provider,
        store,
        source_question_ns,
        recursion_limit=recursion_limit,
    )

    rag = tuple(foundry_cheatcode_tools(rag_db))

    return PureServiceHost(
        models=model_provider,
        rag_tools=rag,
        sort="existing"
    ).bind_source_tools(
        full_source
    )
