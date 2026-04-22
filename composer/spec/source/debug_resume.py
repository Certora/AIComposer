"""Resume a batch_cvl_generation from a stored snapshot.

Usage:
    python -m composer.spec.source.debug_resume <mnemonic> [--cloud]
"""

import composer.certora as _  # noqa: F401 — side-effect import

import argparse
import asyncio

from typing import cast

from graphcore.tools.memory import async_memory_tool

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import create_llm, standard_connections

from composer.spec.context import WorkflowContext, CVLGeneration, WorkflowServices
from composer.spec.source.snapshot import load_snapshot, CVLGenSnapshot
from composer.spec.source.prover import get_prover_tool, CloudConfig
from composer.spec.source.source_env import build_source_env, SourceEnvironment
from composer.spec.source.author import batch_cvl_generation
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.util import string_hash

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.store.base import BaseStore


async def resume_from_snapshot(
    snapshot: CVLGenSnapshot,
    *,
    llm: BaseChatModel,
    env: SourceEnvironment,
    services: WorkflowServices,
    store: BaseStore,
    cloud: CloudConfig | None = None,
) -> GeneratedCVL:
    """Reconstruct inputs from a snapshot and re-enter batch_cvl_generation."""
    source = snapshot.source.restore()
    component = snapshot.component.restore() if snapshot.component else None

    ctx = WorkflowContext.create(
        services=services,
        thread_id=snapshot.thread_id,
        store=store,
        memory_namespace=snapshot.memory_namespace,
        cache_namespace=snapshot.cache_namespace,
    )

    prover_tool = get_prover_tool(
        llm, source.contract_name, source.project_root, cloud=cloud,
    )

    return await batch_cvl_generation(
        ctx=ctx.abstract(CVLGeneration),
        init_config=snapshot.init_config,
        props=snapshot.props,
        component=component,
        resources=snapshot.resources,
        prover_tool=prover_tool,
        env=env,
        description=snapshot.description,
        source=source,
    )


class _ResumeArgs(ModelOptions, RAGDBOptions):
    mnemonic: str
    cloud: bool


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume CVL generation from a snapshot mnemonic"
    )
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument("mnemonic", help="Snapshot mnemonic ID (e.g. proven-lattice-forging-deeply)")
    parser.add_argument("--cloud", action="store_true", help="Run prover in the cloud")

    args = cast(_ResumeArgs, parser.parse_args())

    llm = create_llm(args)
    model = get_model()

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conns,
        PostgreSQLRAGDatabase.rag_context(model) as rag_db,
    ):
        snapshot = await load_snapshot(conns.store, args.mnemonic)
        source = snapshot.source.restore()

        root_key = string_hash(f"{source.project_root}|{source.contract_name}")

        env = build_source_env(
            llm=llm,
            db=rag_db,
            checkpoint=conns.checkpointer,
            forbidden_read=source.forbidden_read,
            kb_ns=("cvl",),
            root=source.project_root,
            store=conns.indexed_store,
            cvl_cache_ns=DEFAULT_CVL_AGENT_INDEX_NS,
            source_question_ns=("source_agent", "cache", root_key),
        )

        result = await resume_from_snapshot(
            snapshot,
            llm=llm,
            env=env,
            services=lambda ns: async_memory_tool(conns.memory(ns)),
            store=conns.store,
            cloud=CloudConfig() if args.cloud else None,
        )

        print(f"--- Generated CVL ---\n{result.cvl}")
        print(f"\n--- Commentary ---\n{result.commentary}")
        if result.skipped:
            print(f"\n--- Skipped ---")
            for s in result.skipped:
                print(f"  #{s.property_index}: {s.reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
