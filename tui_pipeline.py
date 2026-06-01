"""Entry point for the NatSpec multi-agent pipeline TUI."""

import composer.certora as _

import argparse
import asyncio
import pathlib
import uuid
from typing import cast, Protocol

from composer.input.types import DEFAULT_RECURSION_LIMIT, ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import standard_connections
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.spec.services import build_natspec_env

from composer.spec.context import (
    WorkflowContext, SystemDoc,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline
from composer.spec.util import string_hash
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.spec.agent_index import agent_index_config_from_env

from composer.ui.pipeline_app import PipelineApp


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

class PipelineArgs(ModelOptions, RAGDBOptions, Protocol):
    input_file: str
    contract_name: str
    solc_version: str
    max_concurrent: int
    cache_ns: str | None
    memory_ns: str | None
    recursion_limit: int
    max_bug_rounds: int


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="NatSpec multi-agent pipeline TUI"
    )
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT, help=f"The number of iterations of the graph to allow (default: {DEFAULT_RECURSION_LIMIT})")
    parser.add_argument("input_file", help="Path to the design document (text or PDF)")
    parser.add_argument("--solc-version", default="8.29", help="Solidity compiler version (default: 8.29)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument("--max-bug-rounds", type=int, default=3, help="Maximum number of bug-extraction rounds run per component during property analysis (default: 3)")

    args = cast(PipelineArgs, parser.parse_args())

    # Read input document (handles both text and PDF)
    input_path = pathlib.Path(args.input_file)

    model = get_model()

    async with (
        standard_connections(args=args, embedder=DefaultEmbedder(model)) as conn,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag
    ):
        content = await conn.uploader.get_document(input_path)
        if content is None:
            print(f"Error: cannot read {input_path}")
            return 1
        system_doc = SystemDoc(content=content)

        env = build_natspec_env(
            llm=conn.llm,
            checkpoint=conn.checkpointer,
            db=rag,
            cvl_index_config=agent_index_config_from_env(DEFAULT_CVL_AGENT_INDEX_NS),
            kb_ns=DEFAULT_KB_NS,
            store=conn.indexed_store,
            recursion_limit=args.recursion_limit,
            provider=conn.provider,
        )

        cache_root = (args.cache_ns, system_doc.content.to_digest()) if args.cache_ns else None

        thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        ctx = WorkflowContext.create(
            services=conn.memory,
            thread_id=thread_id,
            store=conn.store,
            recursion_limit=args.recursion_limit,
            cache_namespace=cache_root,
            memory_namespace=args.memory_ns,
        )

        # Set up TUI
        app = PipelineApp()

        async def work():
            try:
                result = await run_natspec_pipeline(
                    system_doc=system_doc,
                    solc_version=args.solc_version,
                    tool_env=env,
                    ctx=ctx,
                    store=conn.store,
                    handler_factory=app.make_handler,
                    max_concurrent=args.max_concurrent,
                    max_bug_rounds=args.max_bug_rounds,
                )
                await app.on_pipeline_done(result)
            except Exception as exc:
                app.notify(f"Pipeline failed: {exc}", severity="error")
                app._pipeline_done = True

        app.set_work(work)
        await app.run_async()
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
