"""Entry point for the NatSpec multi-agent pipeline TUI.

This driver covers the ``greenfield`` and ``update`` natspec workflows.
``existing`` (verify-as-is from source) lives in ``console_autoprove``.
"""

import composer.bind as _

import argparse
import asyncio
import json
import pathlib
import sys
import uuid
from typing import cast, Protocol


from graphcore.tools.memory import async_memory_tool

from composer.core.user import user_data_ns
from composer.input.types import ModelOptions, RAGDBOptions, DEFAULT_RECURSION_LIMIT
from composer.input.parsing import add_protocol_args
from composer.io.thread_logging import DEFAULT_META_NS, thread_logger
from composer.spec.agent_index import agent_index_config_from_env
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import create_llm, standard_connections
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.spec.services import build_rag_tool_env

from composer.spec.context import (
    WorkflowContext, SystemDoc,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline
from composer.spec.natspec.run_tags import NatspecRunTags
from composer.spec.util import FS_FORBIDDEN_READ
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.ui.tool_display import async_tool_context

from composer.ui.pipeline_app import NatspecPipelineApp
from composer.cli.natspec_startup import build_mental_model, make_source_factory


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
    source_root: str | None
    forbidden_read: str | None
    prover_conf: str | None
    output_root: str | None
    interactive: bool
    max_bug_rounds: int
    recursion_limit: int


# ---------------------------------------------------------------------------
# MentalModel construction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="NatSpec multi-agent pipeline TUI (greenfield / update)"
    )
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument(
        "--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT,
        help=f"The number of iterations of the graph to allow (default: {DEFAULT_RECURSION_LIMIT})",
    )
    parser.add_argument("input_file", help="Path to the design document (text or PDF)")
    parser.add_argument("--solc-version", default="8.29", help="Solidity compiler version (default: 8.29)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument(
        "--source-root", default=None,
        help="Path to an existing codebase root. When set, natspec runs in `update` mode: "
             "contracts are tagged unchanged/edited/new, and specs are generated only for "
             "the new contracts. When unset, natspec runs in `greenfield` mode.",
    )
    parser.add_argument(
        "--forbidden-read", default=None,
        help="Regex of paths source tools may not read. Defaults to FS_FORBIDDEN_READ "
             "when source-root is set.",
    )
    parser.add_argument(
        "--prover-conf", default=None,
        help="Path to a Certora config JSON file whose keys (packages, link, solc_args, etc.) "
             "are merged into every typecheck invocation. Dynamic keys (files, verify, solc, "
             "compilation_steps_only) are always set by the pipeline.",
    )
    parser.add_argument(
        "--interactive", action="store_true", default=False,
        help="Open a per-component conversation channel during bug analysis so the user "
             "can refine the extracted property list interactively before CVL generation. "
             "Each component's channel is its own focusable panel in the TUI; use the "
             "switcher to navigate.",
    )
    parser.add_argument(
        "--max-bug-rounds", type=int, default=3,
        help="Maximum number of bug-extraction rounds run per component during property "
             "analysis (default: 3). Lower for faster runs at the cost of less thorough "
             "property surfacing; higher to give the agent more room to refine.",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Directory under which to write the `natspec_output/` folder containing "
             "every contract's generated interface, stub, and specs. Defaults to the "
             "current working directory if not set.",
    )

    args = cast(PipelineArgs, parser.parse_args())

    input_path = pathlib.Path(args.input_file)

    # Set up services
    llm = create_llm(args)
    model = get_model()

    logging_ns = user_data_ns() + DEFAULT_META_NS
    run_id = uuid.uuid4().hex

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conn,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag,
        async_tool_context(),
    ):
        content = await conn.uploader.get_document(input_path)
        if content is None:
            print(f"Error: cannot read {input_path}")
            return 1
        system_doc = SystemDoc(content=content)

        source_root_path: pathlib.Path | None = None
        if args.source_root:
            source_root_path = pathlib.Path(args.source_root).resolve()

        sort = "update" if source_root_path is not None else "greenfield"
        forbidden_read = (
            args.forbidden_read or (FS_FORBIDDEN_READ if source_root_path else None)
        )

        thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        doc_digest = system_doc.content.to_digest()
        run_tags = NatspecRunTags(
            root_thread_id=thread_id,
            doc_digest=doc_digest,
            cache_namespace=args.cache_ns,
            memory_namespace=args.memory_ns,
            from_source=source_root_path is not None,
            interactive=args.interactive,
        )

        print(f"[natspec] run_id={run_id}  thread={thread_id}", file=sys.stderr)

        async with thread_logger(
            conn.store, run_tags.model_dump(), logging_ns, run_id=run_id,
        ):
            start_env = build_rag_tool_env(
                sort=sort,
                llm=llm,
                checkpoint=conn.checkpointer,
                db=rag,
                cvl_index_config=agent_index_config_from_env(DEFAULT_CVL_AGENT_INDEX_NS),
                kb_ns=DEFAULT_KB_NS,
                store=conn.indexed_store,
                recursion_limit=args.recursion_limit,
            )

            source_factory = make_source_factory(source_root_path, forbidden_read)

            config_init: dict | None = None
            if args.prover_conf:
                config_init = json.loads(pathlib.Path(args.prover_conf).read_text())

            mental_model = build_mental_model(
                source_root=source_root_path,
                config_init=config_init,
            )

            cache_root = (args.cache_ns, doc_digest) if args.cache_ns else None

            ctx = WorkflowContext.create(
                services=lambda ns: async_memory_tool(conn.memory(ns)),
                thread_id=thread_id,
                store=conn.store,
                recursion_limit=args.recursion_limit,
                cache_namespace=cache_root,
                memory_namespace=args.memory_ns,
            )

            output_root_path: pathlib.Path | None = None
            if args.output_root:
                output_root_path = pathlib.Path(args.output_root).resolve()

            # Set up TUI
            app = NatspecPipelineApp(
                output_root=output_root_path,
            )

            async def work():
                try:
                    result = await run_natspec_pipeline(
                        system_doc=system_doc,
                        solc_version=args.solc_version,
                        start_env=start_env,
                        ctx=ctx,
                        store=conn.store,
                        handler_factory=app.make_handler,
                        mental_model=mental_model,
                        source_factory=source_factory,
                        max_concurrent=args.max_concurrent,
                        interactive=args.interactive,
                        max_bug_rounds=args.max_bug_rounds,
                    )
                    await app.on_pipeline_done(result)
                except Exception as exc:
                    app.notify(f"Pipeline failed: {exc}", severity="error")
                    await app.mount_error(exc)
                    app._pipeline_done = True

            app.set_work(work)
            await app.run_async()
        return 0


def main() -> int:
    return asyncio.run(_main())
