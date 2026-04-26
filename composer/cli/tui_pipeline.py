"""Entry point for the NatSpec multi-agent pipeline TUI.

This driver covers the ``greenfield`` and ``update`` natspec workflows.
``existing`` (verify-as-is from source) lives in ``console_autoprove``.
"""

import composer.bind as _

import argparse
import asyncio
import json
import pathlib
import uuid
from typing import cast, Protocol


from graphcore.tools.memory import async_memory_tool

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import create_llm, standard_connections
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.spec._env_common import build_rag_tool_env
from composer.spec.service_host import PureServiceHost

from composer.spec.context import (
    WorkflowContext, SystemDoc, get_document_input,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline
from composer.spec.util import string_hash, FS_FORBIDDEN_READ
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
        "--output-root", default=None,
        help="Directory to persist pipeline artifacts under. When the VS Code extension is "
             "connected, writes `implementation_plan.json` here; generated files flow through "
             "the extension's preview/accept flow instead. Without the extension, this is "
             "where generated interfaces / stubs / specs are written (required to persist "
             "anything in that mode).",
    )

    args = cast(PipelineArgs, parser.parse_args())

    # Read input document (handles both text and PDF)
    input_path = pathlib.Path(args.input_file)
    content = get_document_input(input_path)
    if content is None:
        print(f"Error: cannot read {input_path}")
        return 1
    system_doc = SystemDoc(content=content)

    # Set up services
    llm = create_llm(args)
    model = get_model()

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conn,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag,
        async_tool_context()
    ):
        source_root_path: pathlib.Path | None = None
        if args.source_root:
            source_root_path = pathlib.Path(args.source_root).resolve()

        sort = "update" if source_root_path is not None else "greenfield"
        forbidden_read = (
            args.forbidden_read or (FS_FORBIDDEN_READ if source_root_path else None)
        )

        rag_env = build_rag_tool_env(
            llm=llm,
            checkpoint=conn.checkpointer,
            db=rag,
            cvl_cache_ns=DEFAULT_CVL_AGENT_INDEX_NS,
            kb_ns=DEFAULT_KB_NS,
            store=conn.indexed_store,
        )

        start_env = PureServiceHost(
            llm=rag_env.llm,
            builder=rag_env.builder,
            cvl_tools=tuple(rag_env.rag_tools),
            sort=sort,
        )

        source_factory = make_source_factory(source_root_path, forbidden_read)

        config_init: dict | None = None
        if args.prover_conf:
            config_init = json.loads(pathlib.Path(args.prover_conf).read_text())

        mental_model = build_mental_model(
            source_root=source_root_path,
            config_init=config_init,
        )

        cache_root = (args.cache_ns, string_hash(str(system_doc.content))) if args.cache_ns else None

        thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        ctx = WorkflowContext.create(
            services=lambda ns: async_memory_tool(conn.memory(ns)),
            thread_id=thread_id,
            store=conn.store,
            cache_namespace=cache_root,
            memory_namespace=args.memory_ns,
        )

        output_root_path: pathlib.Path | None = None
        if args.output_root:
            output_root_path = pathlib.Path(args.output_root).resolve()

        # Set up TUI
        app = NatspecPipelineApp(
            system_doc_path=input_path.resolve(),
            source_root=source_root_path,
            prover_conf=config_init,
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
