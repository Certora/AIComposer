"""Entry point for the NatSpec multi-agent pipeline TUI."""

import composer.certora as _

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
from composer.workflow.services import create_llm, get_checkpointer, get_store, standard_connections
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.spec.natspec.natspec_env import build_natspec_env
from composer.spec.source.source_env import SourceOnlyParams

from composer.spec.context import (
    WorkflowContext, SystemDoc, get_system_doc,
)
from composer.spec.natspec.pipeline import run_natspec_pipeline
from composer.spec.natspec.merge import DEFAULT_NEW_CONTRACTS_SUBDIR, DEFAULT_INTERFACES_SUBDIR
from composer.spec.util import string_hash, FS_FORBIDDEN_READ
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS

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
    source_root: str | None
    forbidden_read: str | None
    new_contracts_root: str | None
    interfaces_root: str | None
    prover_conf: str | None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="NatSpec multi-agent pipeline TUI"
    )
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument("input_file", help="Path to the design document (text or PDF)")
    parser.add_argument("--solc-version", default="8.29", help="Solidity compiler version (default: 8.29)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument("--source-root", default=None,
                        help="Path to an existing codebase root. When set, natspec runs in source-aware mode.")
    parser.add_argument("--forbidden-read", default=None,
                        help="Regex of paths source tools may not read. Defaults to FS_FORBIDDEN_READ when source-root is set.")
    parser.add_argument("--new-contracts-root", default=None,
                        help="Subdirectory (under source-root) where generated stubs are overlaid during typecheck. "
                             "Defaults to 'certora-generated/contracts'.")
    parser.add_argument("--interfaces-root", default=None,
                        help="Subdirectory (under source-root) where generated interfaces are overlaid during typecheck. "
                             "Defaults to 'certora-generated/interfaces'.")
    parser.add_argument("--prover-conf", default=None,
                        help="Path to a Certora config JSON file whose keys (packages, link, solc_args, etc.) are "
                             "merged into every typecheck invocation. Dynamic keys (files, verify, solc, "
                             "compilation_steps_only) are always set by the pipeline.")

    args = cast(PipelineArgs, parser.parse_args())

    # Read input document (handles both text and PDF)
    input_path = pathlib.Path(args.input_file)
    content = get_system_doc(input_path)
    if content is None:
        print(f"Error: cannot read {input_path}")
        return 1
    system_doc = SystemDoc(content=content)

    # Set up services
    llm = create_llm(args)
    store = get_store()

    model = get_model()

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conn,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag
    ):
        source: SourceOnlyParams | None = None
        source_root_path: pathlib.Path | None = None
        if args.source_root:
            source_root_path = pathlib.Path(args.source_root).resolve()
            source = SourceOnlyParams(
                root=str(source_root_path),
                forbidden_read=args.forbidden_read or FS_FORBIDDEN_READ,
                source_question_ns=(
                    "source_agent", "cache",
                    args.cache_ns or "default",
                    string_hash(str(system_doc.content)),
                ),
            )

        env = build_natspec_env(
            llm=llm,
            checkpoint=conn.checkpointer,
            db=rag,
            cvl_cache_ns=DEFAULT_CVL_AGENT_INDEX_NS,
            kb_ns=DEFAULT_KB_NS,
            store=conn.indexed_store,
            source=source,
        )

        new_contracts_subdir = (
            pathlib.Path(args.new_contracts_root) if args.new_contracts_root else DEFAULT_NEW_CONTRACTS_SUBDIR
        )
        interfaces_subdir = (
            pathlib.Path(args.interfaces_root) if args.interfaces_root else DEFAULT_INTERFACES_SUBDIR
        )
        conf_overrides: dict | None = None
        if args.prover_conf:
            conf_overrides = json.loads(pathlib.Path(args.prover_conf).read_text())

        cache_root = (args.cache_ns, string_hash(str(system_doc.content))) if args.cache_ns else None

        thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        ctx = WorkflowContext.create(
            services=lambda ns: async_memory_tool(conn.memory(ns)),
            thread_id=thread_id,
            store=store,
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
                    store=store,
                    handler_factory=app.make_handler,
                    max_concurrent=args.max_concurrent,
                    source_root=source_root_path,
                    new_contracts_subdir=new_contracts_subdir,
                    interfaces_subdir=interfaces_subdir,
                    conf_overrides=conf_overrides,
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
