"""Entry point for the NatSpec multi-agent pipeline TUI."""

import composer.certora as _

import argparse
import asyncio
import pathlib
import uuid
from typing import cast, Protocol

from langchain_core.tools import BaseTool

from graphcore.graph import Builder
from graphcore.tools.memory import memory_tool

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from composer.tools.search import cvl_manual_tools
from composer.workflow.services import create_llm, get_checkpointer, get_store, get_memory
from composer.kb.knowledge_base import kb_tools

from composer.spec.context import (
    WorkflowContext, SystemDoc, PlainBuilder, CVLOnlyBuilder,
    get_system_doc,
)
from composer.spec.pipeline import run_natspec_pipeline
from composer.spec.util import string_hash

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


# ---------------------------------------------------------------------------
# WorkflowServices implementation
# ---------------------------------------------------------------------------

class _PipelineServices:
    """Concrete ``WorkflowServices`` for the pipeline entry point."""

    def __init__(self, checkpointer, store, rag_db: PostgreSQLRAGDatabase):
        self._checkpointer = checkpointer
        self._store = store
        self._rag_db = rag_db

    def kb_tools(self, read_only: bool) -> list[BaseTool]:
        return kb_tools(self._store, ("natspec_pipeline", "kb"), read_only)

    def memory_tool(self, namespace: str) -> BaseTool:
        backend = get_memory(namespace)
        return memory_tool(backend)

    @property
    def checkpointer(self):
        return self._checkpointer


# ---------------------------------------------------------------------------
# Builder construction
# ---------------------------------------------------------------------------

def _make_builders(
    llm, rag_db: PostgreSQLRAGDatabase,
) -> tuple[PlainBuilder, CVLOnlyBuilder, CVLOnlyBuilder]:
    """Create role-based builders for the pipeline.

    Returns (analysis_builder, cvl_authorship, cvl_research).
    Doc-only pipeline: analysis builder has no tools,
    CVL variants get the full CVL manual tool suite.
    """
    base = Builder().with_llm(llm).with_loader(load_jinja_template)
    cvl_tools = cvl_manual_tools(rag_db)
    cvl = base.with_tools(cvl_tools)
    return base, cvl, cvl


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
    parser.add_argument("--contract-name", required=True, help="Expected contract name")
    parser.add_argument("--solc-version", default="8.29", help="Solidity compiler version (default: 8.29)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")

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
    checkpointer = get_checkpointer()
    store = get_store()
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db, model=get_model(), skip_test=True
    )

    services = _PipelineServices(checkpointer, store, rag_db)
    analysis_builder, cvl_authorship, cvl_research = _make_builders(llm, rag_db)

    cache_root = (args.cache_ns, string_hash(str(system_doc.content))) if args.cache_ns else None

    thread_id = f"pipeline_{uuid.uuid4().hex[:12]}"
    ctx = WorkflowContext.create(
        services=services,
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
                contract_name=args.contract_name,
                solc_version=args.solc_version,
                analysis_builder=analysis_builder,
                cvl_authorship=cvl_authorship,
                cvl_research=cvl_research,
                ctx=ctx,
                store=store,
                handler_factory=app.make_handler,
                max_concurrent=args.max_concurrent,
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
