"""Entry point for the auto-prove multi-agent pipeline TUI."""

import composer.certora as _

import argparse
import asyncio
import hashlib
import pathlib
import uuid
from typing import cast, Protocol

from langchain_core.tools import BaseTool

from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from graphcore.graph import Builder
from graphcore.tools.memory import memory_tool
from graphcore.tools.vfs import fs_tools

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from composer.tools.search import cvl_manual_tools
from composer.workflow.services import create_llm, get_checkpointer, get_store, get_memory, get_indexed_store
from composer.kb.knowledge_base import kb_tools

from composer.spec.context import (
    WorkflowContext, SourceCode, SourceBuilder, CVLBuilder, CVLOnlyBuilder,
    get_system_doc,
)
from composer.spec.source.pipeline import run_autoprove_pipeline
from composer.spec.source.prover import CloudConfig
from composer.spec.source.source_env import build_source_env

from composer.io.autoprove_app import AutoProveApp

from composer.spec.util import FS_FORBIDDEN_READ


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

class AutoProveArgs(ModelOptions, RAGDBOptions, Protocol):
    project_root: str
    main_contract: str
    system_doc: str
    max_concurrent: int
    cache_ns: str | None
    memory_ns: str | None
    cloud: bool


# ---------------------------------------------------------------------------
# WorkflowServices implementation
# ---------------------------------------------------------------------------

class _AutoProveServices:
    """Concrete ``WorkflowServices`` for the auto-prove entry point."""

    def __init__(self, checkpointer: Checkpointer, indexed_store: BaseStore):
        self._checkpointer = checkpointer
        self._indexed_store = indexed_store

    def kb_tools(self, read_only: bool) -> list[BaseTool]:
        return kb_tools(self._indexed_store, ("cvl",), read_only)

    def memory_tool(self, namespace: str) -> BaseTool:
        backend = get_memory(namespace)
        return memory_tool(backend)

    @property
    def checkpointer(self) -> Checkpointer:
        return self._checkpointer


# ---------------------------------------------------------------------------
# Builder construction
# ---------------------------------------------------------------------------

def _make_builders(
    llm,
    rag_db: PostgreSQLRAGDatabase,
    project_root: str,
) -> tuple[SourceBuilder, CVLBuilder, CVLOnlyBuilder]:
    """Create role-based builders for the auto-prove pipeline.

    Returns (source_tools, cvl_authorship, cvl_research).
    """
    base = Builder().with_llm(llm).with_loader(load_jinja_template)
    cvl_manual = cvl_manual_tools(rag_db)
    project_fs = fs_tools(project_root, forbidden_read=FS_FORBIDDEN_READ)

    return (
        base.with_tools(project_fs),
        base.with_tools([*cvl_manual, *project_fs]),
        base.with_tools(cvl_manual),
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _root_cache_key(
    project_root: str,
    system_doc_path: pathlib.Path,
    relative_path: str,
    contract_name: str,
) -> str:
    """Generate a cache key from all inputs that affect the analysis."""
    doc_hash = hashlib.sha256(system_doc_path.read_bytes()).hexdigest()
    combined = "|".join([project_root, doc_hash, relative_path, contract_name])
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-prove multi-agent pipeline TUI"
    )
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument("project_root", help="Root directory of the Solidity project")
    parser.add_argument("main_contract", help="Main contract as path:ContractName")
    parser.add_argument("system_doc", help="Path to the design document (text or PDF)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument("--cloud", action="store_true", help="Run prover jobs in the cloud")

    args = cast(AutoProveArgs, parser.parse_args())

    # Parse main_contract (path:ContractName)
    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)

    full_contract_path = pathlib.Path(main_contract_path).resolve()
    if not full_contract_path.is_relative_to(project_root):
        print(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")
        return 1

    relative_path = str(full_contract_path.relative_to(project_root))

    # Read input document
    sys_path = pathlib.Path(args.system_doc)
    content = get_system_doc(sys_path)
    if content is None:
        print(f"Error: cannot read {sys_path}")
        return 1

    system_doc = SourceCode(
        content=content,
        project_root=args.project_root,
        contract_name=contract_name,
        relative_path=relative_path,
        forbidden_read=FS_FORBIDDEN_READ,
        solidity_compiler="solc8.31"
    )

    # Set up services
    llm = create_llm(args)
    checkpointer = get_checkpointer()
    store = get_store()
    model = get_model()
    indexed_store = get_indexed_store(DefaultEmbedder(model))
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db, model=model, skip_test=True
    )

    services = _AutoProveServices(checkpointer, indexed_store)
    source_tools, cvl_authorship, cvl_research = _make_builders(llm, rag_db, args.project_root)

    cache_root: tuple[str, str] | None = None
    if args.cache_ns is not None:
        cache_root = (args.cache_ns, _root_cache_key(
            args.project_root, sys_path, relative_path, contract_name,
        ))

    thread_id = f"autoprove_{uuid.uuid4().hex[:12]}"
    ctx = WorkflowContext.create(
        services=services,
        thread_id=thread_id,
        store=store,
        cache_namespace=cache_root,
        memory_namespace=args.memory_ns,
    )

    source_env = build_source_env(
        llm=llm,
        db=rag_db,
        checkpoint=checkpointer,
        forbidden_read=FS_FORBIDDEN_READ,
        kb_ns=("cvl",),
        root=args.project_root,
        store=indexed_store
    )

    # Set up TUI
    app = AutoProveApp()

    async def work():
        try:
            result = await run_autoprove_pipeline(
                ctx=ctx,
                source_input=system_doc,
                env=source_env,
                handler_factory=app.make_handler,
                cloud=CloudConfig() if args.cloud else None,
                max_concurrent=args.max_concurrent,
            )
            summary = (
                f"Auto-prove complete: {result.n_components} components, "
                f"{result.n_properties} properties"
            )
            if result.failures:
                summary += f", {len(result.failures)} failures"
            app.notify(summary)
            app._pipeline_done = True
        except Exception as exc:
            app.notify(f"Pipeline failed: {exc}", severity="error")
            app._pipeline_done = True

    app.set_work(work)
    await app.run_async()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
