"""Console (non-TUI) entry point for the auto-prove pipeline.

Streams minimal start/end workflow notifications to stdout.
"""

import composer.certora as _

import argparse
import asyncio
import hashlib
import pathlib
import uuid
from typing import Any, cast, Callable, Protocol

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
from composer.spec.autoprove_pipeline import run_autoprove_pipeline
from composer.spec.prover import CloudConfig

from composer.io.autoprove_app import AutoProvePhase
from composer.io.multi_job_app import TaskInfo, TaskHandle
from composer.io.event_handler import NullEventHandler

from composer.spec.util import FS_FORBIDDEN_READ


# ---------------------------------------------------------------------------
# Console IOHandler — very brief start/end logging
# ---------------------------------------------------------------------------

class _ConsoleHandler:
    """Minimal IOHandler that prints workflow start/end to stdout."""

    def __init__(self, label: str):
        self._label = label

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        pass

    async def log_state_update(self, path: list[str], st: dict) -> None:
        pass

    async def progress_update(self, path: list[str], upd: Any) -> None:
        pass

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None) -> None:
        print(f"[START] {self._label}: {description}")

    async def log_end(self, path: list[str]) -> None:
        print(f"[END]   {self._label}")

    async def human_interaction(self, ty: Any, debug_thunk: Callable[[], None]) -> str:
        raise NotImplementedError("Console mode does not support HITL interactions")


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

async def _console_factory(info: TaskInfo[AutoProvePhase]) -> TaskHandle[Any, Any]:
    handler = _ConsoleHandler(info.label)
    return TaskHandle(
        handler=handler,
        event_handler=NullEventHandler(),
        on_start=lambda: print(f"[RUN]   {info.label}"),
        on_done=lambda: print(f"[DONE]  {info.label}"),
        on_error=lambda exc, tb: _log_error(info.label, exc, tb),
    )


async def _log_error(label: str, exc: Exception, tb: str) -> None:
    print(f"[ERROR] {label}: {exc}")


# ---------------------------------------------------------------------------
# Args (reused from tui_autoprove)
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
# Services (reused from tui_autoprove)
# ---------------------------------------------------------------------------

class _AutoProveServices:
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


def _make_builders(
    llm,
    rag_db: PostgreSQLRAGDatabase,
    project_root: str,
) -> tuple[SourceBuilder, CVLBuilder, CVLOnlyBuilder]:
    base = Builder().with_llm(llm).with_loader(load_jinja_template)
    cvl_manual = cvl_manual_tools(rag_db)
    project_fs = fs_tools(project_root, forbidden_read=FS_FORBIDDEN_READ)

    return (
        base.with_tools(project_fs),
        base.with_tools([*cvl_manual, *project_fs]),
        base.with_tools(cvl_manual),
    )


def _root_cache_key(
    project_root: str,
    system_doc_path: pathlib.Path,
    relative_path: str,
    contract_name: str,
) -> str:
    doc_hash = hashlib.sha256(system_doc_path.read_bytes()).hexdigest()
    combined = "|".join([project_root, doc_hash, relative_path, contract_name])
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-prove pipeline (console mode)"
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

    # Run pipeline with console handler factory
    result = await run_autoprove_pipeline(
        system_doc=system_doc,
        source_tools=source_tools,
        cvl_authorship=cvl_authorship,
        cvl_research=cvl_research,
        ctx=ctx,
        store=store,
        handler_factory=_console_factory,
        llm=llm,
        cloud=CloudConfig() if args.cloud else None,
        max_concurrent=args.max_concurrent,
    )

    print(
        f"\nAuto-prove complete: {result.n_components} components, "
        f"{result.n_properties} properties"
    )
    if result.failures:
        for f in result.failures:
            print(f"  FAIL: {f}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
