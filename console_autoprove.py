"""Entry point for the auto-prove pipeline — console (no TUI) mode."""

import composer.certora as _

import argparse
import asyncio
import hashlib
import pathlib
import uuid
from typing import cast, Protocol


from graphcore.tools.memory import memory_tool

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import (
    create_llm, get_checkpointer, get_store, get_memory,
    get_indexed_store,
)

from composer.spec.context import WorkflowContext, SourceCode, get_system_doc
from composer.spec.source.pipeline import run_autoprove_pipeline
from composer.spec.source.prover import CloudConfig
from composer.spec.source.source_env import build_source_env
from composer.spec.util import FS_FORBIDDEN_READ

from composer.ui.autoprove_console import AutoProveConsoleHandler


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
# Cache key  (matches tui_autoprove._root_cache_key exactly)
# ---------------------------------------------------------------------------

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
    parser.add_argument("system_doc", help="Design document (text or PDF)")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--cache-ns", default=None, dest="cache_ns",
                        help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, dest="memory_ns")
    parser.add_argument("--cloud", action="store_true",
                        help="Run prover jobs in the cloud")

    args = cast(AutoProveArgs, parser.parse_args())

    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)
    full_contract_path = pathlib.Path(main_contract_path).resolve()
    if not full_contract_path.is_relative_to(project_root):
        print(f"Error: {full_contract_path} is not within {project_root}")
        return 1
    relative_path = str(full_contract_path.relative_to(project_root))

    sys_path = pathlib.Path(args.system_doc)
    content = get_system_doc(sys_path)
    if content is None:
        print(f"Error: cannot read {sys_path}")
        return 1

    source_input = SourceCode(
        content=content,
        project_root=args.project_root,
        contract_name=contract_name,
        relative_path=relative_path,
        forbidden_read=FS_FORBIDDEN_READ,
    )

    llm = create_llm(args)
    checkpointer = get_checkpointer()
    store = get_store()
    model = get_model()
    indexed_store = get_indexed_store(DefaultEmbedder(model))
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db, model=model, skip_test=True
    )

    cache_root = None
    if args.cache_ns is not None:
        cache_root = (args.cache_ns, _root_cache_key(
            args.project_root, sys_path, relative_path, contract_name,
        ))

    thread_id = f"autoprove_{uuid.uuid4().hex[:12]}"
    ctx = WorkflowContext.create(
        services=lambda namespace: memory_tool(get_memory(namespace)),
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
        store=indexed_store,
    )

    handler = AutoProveConsoleHandler()

    result = await run_autoprove_pipeline(
        llm=llm,
        ctx=ctx,
        source_input=source_input,
        env=source_env,
        handler_factory=handler.make_handler,
        cloud=CloudConfig() if args.cloud else None,
        max_concurrent=args.max_concurrent,
    )

    print(f"\n{'=' * 60}")
    print("Auto-prove complete")
    print(f"  Components:  {result.n_components}")
    print(f"  Properties:  {result.n_properties}")
    if result.failures:
        print(f"  Failures:    {len(result.failures)}")
        for f in result.failures:
            print(f"    - {f}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
