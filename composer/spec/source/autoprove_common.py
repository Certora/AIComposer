"""Entry point for the auto-prove multi-agent pipeline TUI."""

import argparse
import hashlib
import pathlib
import uuid
from typing import cast, Protocol, Callable, Awaitable

from graphcore.tools.memory import async_memory_tool

from composer.input.types import ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import create_llm, standard_connections

from composer.spec.context import (
    WorkflowContext, SourceCode, get_document_input,
)
from composer.spec.source.pipeline import run_autoprove_pipeline, AutoProveResult
from composer.spec.source.direct_pipeline import run_autoprove_pipeline as run_direct_autoprove_pipeline
from composer.spec.source.prover import CloudConfig
from composer.spec.source.source_env import build_source_env
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.ui.autoprove_app import AutoProvePhase
from composer.ui.tool_display import async_tool_context

from composer.spec.util import FS_FORBIDDEN_READ
from composer.io.multi_job import HandlerFactory


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
    interactive: bool
    threat_model: str
    properties_path: str
    custom_summary_path: str | None
    standard_summary_path: str | None
    config_paths: list[str]
    skip_setup: bool

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

type Executor = Callable[[HandlerFactory[AutoProvePhase, None]], Awaitable[AutoProveResult]]
type ExecutorCB = Callable[[Executor], Awaitable[int]]

async def _entry_point(cb: ExecutorCB) -> int:
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
    parser.add_argument("--interactive", action="store_true", help="Interactively refine the security properties after extraction")
    parser.add_argument("--threat-model", type=str, default=None, help="Path to a 'thread' model (text or pdf) with which to seed the property extraction process")
    parser.add_argument("--custom-summary-path", type=str, default=None, help="Path to a custom CVL summaries file specific to the contract")
    parser.add_argument("--standard-summary-path", type=str, default=None, help="Path to the standard PreAudit-generated CVL summaries file")
    parser.add_argument("--config-paths", type=lambda s: [p.strip() for p in s.split(",") if p.strip()], default=[], help="Comma-separated list of .conf paths, relative to project_root. If empty, all configs in the folders are used.",)
    parser.add_argument("--skip-setup", action="store_true", help="Assumes the project has already been setup")
    parser.add_argument("--properties-path", type=str, default=None, help="Path to the list of properties - a document (text or PDF)")

    args = cast(AutoProveArgs, parser.parse_args())

    # Parse main_contract (path:ContractName)
    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)

    full_contract_path = pathlib.Path(main_contract_path).resolve()
    if not full_contract_path.is_relative_to(project_root):
        print(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")
        return 1
    
    if args.config_paths:
        if not args.skip_setup:
            print(f"Supplying config path is only allowed when the setup has been done (AutoSetup ran before)")
            return 1
        for p in args.config_paths:
            if not p.endswith(".conf"):
                print(f"Invalid config path: {p} is not a .conf file")
                return 1
        args.config_paths = [str((project_root / p).resolve()) for p in args.config_paths]
    else:
        args.config_paths = [str(p) for p in sorted(project_root.rglob("*.conf"))]

    relative_path = str(full_contract_path.relative_to(project_root))

    # Read input document
    sys_path = pathlib.Path(args.system_doc)
    content = get_document_input(sys_path)
    if content is None:
        print(f"Error: cannot read {sys_path}")
        return 1

    system_doc = SourceCode(
        content=content,
        project_root=str(project_root),
        contract_name=contract_name,
        relative_path=relative_path,
        forbidden_read=FS_FORBIDDEN_READ,
    )

    # Set up services
    llm = create_llm(args)
    model = get_model()


    cache_root: tuple[str, str] | None = None

    root_key = _root_cache_key(
            args.project_root, sys_path, relative_path, contract_name,
        )

    if args.cache_ns is not None:
        cache_root = (args.cache_ns, root_key)

    thread_id = f"autoprove_{uuid.uuid4().hex[:12]}"

    threat_model = get_document_input(pathlib.Path(threat_path)) if (threat_path := args.threat_model) is not None else None

    properties = get_document_input(pathlib.Path(properties_path)) if (properties_path := args.properties_path) is not None else None

    async with (
        standard_connections(
            embedder=DefaultEmbedder(model)
        ) as conns,
        PostgreSQLRAGDatabase.rag_context(model) as rag_db,
        async_tool_context()
    ):
        source_env = build_source_env(
            llm=llm,
            db=rag_db,
            checkpoint=conns.checkpointer,
            forbidden_read=FS_FORBIDDEN_READ,
            kb_ns=("cvl",),
            root=args.project_root,
            store=conns.indexed_store,
            cvl_cache_ns=DEFAULT_CVL_AGENT_INDEX_NS,
            source_question_ns=("source_agent", "cache", root_key)
        )
        ctx = WorkflowContext.create(
            services=lambda namespace: async_memory_tool(conns.memory(namespace)),
            thread_id=thread_id,
            store=conns.store,
            cache_namespace=cache_root,
            memory_namespace=args.memory_ns,
        )

        async def runner(handler: HandlerFactory[AutoProvePhase, None]) -> AutoProveResult:
            if args.skip_setup:
                return await run_direct_autoprove_pipeline(
                        llm=llm,
                        ctx=ctx,
                        source_input=system_doc,
                        config_paths=args.config_paths,
                        env=source_env,
                        handler_factory=handler,
                        custom_summary_path=args.custom_summary_path,
                        standard_summary_path=args.standard_summary_path,
                        cloud=CloudConfig() if args.cloud else None,
                        max_concurrent=args.max_concurrent,
                        threat_model=threat_model,
                        interactive=args.interactive,
                        properties=properties
                    )
            return await run_autoprove_pipeline(
                    llm=llm,
                    ctx=ctx,
                    source_input=system_doc,
                    env=source_env,
                    handler_factory=handler,
                    cloud=CloudConfig() if args.cloud else None,
                    max_concurrent=args.max_concurrent,
                    interactive=args.interactive,
                    threat_model=threat_model
                )
        return await cb(runner)
