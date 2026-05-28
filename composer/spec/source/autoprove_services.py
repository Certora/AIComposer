"""Shared services wiring for the auto-prove entry points.

Both ``autoprove_common`` (the standard pipeline) and
``autoprove_properties_common`` (the property-driven pipeline) need the same
LLM / RAG / source-env / WorkflowContext / ProverOptions setup. This module
exposes:

- ``add_base_autoprove_args`` — adds the CLI flags shared by every entry point
  to an argparse parser.
- ``BaseAutoProveArgs`` — Protocol type for those args.
- ``autoprove_services`` — async context manager that takes parsed args and
  yields a dataclass with the wired services.
"""

import argparse
import hashlib
import pathlib
import shlex
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from langchain_core.language_models.chat_models import BaseChatModel

from graphcore.tools.memory import async_memory_tool

from composer.input.types import DEFAULT_RECURSION_LIMIT, ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.kb.knowledge_base import DefaultEmbedder
from composer.prover.core import ProverOptions, make_prover_options
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.spec.context import SourceCode, WorkflowContext, get_document_input
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.spec.source.source_env import SourceEnvironment, build_source_env
from composer.spec.util import FS_FORBIDDEN_READ
from composer.ui.tool_display import async_tool_context
from composer.workflow.services import create_llm, standard_connections


class BaseAutoProveArgs(ModelOptions, RAGDBOptions, Protocol):
    """Fields every auto-prove entry point's arg-parser produces."""
    project_root: str
    main_contract: str
    system_doc: str
    max_concurrent: int
    cache_ns: str | None
    memory_ns: str | None
    cloud: bool
    prover_extra_args: str | None
    interactive: bool
    recursion_limit: int


def add_base_autoprove_args(parser: argparse.ArgumentParser) -> None:
    """Register every flag in ``BaseAutoProveArgs`` on the given parser."""
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument(
        "--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT,
        help=f"The number of iterations of the graph to allow (default: {DEFAULT_RECURSION_LIMIT})",
    )
    parser.add_argument("project_root", help="Root directory of the Solidity project")
    parser.add_argument("main_contract", help="Main contract as path:ContractName")
    parser.add_argument("system_doc", help="Path to the design document (text or PDF)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument("--cloud", action="store_true", help="Run prover jobs in the cloud")
    parser.add_argument(
        "--prover-extra-args", default=None,
        help='Extra arguments forwarded to certoraRun as a quoted string (e.g. "--rule_sanity advanced --smt_timeout 600")',
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactively refine the security properties after extraction",
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


@dataclass
class AutoProveServices:
    """The wired services every auto-prove entry point shares."""
    llm: BaseChatModel
    ctx: WorkflowContext[None]
    source_env: SourceEnvironment
    system_doc: SourceCode
    prover_opts: ProverOptions
    relative_path: str
    contract_name: str
    project_root: pathlib.Path


@asynccontextmanager
async def autoprove_services(
    args: BaseAutoProveArgs,
    parser: argparse.ArgumentParser,
    thread_prefix: str = "autoprove",
) -> AsyncIterator[AutoProveServices]:
    """Resolve services from parsed args.

    ``parser`` is forwarded so we can surface argument errors via
    ``parser.error`` (consistent with how the existing common module reports
    them).
    """
    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)

    full_contract_path = pathlib.Path(main_contract_path).resolve()
    if not full_contract_path.is_relative_to(project_root):
        parser.error(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")

    relative_path = str(full_contract_path.relative_to(project_root))

    sys_path = pathlib.Path(args.system_doc)
    content = get_document_input(sys_path)
    if content is None:
        parser.error(f"cannot read {sys_path}")

    system_doc = SourceCode(
        content=content,
        project_root=str(project_root),
        contract_name=contract_name,
        relative_path=relative_path,
        forbidden_read=FS_FORBIDDEN_READ,
    )

    llm = create_llm(args)
    model = get_model()

    root_key = _root_cache_key(args.project_root, sys_path, relative_path, contract_name)
    cache_root: tuple[str, str] | None = (args.cache_ns, root_key) if args.cache_ns is not None else None
    thread_id = f"{thread_prefix}_{uuid.uuid4().hex[:12]}"

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conns,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag_db,
        async_tool_context(),
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
            source_question_ns=("source_agent", "cache", root_key),
            recursion_limit=args.recursion_limit,
        )
        ctx = WorkflowContext.create(
            services=lambda namespace: async_memory_tool(conns.memory(namespace)),
            thread_id=thread_id,
            store=conns.store,
            recursion_limit=args.recursion_limit,
            cache_namespace=cache_root,
            memory_namespace=args.memory_ns,
        )

        prover_opts = make_prover_options(
            cloud=args.cloud,
            user_extra_args=shlex.split(args.prover_extra_args) if args.prover_extra_args else [],
        )

        yield AutoProveServices(
            llm=llm,
            ctx=ctx,
            source_env=source_env,
            system_doc=system_doc,
            prover_opts=prover_opts,
            relative_path=relative_path,
            contract_name=contract_name,
            project_root=project_root,
        )
