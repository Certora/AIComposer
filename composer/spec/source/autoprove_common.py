"""Shared entry-point plumbing for the auto-prove CLIs.

This module is the pipeline-agnostic *services* layer: it parses the shared CLI
arguments and stands up everything a pipeline needs (LLM, documents, source
environment, workflow context, prover options) inside one async context manager,
``_autoprove_services``. Each concrete entry point (``_entry_point`` for the
inference pipeline, ``_properties_entry_point`` for the known-properties pipeline)
adds its own arguments, enters the services CM, and yields a ``runner`` closure
that drives *its* pipeline. The services layer never imports a concrete pipeline;
the entry points do.
"""

import argparse
import hashlib
import logging
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast, AsyncIterator, Protocol, Callable, Awaitable

from langchain_core.language_models.chat_models import BaseChatModel

from graphcore.tools.memory import async_memory_tool

from composer.diagnostics.logging_setup import setup_autoprove_logging
from composer.diagnostics.timing import RunSummary, install_run_summary
from composer.input.types import DEFAULT_RECURSION_LIMIT, ModelOptions, RAGDBOptions
from composer.input.parsing import add_protocol_args
from composer.input.files import Document
from composer.kb.knowledge_base import DefaultEmbedder, DEFAULT_KB_NS
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.workflow.services import create_llm, standard_connections

from composer.spec.system_model import SolidityIdentifier
from composer.spec.context import (
    WorkflowContext, SourceCode,
)
from composer.spec.source.pipeline import run_autoprove_pipeline
from composer.spec.source.properties_pipeline import run_properties_pipeline
from composer.spec.source.common_pipeline import dump_token_usage, AutoProveResult
from composer.spec.source.known_properties import load_known_properties, KnownPropertiesError
from composer.prover.core import make_prover_options, ProverOptions
from composer.spec.source.source_env import build_source_env, SourceEnvironment
from composer.spec.agent_index import agent_index_config_from_env
from composer.core.user import get_uid, user_data_ns
from composer.spec.cvl_research import DEFAULT_CVL_AGENT_INDEX_NS
from composer.ui.autoprove_app import AutoProvePhase
from composer.ui.tool_display import async_tool_context
from composer.io.thread_logging import thread_logger, DEFAULT_META_NS

from composer.spec.util import FS_FORBIDDEN_READ
from composer.io.multi_job import HandlerFactory

_logger = logging.getLogger(__name__)

def user_ns(
    *parts: str | tuple[str, ...]
) -> tuple[str,...]:
    to_ret : list[str] = []
    for p in parts:
        if isinstance(p, str):
            to_ret.append(p)
        else:
            to_ret.extend(p)
    return user_data_ns() + tuple(to_ret)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

class CommonAutoProveArgs(ModelOptions, RAGDBOptions, Protocol):
    """The arguments shared by every auto-prove entry point."""
    project_root: str
    main_contract: str
    system_doc: str
    max_concurrent: int
    cache_ns: str | None
    memory_ns: str | None
    cloud: bool
    interactive: bool
    recursion_limit: int
    # Only the inference pipeline exposes --threat-model; the services layer
    # reads it uniformly, so it is nullable here and the properties entry point
    # set_defaults() it to None.
    threat_model: str | None


class AutoProveArgs(CommonAutoProveArgs, Protocol):
    """Inference-pipeline arguments (adds the bug-analysis concepts)."""
    interactive: bool
    max_bug_rounds: int


class PropertiesArgs(CommonAutoProveArgs, Protocol):
    """Known-properties-pipeline arguments."""
    properties: str
    skip_setup: bool
    conf: str | None
    summary: str | None


def add_common_autoprove_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every auto-prove entry point."""
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, ModelOptions)
    parser.add_argument("--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT, help=f"The number of iterations of the graph to allow (default: {DEFAULT_RECURSION_LIMIT})")
    parser.add_argument("project_root", help="Root directory of the Solidity project")
    parser.add_argument("main_contract", help="Main contract as path:ContractName")
    parser.add_argument("system_doc", help="Path to the design document (text or PDF)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument("--cloud", action="store_true", help="Run prover jobs in the cloud")
    parser.add_argument("--prover-extra-args", default=None, help='Extra arguments forwarded to certoraRun as a quoted string (e.g. "--rule_sanity advanced --smt_timeout 600")')

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
# Services
# ---------------------------------------------------------------------------

type Executor = Callable[[HandlerFactory[AutoProvePhase, None]], Awaitable[AutoProveResult]]  # pyright: ignore[reportInvalidTypeForm]


@dataclass
class AutoProveServices:
    """Everything a pipeline needs, stood up from the common arguments."""
    ctx: WorkflowContext[None]
    llm: BaseChatModel
    source_env: SourceEnvironment
    source: SourceCode
    prover_opts: ProverOptions
    args: CommonAutoProveArgs
    threat_model: Document | None


@asynccontextmanager
async def _autoprove_services(
    parser: argparse.ArgumentParser,
    args: CommonAutoProveArgs,
    summary: RunSummary,
) -> AsyncIterator[AutoProveServices]:
    """Stand up the shared services for a parsed set of common arguments.

    Pipeline-agnostic: the yielded ``AutoProveServices`` is everything any
    pipeline needs.
    """
    # Parse main_contract (path:ContractName)
    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path, contract_name = args.main_contract.split(":", 1)

    full_contract_path = pathlib.Path(main_contract_path).resolve()
    if not full_contract_path.is_relative_to(project_root):
        parser.error(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")

    relative_path = str(full_contract_path.relative_to(project_root))

    sys_path = pathlib.Path(args.system_doc)

    # Set up services
    llm = create_llm(args)
    model = get_model()

    cache_root: tuple[str, ...] | None = None

    root_key = _root_cache_key(
            str(project_root), sys_path, relative_path, contract_name,
        )

    if args.cache_ns is not None:
        cache_root = user_ns(args.cache_ns, root_key)

    thread_id = f"autoprove_{uuid.uuid4().hex[:12]}"

    text_log, events_log = setup_autoprove_logging(project_root, thread_id)
    print(f"autoprove logs: {text_log}\n           events: {events_log}", file=sys.stderr)
    install_run_summary(summary)

    try:
        async with (
            standard_connections(
                embedder=DefaultEmbedder(model)
            ) as conns,
            PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as rag_db,
            async_tool_context(),
            thread_logger(
                conns.store,
                {"root_thread_id": thread_id},
                user_ns(DEFAULT_META_NS),
                run_id=summary.run_id,
                # Persist final token usage into RunMeta.tags at run close (totals
                # known only once the pipeline is done). Mirrors token_usage.json.
                finalize_tags=lambda: {"token_usage": summary.token_usage_summary()},
            )
        ):
            # Source-code agent caches are always per-user — the conventional
            # ``user_data_ns(uid)`` prefix lives directly in the ns we pass
            # so the AgentIndex runs single-pool (no overlay).
            source_data_ns = user_ns("source_agent", "cache", root_key)
            # Read input documents now that the uploader is available.
            content = await conns.uploader.get_document(sys_path)
            if content is None:
                parser.error(f"cannot read {sys_path}")

            system_doc = SourceCode(
                content=content,
                project_root=str(project_root),
                contract_name=SolidityIdentifier(contract_name),
                relative_path=relative_path,
                forbidden_read=FS_FORBIDDEN_READ,
            )

            threat_model = (
                await conns.uploader.get_document(pathlib.Path(args.threat_model))
                if args.threat_model is not None else None
            )
            source_env = build_source_env(
                llm=llm,
                db=rag_db,
                checkpoint=conns.checkpointer,
                forbidden_read=FS_FORBIDDEN_READ,
                kb_ns=DEFAULT_KB_NS,
                root=args.project_root,
                store=conns.indexed_store,
                source_question_ns=source_data_ns,
                recursion_limit=args.recursion_limit,
                cvl_index_config=agent_index_config_from_env(DEFAULT_CVL_AGENT_INDEX_NS),
            )

            memory_ns = args.memory_ns
            if memory_ns:
                memory_ns = get_uid() + "/" + memory_ns
            ctx = WorkflowContext.create(
                services=lambda namespace: async_memory_tool(conns.memory(namespace)),
                thread_id=thread_id,
                store=conns.store,
                recursion_limit=args.recursion_limit,
                cache_namespace=cache_root,
                memory_namespace=memory_ns,
            )

            prover_opts = make_prover_options(cloud=args.cloud)

            yield AutoProveServices(
                ctx=ctx,
                llm=llm,
                source_env=source_env,
                source=system_doc,
                prover_opts=prover_opts,
                args=args,
                threat_model=threat_model,
            )
    finally:
        # Dump final LLM token usage for the run (success or failure). Single
        # choke point both console and TUI entry points pass through, with
        # project_root in scope and the summary fully populated. Guarded so a
        # diagnostics-dump failure can never mask the pipeline's own outcome.
        try:
            dump_token_usage(str(project_root), summary)
        except Exception:
            _logger.exception("failed to dump token usage")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _entry_point(summary: RunSummary) -> AsyncIterator[Executor]:
    """Inference-pipeline entry point (the original ``*-autoprove`` commands)."""
    parser = argparse.ArgumentParser(
        description="Auto-prove multi-agent pipeline TUI"
    )
    add_common_autoprove_args(parser)
    parser.add_argument("--interactive", action="store_true", help="Interactively refine the security properties after extraction")
    parser.add_argument("--threat-model", type=str, default=None, help="Path to a 'thread' model (text or pdf) with which to seed the property extraction process")
    parser.add_argument("--max-bug-rounds", type=int, default=3, help="Maximum number of bug-extraction rounds run per component during property analysis (default: 3)")

    args = cast(AutoProveArgs, parser.parse_args())

    async with _autoprove_services(parser, args, summary) as svc:
        async def runner(handler: HandlerFactory[AutoProvePhase, None]) -> AutoProveResult:
            return await run_autoprove_pipeline(
                llm=svc.llm,
                ctx=svc.ctx,
                source_input=svc.source,
                env=svc.source_env,
                handler_factory=handler,
                prover_opts=svc.prover_opts,
                max_concurrent=args.max_concurrent,
                interactive=args.interactive,
                threat_model=svc.threat_model,
                max_bug_rounds=args.max_bug_rounds,
            )

        yield runner


@asynccontextmanager
async def _properties_entry_point(summary: RunSummary) -> AsyncIterator[Executor]:
    """Known-properties-pipeline entry point (the ``*-autoprove-properties``
    commands)."""
    parser = argparse.ArgumentParser(
        description="Auto-prove pipeline for a known list of properties"
    )
    add_common_autoprove_args(parser)
    parser.add_argument("--properties", required=True, help="Path to a YAML file listing the properties to prove")
    parser.add_argument("--skip-setup", action="store_true", help="Skip autosetup AND harness creation; supply --conf and --summary instead")
    parser.add_argument("--conf", default=None, help="Path to a prover .conf file (required with --skip-setup)")
    parser.add_argument("--summary", default=None, help="Path to a single summary spec, certora/-relative (required with --skip-setup)")
    # This pipeline has no --threat-model; satisfy CommonAutoProveArgs.threat_model.
    parser.set_defaults(threat_model=None)

    args = cast(PropertiesArgs, parser.parse_args())

    # Eager validation — before any LLM work.
    if args.skip_setup and (args.conf is None or args.summary is None):
        parser.error("--skip-setup requires both --conf and --summary")
    try:
        known = load_known_properties(pathlib.Path(args.properties))
    except KnownPropertiesError as exc:
        parser.error(str(exc))

    async with _autoprove_services(parser, args, summary) as svc:
        async def runner(handler: HandlerFactory[AutoProvePhase, None]) -> AutoProveResult:
            return await run_properties_pipeline(
                llm=svc.llm,
                ctx=svc.ctx,
                source_input=svc.source,
                env=svc.source_env,
                handler_factory=handler,
                known_properties=known,
                prover_opts=svc.prover_opts,
                max_concurrent=args.max_concurrent,
                skip_setup=args.skip_setup,
                conf_path=args.conf,
                summary_path=args.summary,
            )

        yield runner
