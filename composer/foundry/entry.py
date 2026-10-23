"""CLI entry-point wiring for the foundry test author.

Mirrors ``composer/spec/source/autoprove_common.py``'s ``_entry_point``
shape (parse args → set up DB / RAG / store / checkpointer / logging /
thread logger → yield a closure the caller drives with a handler factory)
but plugs in foundry-specific pieces: the foundry RAG database connection
(distinct from the CVL manual RAG), the foundry env builder, and the
foundry pipeline.

Deliberately self-contained — does NOT import or modify anything under
``composer/spec/source/``. Reuses the cross-workflow infrastructure
(``standard_connections``, ``thread_logger``, ``WorkflowContext``,
``run_component_analysis`` / ``run_property_inference`` from the
non-source spec modules).
"""

import argparse
import hashlib
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Awaitable, Callable, Protocol, cast

from graphcore.tools.memory import async_memory_tool

from composer.core.user import user_data_ns
from composer.diagnostics.logging_setup import setup_autoprove_logging
from composer.diagnostics.timing import RunSummary, install_run_summary
from composer.input.parsing import Arg, add_protocol_args
from composer.input.types import DEFAULT_RECURSION_LIMIT, TieredModelOptions, RAGDBOptions
from composer.io.multi_job import HandlerFactory
from composer.io.thread_logging import DEFAULT_META_NS, thread_logger, default_logging_ns
from composer.kb.knowledge_base import DefaultEmbedder
from composer.rag.db import FOUNDRY_DEFAULT_CONNECTION, PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.spec.context import WorkflowContext
from composer.spec.system_model import SolidityIdentifier
from composer.spec.service_host import ModelProvider
from composer.spec.util import FS_FORBIDDEN_READ
from composer.ui.tool_display import async_tool_context
from composer.workflow.services import standard_connections, llm_factory

from composer.foundry.artifacts import FoundrySourceCode
from composer.foundry.env import build_foundry_env
from composer.foundry.pipeline import (
    FoundryPhase, FoundryPipelineResult, run_foundry_pipeline,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


class FoundryRAGDBOptions(RAGDBOptions, Protocol):
    """Overrides ``RAGDBOptions.rag_db``'s default so ``--rag-db`` points
    at the foundry cheatcodes database (not the CVL manual one). The
    foundry workflow only talks to a single RAG DB; we reuse the existing
    ``--rag-db`` flag rather than introducing a separate one."""
    rag_db: Annotated[str, Arg(
        help="Database connection string for the foundry cheatcodes RAG",
        default=FOUNDRY_DEFAULT_CONNECTION,
    )]


class FoundryArgs(TieredModelOptions, FoundryRAGDBOptions, Protocol):
    project_root: str
    main_contract: str
    system_doc: str
    max_concurrent: int
    cache_ns: str | None
    memory_ns: str | None
    interactive: bool
    max_bug_rounds: int
    recursion_limit: int
    forge_binary: str
    forge_timeout_s: int
    max_forge_runners: int


def _user_ns(*parts: str | tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        else:
            out.extend(p)
    return user_data_ns() + tuple(out)


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
# Entry point
# ---------------------------------------------------------------------------


type FoundryRunner = Callable[
    [HandlerFactory[FoundryPhase, None]], Awaitable[FoundryPipelineResult],
]


@asynccontextmanager
async def _entry_point(summary: RunSummary) -> AsyncIterator[FoundryRunner]:
    parser = argparse.ArgumentParser(
        description="Foundry-test author for a property-extraction pipeline",
    )
    add_protocol_args(parser, FoundryRAGDBOptions)
    add_protocol_args(parser, TieredModelOptions)
    parser.add_argument(
        "--recursion-limit", type=int, default=DEFAULT_RECURSION_LIMIT,
        help=f"Max graph iterations (default: {DEFAULT_RECURSION_LIMIT})",
    )
    parser.add_argument("project_root", help="Foundry project root (contains foundry.toml).")
    parser.add_argument("main_contract", help="Main contract as path:ContractName")
    parser.add_argument("system_doc", help="Path to the design document (text or PDF)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent agents (default: 4)")
    parser.add_argument("--max-forge-runners", default=1, type=int, help="Max concurrent forge runners (default: 1)")
    parser.add_argument("--cache-ns", default=None, help="Cache namespace (enables cross-run caching)")
    parser.add_argument("--memory-ns", default=None, help="Memory namespace (default: thread id)")
    parser.add_argument("--interactive", action="store_true", help="Interactively refine extracted properties")
    parser.add_argument("--max-bug-rounds", type=int, default=3, help="Max bug-extraction rounds per component (default: 3)")
    parser.add_argument("--forge-binary", default="forge", help="`forge` executable on PATH (default: forge)")
    parser.add_argument("--forge-timeout-s", type=int, default=600, help="Per-`forge test` invocation timeout in seconds (default: 600)")

    args = cast(FoundryArgs, parser.parse_args())

    project_root = pathlib.Path(args.project_root).resolve()
    if not (project_root / "foundry.toml").is_file():
        parser.error(f"{project_root}/foundry.toml not found — not a foundry project")
    main_path, contract_name = args.main_contract.split(":", 1)
    contract_name = SolidityIdentifier(contract_name)
    full_path = pathlib.Path(main_path).resolve()
    if not full_path.is_relative_to(project_root):
        parser.error(f"Invalid path: {full_path} doesn't appear in project root {project_root}")
    relative_path = str(full_path.relative_to(project_root))

    sys_path = pathlib.Path(args.system_doc)

    model = get_model()

    root_key = _root_cache_key(str(project_root), sys_path, relative_path, contract_name)
    cache_root: tuple[str, ...] | None = (
        _user_ns(args.cache_ns, root_key) if args.cache_ns is not None else None
    )

    thread_id = f"foundry_{uuid.uuid4().hex[:12]}"
    text_log, events_log = setup_autoprove_logging(project_root, thread_id)
    print(f"foundry logs: {text_log}\n         events: {events_log}", file=sys.stderr)
    install_run_summary(summary)

    model_fact = llm_factory(args)

    async with (
        standard_connections(embedder=DefaultEmbedder(model)) as conns,
        PostgreSQLRAGDatabase.rag_context(model, args.rag_db) as foundry_rag_db,
        async_tool_context(),
        thread_logger(
            conns.store,
            {
                "root_thread_id": thread_id,
                "workflow": "foundry",
                # Resolved cache root (null when caching is disabled) and
                # effective memory namespace, so run-trail tooling can find
                # this run's cache/memory entries without reverse-engineering
                # namespaces from thread ids.
                "cache_root": list(cache_root) if cache_root is not None else None,
                "memory_ns": args.memory_ns if args.memory_ns is not None else thread_id,
            },
            default_logging_ns(uid=None),
            run_id=summary.run_id,
        ),
    ):
        content = await conns.uploader.get_document(sys_path)
        if content is None:
            parser.error(f"cannot read {sys_path}")
        source_input = FoundrySourceCode(
            content=content,
            project_root=str(project_root),
            contract_name=contract_name,
            relative_path=relative_path,
            forbidden_read=FS_FORBIDDEN_READ,
        )

        model_provider = ModelProvider(
            checkpointer=conns.checkpointer,
            factory=model_fact,
            heavy_model=args.heavy_model,
            lite_model=args.lite_model
        )

        # Per-user cache namespace for the indexed code_explorer's question
        # cache (mirrors what autoprove_common does for the source_question_ns).
        source_question_ns = _user_ns("source_agent", "cache", root_key)

        env = build_foundry_env(
            model_provider=model_provider,
            project_root=str(project_root),
            forbidden_read=FS_FORBIDDEN_READ,
            rag_db=foundry_rag_db,
            store=conns.indexed_store,
            source_question_ns=source_question_ns,
            recursion_limit=args.recursion_limit,
        )

        memory_ns = args.memory_ns
        ctx = WorkflowContext.create(
            services=lambda ns: async_memory_tool(conns.memory(ns)),
            thread_id=thread_id,
            store=conns.store,
            recursion_limit=args.recursion_limit,
            cache_namespace=cache_root,
            memory_namespace=memory_ns,
        )

        async def runner(
            handler: HandlerFactory[FoundryPhase, None],
        ) -> FoundryPipelineResult:
            return await run_foundry_pipeline(
                source_input=source_input,
                ctx=ctx,
                handler_factory=handler,
                env=env,
                max_concurrent=args.max_concurrent,
                max_bug_rounds=args.max_bug_rounds,
                interactive=args.interactive,
                forge_binary=args.forge_binary,
                forge_timeout_s=args.forge_timeout_s,
                forge_concurrency=args.max_forge_runners
            )

        yield runner
