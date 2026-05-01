"""Entry point for the auto-prove multi-agent pipeline TUI.

Two public surfaces:

  - :func:`run_autoprove` opens the service contexts (Postgres, rag,
    tool-display scope) and runs the pipeline against a supplied
    handler factory. CLI and non-CLI callers (web frontend) both go
    through this; the CLI layer just builds the inputs from argparse.
  - :func:`_entry_point` is the CLI plumbing — argparse + build inputs +
    call ``run_autoprove`` via the existing ``cb`` callback pattern.
"""

import argparse
import hashlib
import pathlib
import uuid
from dataclasses import dataclass
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
    max_bug_rounds: int


@dataclass
class AutoProveInputs:
    """Resolved inputs for an autoprove run.

    Built from CLI argparse for ``tui-autoprove`` / ``console-autoprove``,
    or from form data for the web frontend. Flat by design — satisfies
    :class:`ModelOptions` / :class:`RAGDBOptions` Protocols by attribute
    shape so ``create_llm(inputs)`` can use it directly.
    """
    # Resolved paths — caller is expected to have validated existence
    # and that ``main_contract_path`` is inside ``project_root``.
    project_root: pathlib.Path
    main_contract_path: pathlib.Path
    contract_name: str
    system_doc_path: pathlib.Path
    threat_model_path: pathlib.Path | None = None

    # Run config
    max_concurrent: int = 4
    cloud: bool = False
    interactive: bool = False
    cache_ns: str | None = None
    memory_ns: str | None = None
    max_bug_rounds: int = 3

    # ModelOptions + RAGDBOptions (consumed by ``create_llm``)
    rag_db: str = ""
    model: str = "claude-opus-4-6"
    tokens: int = 10_000
    thinking_tokens: int = 2048
    memory_tool: bool = True
    interleaved_thinking: bool = False


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
# Reusable entry point
# ---------------------------------------------------------------------------

async def run_autoprove(
    inputs: AutoProveInputs,
    *,
    handler_factory: HandlerFactory[AutoProvePhase, None],
    thread_id: str | None = None,
) -> AutoProveResult:
    """Open service contexts and run the autoprove pipeline.

    *thread_id* is the langgraph root thread id for this run. Pass
    ``None`` (the default) for a fresh UUID. Pass an existing thread
    id to resume from its checkpoints — Phase 5's "soft retry" uses
    this to re-enter a failing run with the same agent state, while
    "hard retry" passes ``None`` (or a fresh id) so cached phases
    skip but the failing phase gets a clean conversation.
    """
    relative_path = str(inputs.main_contract_path.relative_to(inputs.project_root))
    content = get_document_input(inputs.system_doc_path)
    if content is None:
        raise ValueError(f"cannot read system doc at {inputs.system_doc_path}")

    system_doc = SourceCode(
        content=content,
        project_root=str(inputs.project_root),
        contract_name=inputs.contract_name,
        relative_path=relative_path,
        forbidden_read=FS_FORBIDDEN_READ,
    )

    llm = create_llm(inputs)
    model = get_model()

    root_key = _root_cache_key(
        str(inputs.project_root), inputs.system_doc_path, relative_path, inputs.contract_name,
    )
    cache_root: tuple[str, str] | None = (
        (inputs.cache_ns, root_key) if inputs.cache_ns is not None else None
    )

    if thread_id is None:
        thread_id = f"autoprove_{uuid.uuid4().hex[:12]}"

    threat_model = (
        get_document_input(inputs.threat_model_path)
        if inputs.threat_model_path is not None else None
    )

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
            root=str(inputs.project_root),
            store=conns.indexed_store,
            cvl_cache_ns=DEFAULT_CVL_AGENT_INDEX_NS,
            source_question_ns=("source_agent", "cache", root_key)
        )
        ctx = WorkflowContext.create(
            services=lambda namespace: async_memory_tool(conns.memory(namespace)),
            thread_id=thread_id,
            store=conns.store,
            cache_namespace=cache_root,
            memory_namespace=inputs.memory_ns,
        )
        return await run_autoprove_pipeline(
            llm=llm,
            ctx=ctx,
            source_input=system_doc,
            env=source_env,
            handler_factory=handler_factory,
            cloud=CloudConfig() if inputs.cloud else None,
            max_concurrent=inputs.max_concurrent,
            interactive=inputs.interactive,
            threat_model=threat_model,
            max_bug_rounds=inputs.max_bug_rounds,
        )


# ---------------------------------------------------------------------------
# CLI plumbing
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
    parser.add_argument("--max-bug-rounds", type=int, default=3, help="Maximum number of bug-extraction rounds run per component during property analysis (default: 3)")

    args = cast(AutoProveArgs, parser.parse_args())

    project_root = pathlib.Path(args.project_root).resolve()
    main_contract_path_str, contract_name = args.main_contract.split(":", 1)
    main_contract_path = pathlib.Path(main_contract_path_str).resolve()
    if not main_contract_path.is_relative_to(project_root):
        print(
            f"Invalid path: {main_contract_path} doesn't appear in project root {project_root}"
        )
        return 1

    sys_path = pathlib.Path(args.system_doc)
    if get_document_input(sys_path) is None:
        # ``run_autoprove`` would raise; surface the same friendly
        # message we used to print here before the refactor.
        print(f"Error: cannot read {sys_path}")
        return 1

    inputs = AutoProveInputs(
        project_root=project_root,
        main_contract_path=main_contract_path,
        contract_name=contract_name,
        system_doc_path=sys_path,
        threat_model_path=(
            pathlib.Path(args.threat_model) if args.threat_model is not None else None
        ),
        max_concurrent=args.max_concurrent,
        cloud=args.cloud,
        interactive=args.interactive,
        cache_ns=args.cache_ns,
        memory_ns=args.memory_ns,
        max_bug_rounds=args.max_bug_rounds,
        rag_db=args.rag_db,
        model=args.model,
        tokens=args.tokens,
        thinking_tokens=args.thinking_tokens,
        memory_tool=args.memory_tool,
        interleaved_thinking=args.interleaved_thinking,
    )

    async def runner(handler: HandlerFactory[AutoProvePhase, None]) -> AutoProveResult:
        return await run_autoprove(inputs, handler_factory=handler)
    return await cb(runner)
