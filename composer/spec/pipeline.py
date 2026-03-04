"""
NatSpec multi-agent pipeline orchestration.

Replaces the monolithic natspec workflow with a multi-agent pipeline:
1. Component analysis (single agent)
2. Per-component property extraction (parallel)
3. Interface generation (single agent)
4. Initial stub generation (single agent)
5. Per-component batch CVL generation (parallel, semaphore-bounded) with merge

This is a plain asyncio orchestrator, not a LangGraph graph.

Every top-level agent invocation is wrapped in a per-task ``with_handler``
created by the caller-provided ``HandlerFactory``.  The TUI uses these to
populate a summary panel (collapsible by phase) with drill-down into
individual task event streams.
"""

import asyncio
import traceback
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.store.base import BaseStore

from composer.io.context import with_handler
from composer.io.protocol import IOHandler
from composer.io.event_handler import EventHandler

from composer.spec.cas import SharedArtifact
from composer.spec.context import (
    WorkflowContext, Builders,
    SystemDoc, PlainBuilder, CVLOnlyBuilder,
    CacheKey, Properties, ComponentGroup, CVLGeneration,
)
from composer.spec.util import string_hash
from composer.spec.component import (
    ApplicationComponent,
    ComponentInst,
    run_component_analysis,
    DESCRIPTION as COMPONENT_ANALYSIS_DESC,
)
from composer.spec.bug import run_bug_analysis
from composer.spec.prop import PropertyFormulation
from composer.spec.interface_gen import generate_interface, DESCRIPTION as INTERFACE_GEN_DESC
from composer.spec.stub_gen import generate_stub, DESCRIPTION as STUB_GEN_DESC, STUB_KEY
from composer.spec.registry import StubRegistry
from composer.spec.merge import make_publish_tools, make_advisory_typecheck_tool
from composer.spec.cvl_generation import GenerationEnv, GeneratedCVL, generate_batch_cvl


# ---------------------------------------------------------------------------
# Handler factory types
# ---------------------------------------------------------------------------

type Phase = Literal[
    "component_analysis",
    "bug_analysis",
    "interface_gen",
    "stub_gen",
    "cvl_gen",
]


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    label: str
    phase: Phase


@dataclass(frozen=True)
class TaskHandle:
    """Returned by the handler factory — bundles IO with lifecycle callbacks."""
    handler: IOHandler[Any, Any]
    event_handler: EventHandler
    on_error: Callable[[Exception, str], Awaitable[None]]
    on_start: Callable[[], None] = lambda: None
    on_done: Callable[[], None] = lambda: None


type HandlerFactory = Callable[[TaskInfo], Awaitable[TaskHandle]]


# ---------------------------------------------------------------------------
# Cache key helpers  (mirrors auto-prover's hash-based approach)
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[None, Properties]("properties")

def _component_cache_key(
    component: ApplicationComponent,
    app_type: str,
) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([component.model_dump_json(), app_type])
    return CacheKey(string_hash(combined))


def _batch_cache_key(props: list[PropertyFormulation]) -> CacheKey[ComponentGroup, GeneratedCVL]:
    combined = "|".join(p.model_dump_json() for p in props)
    return CacheKey(string_hash(combined))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PropertyFailure:
    prop: PropertyFormulation
    reason: str


@dataclass
class PipelineResult:
    interface: str
    stub: str
    spec: str
    contract_name: str
    solc_version: str
    failures: list[PropertyFailure] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

MASTER_SPEC_NS = ("natspec_pipeline", "master_spec")
STUB_NS = ("natspec_pipeline", "stub")


async def run_natspec_pipeline(
    system_doc: SystemDoc,
    contract_name: str,
    solc_version: str,
    builders: Builders,
    ctx: WorkflowContext[None],
    store: BaseStore,
    handler_factory: HandlerFactory,
    *,
    max_concurrent: int = 4,
) -> PipelineResult:
    """Run the full natspec multi-agent pipeline.

    Every agent invocation is wrapped in a per-task ``with_handler``
    obtained from ``handler_factory``.  The TUI can group tasks by
    ``TaskInfo.phase`` into collapsible sections.

    Cache hierarchy mirrors auto-prover::

        root [None]
          └── properties [Properties]
              └── <component-hash> [ComponentGroup]
                  ├── bug_analysis (internal to bug.py)
                  └── <batch-hash> [GeneratedCVL] → abstract(CVLGeneration)

    Args:
        system_doc: The design document for the application.
        contract_name: The expected contract name.
        solc_version: Solidity compiler version (e.g., "8.21").
        builders: Pre-configured builder variants.
        ctx: Root workflow context.
        store: BaseStore for shared artifacts and caching.
        handler_factory: Creates per-task ``(IOHandler, EventHandler)``
            pairs.  Called once per top-level agent invocation.
        max_concurrent: Maximum concurrent LLM agents.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    plain_builder = builders.source
    cvl_builder: PlainBuilder | CVLOnlyBuilder = builders.cvl_only
    analysis_input = (system_doc, plain_builder)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis
    # ------------------------------------------------------------------
    summary = await _run_task(
        handler_factory,
        TaskInfo("component-analysis", COMPONENT_ANALYSIS_DESC, "component_analysis"),
        lambda: run_component_analysis(ctx, analysis_input),
    )
    if summary is None:
        raise ValueError("Component analysis produced no result — is the system doc empty?")

    # ------------------------------------------------------------------
    # Phase 3: Interface generation
    # ------------------------------------------------------------------
    interface = await _run_task(
        handler_factory,
        TaskInfo("interface-gen", INTERFACE_GEN_DESC, "interface_gen"),
        lambda: generate_interface(ctx, summary, system_doc, plain_builder, solc_version),
    )

    # ------------------------------------------------------------------
    # Phase 4: Initial stub generation
    # ------------------------------------------------------------------
    initial_stub = await _run_task(
        handler_factory,
        TaskInfo("stub-gen", STUB_GEN_DESC, "stub_gen"),
        lambda: generate_stub(ctx, interface, contract_name, plain_builder, solc_version),
    )

    # ------------------------------------------------------------------
    # Shared artifacts for Phase 5
    # ------------------------------------------------------------------
    master_spec = SharedArtifact.create(
        store, MASTER_SPEC_NS, "master", initial_content="",
    )
    registry = StubRegistry.create(
        store, STUB_NS, cvl_builder, ctx, interface, initial_stub, solc_version,
    )

    # ------------------------------------------------------------------
    # Phase 2 + 5:  Per-component extraction → per-component batch CVL gen
    # ------------------------------------------------------------------

    prop_context = ctx.child(PROPERTIES_KEY)

    results: list[GeneratedCVL] = []
    failures: list[PropertyFailure] = []

    # Phase 2: per-component property extraction
    @dataclass
    class _ComponentBatch:
        feat: ComponentInst
        props: list[PropertyFormulation]
        feat_ctx: WorkflowContext[ComponentGroup]

    async def _analyze_component(component_idx: int) -> _ComponentBatch | None:
        feat = ComponentInst(summ=summary, ind=component_idx)
        name = feat.component.name
        feat_ctx = prop_context.child(
            _component_cache_key(feat.component, summary.application_type),
            {
                "component": feat.component.model_dump(),
                "app_type": summary.application_type,
            },
        )

        props = await _run_task(
            handler_factory,
            TaskInfo(f"bug-{component_idx}", name, "bug_analysis"),
            lambda: run_bug_analysis(feat_ctx, feat, analysis_input),
            semaphore,
        )

        if not props:
            return None
        return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)

    extraction_results = await asyncio.gather(*[
        _analyze_component(i) for i in range(len(summary.components))
    ])

    component_batches = [b for b in extraction_results if b is not None]

    if not component_batches:
        raise ValueError("No properties extracted from any component.")

    # Phase 5: per-component batch CVL generation
    async def _generate_batch(
        batch_idx: int,
        batch: _ComponentBatch,
    ) -> GeneratedCVL:
        batch_ctx = batch.feat_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        ).abstract(CVLGeneration)

        stub_tools = registry.get_tools()
        typecheck_tool = make_advisory_typecheck_tool(
            registry.read_stub, interface, contract_name, solc_version,
        )

        stub_content = registry.read_stub()
        env = GenerationEnv(
            input=system_doc,
            builders=builders,
            extra_tools=[*stub_tools, typecheck_tool],
            extra_input=[
                "The current verification stub is:",
                stub_content,
            ],
            result_tools=lambda validator: make_publish_tools(
                master_spec, registry.read_stub, interface,
                contract_name, solc_version, cvl_builder, batch_ctx,
                validator=validator,
            ),
        )

        label = f"{batch.feat.component.name} ({len(batch.props)} properties)"
        return await _run_task(
            handler_factory,
            TaskInfo(f"cvl-{batch_idx}", label, "cvl_gen"),
            lambda: generate_batch_cvl(
                batch_ctx, batch.props, batch.feat, env, with_memory=True,
                description=label,
            ),
            semaphore,
        )

    generation_results = await asyncio.gather(
        *[
            _generate_batch(i, batch)
            for i, batch in enumerate(component_batches)
        ],
        return_exceptions=True,
    )

    for batch, result in zip(component_batches, generation_results):
        if isinstance(result, BaseException):
            for prop in batch.props:
                failures.append(PropertyFailure(prop=prop, reason=str(result)))
        elif isinstance(result, GeneratedCVL):
            if result.commentary.startswith("GAVE_UP:"):
                reason = result.commentary.removeprefix("GAVE_UP:").strip()
                for prop in batch.props:
                    failures.append(PropertyFailure(prop=prop, reason=reason))
            else:
                results.append(result)
                # Record skipped properties as failures
                for skip in result.skipped:
                    if skip.property_index in range(1, len(batch.props) + 1):
                        failures.append(PropertyFailure(
                            prop=batch.props[skip.property_index - 1],
                            reason=f"Skipped: {skip.reason}",
                        ))

    # Read final master spec and stub
    final_spec = master_spec.read_unsync() or ""
    final_stub = registry.read_stub()

    return PipelineResult(
        interface=interface,
        stub=final_stub,
        spec=final_spec,
        contract_name=contract_name,
        solc_version=solc_version,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_task[T](
    factory: HandlerFactory,
    info: TaskInfo,
    fn: Callable[[], Awaitable[T]],
    semaphore: asyncio.Semaphore | None = None,
) -> T:
    """Run ``fn`` inside a per-task ``with_handler`` from the factory.

    If *semaphore* is provided the task is created immediately (PENDING)
    but only transitions to RUNNING (via ``handle.on_start``) once the
    semaphore is acquired.
    """
    handle = await factory(info)
    try:
        if semaphore is not None:
            async with semaphore:
                handle.on_start()
                async with with_handler(handle.handler, handle.event_handler):
                    result = await fn()
        else:
            handle.on_start()
            async with with_handler(handle.handler, handle.event_handler):
                result = await fn()
    except Exception as exc:
        await handle.on_error(exc, traceback.format_exc())
        raise
    else:
        handle.on_done()
        return result
