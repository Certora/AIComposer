"""
NatSpec multi-agent pipeline orchestration.

Replaces the monolithic natspec workflow with a multi-agent pipeline:
1. Component analysis (single agent)
2. Per-component property extraction (parallel)
3. Interface generation (single agent)
4. Initial stub generation (single agent)
5. Per-property CVL generation (parallel, semaphore-bounded) with merge

This is a plain asyncio orchestrator, not a LangGraph graph.

Every top-level agent invocation is wrapped in a per-task ``with_handler``
created by the caller-provided ``HandlerFactory``.  The TUI uses these to
populate a summary panel (collapsible by phase) with drill-down into
individual task event streams.
"""

import asyncio
import hashlib
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
from composer.spec.component import (
    ApplicationComponent,
    ComponentInst,
    run_component_analysis,
)
from composer.spec.bug import run_bug_analysis
from composer.spec.prop import PropertyFormulation
from composer.spec.interface_gen import generate_interface
from composer.spec.stub_gen import generate_stub
from composer.spec.registry import StubRegistry
from composer.spec.merge import make_publish_tools, make_advisory_typecheck_tool
from composer.spec.cvl_generation import GenerationEnv, GeneratedCVL, generate_property_cvl


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


type HandlerFactory = Callable[[TaskInfo], tuple[IOHandler[Any, Any], EventHandler]]


# ---------------------------------------------------------------------------
# Cache key helpers  (mirrors auto-prover's hash-based approach)
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[None, Properties]("properties")


def _string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _component_cache_key(
    component: ApplicationComponent,
    app_type: str,
) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([component.model_dump_json(), app_type])
    return CacheKey(_string_hash(combined))


def _property_cache_key(prop: PropertyFormulation) -> CacheKey[ComponentGroup, GeneratedCVL]:
    return CacheKey(_string_hash(prop.model_dump_json()))


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
                  └── <property-hash> [GeneratedCVL] → abstract(CVLGeneration)

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
    base_builder: PlainBuilder | CVLOnlyBuilder = builders.cvl_only
    analysis_input = (system_doc, base_builder)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis
    # ------------------------------------------------------------------
    summary = await _run_task(
        handler_factory,
        TaskInfo("component-analysis", "Component analysis", "component_analysis"),
        lambda: run_component_analysis(ctx, analysis_input),
    )
    if summary is None:
        raise ValueError("Component analysis produced no result — is the system doc empty?")

    # ------------------------------------------------------------------
    # Phase 3: Interface generation
    # ------------------------------------------------------------------
    interface = await _run_task(
        handler_factory,
        TaskInfo("interface-gen", "Interface generation", "interface_gen"),
        lambda: generate_interface(ctx, summary, system_doc, base_builder, solc_version),
    )

    # ------------------------------------------------------------------
    # Phase 4: Initial stub generation
    # ------------------------------------------------------------------
    initial_stub = await _run_task(
        handler_factory,
        TaskInfo("stub-gen", "Stub generation", "stub_gen"),
        lambda: generate_stub(ctx, interface, contract_name, base_builder, solc_version),
    )

    # ------------------------------------------------------------------
    # Shared artifacts for Phase 5
    # ------------------------------------------------------------------
    master_spec = SharedArtifact.create(
        store, MASTER_SPEC_NS, "master", initial_content="",
    )
    registry = StubRegistry.create(
        store, STUB_NS, base_builder, ctx, interface, initial_stub, solc_version,
    )

    # ------------------------------------------------------------------
    # Phase 2 + 5:  Per-component extraction → per-property CVL gen
    # ------------------------------------------------------------------

    prop_context = ctx.child(PROPERTIES_KEY)

    results: list[GeneratedCVL] = []
    failures: list[PropertyFailure] = []

    async def _analyze_component(
        component_idx: int,
    ) -> list[tuple[PropertyFormulation, ComponentInst, WorkflowContext[ComponentGroup]]]:
        feat = ComponentInst(summ=summary, ind=component_idx)
        name = feat.component.name
        feat_ctx = prop_context.child(
            _component_cache_key(feat.component, summary.application_type),
            {
                "component": feat.component.model_dump(),
                "app_type": summary.application_type,
            },
        )

        async def _extract() -> list[PropertyFormulation] | None:
            async with semaphore:
                return await run_bug_analysis(feat_ctx, feat, analysis_input)

        props = await _run_task(
            handler_factory,
            TaskInfo(f"bug-{component_idx}", name, "bug_analysis"),
            _extract,
        )

        if props is None:
            return []
        return [(p, feat, feat_ctx) for p in props]

    extraction_results = await asyncio.gather(*[
        _analyze_component(i) for i in range(len(summary.components))
    ])

    all_properties: list[tuple[PropertyFormulation, ComponentInst, WorkflowContext[ComponentGroup]]] = []
    for batch in extraction_results:
        all_properties.extend(batch)

    if not all_properties:
        raise ValueError("No properties extracted from any component.")

    # Phase 5: per-property CVL generation, nested under component contexts
    async def _generate_one(
        prop_idx: int,
        prop: PropertyFormulation,
        feat: ComponentInst,
        feat_ctx: WorkflowContext[ComponentGroup],
    ) -> GeneratedCVL:
        prop_ctx = feat_ctx.child(
            _property_cache_key(prop),
            prop.model_dump(),
        ).abstract(CVLGeneration)

        stub_tools = registry.get_tools()
        typecheck_tool = make_advisory_typecheck_tool(
            registry.read_stub, interface, contract_name, solc_version,
        )
        publish, give_up = make_publish_tools(
            master_spec, registry.read_stub, interface,
            contract_name, solc_version, base_builder, prop_ctx,
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
            result_tools=(publish, give_up),
        )

        async def _gen() -> GeneratedCVL:
            async with semaphore:
                return await generate_property_cvl(
                    prop_ctx, prop, feat, env, with_memory=False,
                )

        label = f"{feat.component.name}: {prop.description[:50]}"
        return await _run_task(
            handler_factory,
            TaskInfo(f"cvl-{prop_idx}", label, "cvl_gen"),
            _gen,
        )

    generation_results = await asyncio.gather(
        *[
            _generate_one(i, prop, feat, feat_ctx)
            for i, (prop, feat, feat_ctx) in enumerate(all_properties)
        ],
        return_exceptions=True,
    )

    for (prop, _, _), result in zip(all_properties, generation_results):
        if isinstance(result, BaseException):
            failures.append(PropertyFailure(prop=prop, reason=str(result)))
        elif isinstance(result, GeneratedCVL):
            if result.commentary.startswith("GAVE_UP:"):
                reason = result.commentary.removeprefix("GAVE_UP:").strip()
                failures.append(PropertyFailure(prop=prop, reason=reason))
            else:
                results.append(result)

    # Read final master spec and stub
    final_spec, _ = await master_spec.read()
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
) -> T:
    """Run ``fn`` inside a per-task ``with_handler`` from the factory."""
    h, eh = factory(info)
    async with with_handler(h, eh):
        return await fn()
