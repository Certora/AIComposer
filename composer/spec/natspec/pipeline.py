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
from dataclasses import dataclass, field
from typing import Literal, Awaitable

from typing_extensions import TypedDict

from langgraph.store.base import BaseStore

from composer.ui.multi_job_app import (
    TaskInfo, HandlerFactory, run_task,
)

from composer.spec.natspec.cas import SharedArtifact
from composer.spec.context import (
    WorkflowContext,
    SystemDoc, PlainBuilder, CVLOnlyBuilder,
    CacheKey, Properties, ComponentGroup, CVLGeneration,
    Contract
)
from composer.spec.util import string_hash
from composer.spec.bug import run_bug_analysis
from composer.spec.prop import PropertyFormulation
from composer.spec.natspec.interface_gen import generate_interface, DESCRIPTION as INTERFACE_GEN_DESC, InterfaceResult, InterfaceDecl
from composer.spec.natspec.stub_gen import generate_stub, StubDeclaration
from composer.spec.natspec.registry import StubRegistry
from composer.spec.natspec.merge import make_publish_tools, make_advisory_typecheck_tool
from composer.spec.cvl_generation import GenerationEnv, GeneratedCVL, generate_batch_cvl
from composer.spec.gen_types import TemplateInstantiation, TypedTemplate, GenerationPrompt
from composer.spec.natspec.system_analysis import run_component_analysis, DESCRIPTION as SYSTEM_DESC
from composer.spec.system_model import ContractInstance, ContractComponentInstance, ContractComponent, Application


# ---------------------------------------------------------------------------
# Phase type
# ---------------------------------------------------------------------------

type Phase = Literal[
    "component_analysis",
    "bug_analysis",
    "interface_gen",
    "stub_gen",
    "cvl_gen",
]


# ---------------------------------------------------------------------------
# Cache key helpers  (mirrors auto-prover's hash-based approach)
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[Contract, Properties]("properties")

def _component_cache_key(
    component: ContractComponent,
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
class ContractFormulation:
    interface: InterfaceDecl
    stub: StubDeclaration
    name: str
    failures: list[PropertyFailure]
    spec: str

@dataclass
class PipelineResult:
    app: Application
    contracts: list[ContractFormulation] = field(default_factory=list)



# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

MASTER_SPEC_NS = ("natspec_pipeline", "master_spec")
STUB_NS = ("natspec_pipeline", "stub")

@dataclass
class ContractResult:
    spec: str
    failures: list[PropertyFailure] = field(default_factory=list)

@dataclass
class PipelineServices:
    sem: asyncio.Semaphore
    store: BaseStore
    factory: HandlerFactory[Phase, None]
    builder: PlainBuilder
    cvl_research: CVLOnlyBuilder

class NatspecGenerationParams(TypedDict):
    context: ContractComponentInstance

class FeedbackPromptParams(TypedDict):
    context: ContractComponentInstance
    has_source: bool

NoSourceGenerationPrompt = TypedTemplate[NatspecGenerationParams]("nosource_property_generation_prompt.j2")
FeedbackPrompt = TypedTemplate[FeedbackPromptParams]("property_judge_prompt.j2")

async def analyze_single_contract(
    system_doc: SystemDoc,
    ctx: WorkflowContext[Contract],
    services: PipelineServices,
    solc_version: str,
    intf: InterfaceResult,
    summary: ContractInstance,
    stub_registry: StubRegistry,
    stub: StubDeclaration
) -> ContractResult:
    
    contract_name = summary.contract.name
    handler_factory = services.factory
    store = services.store
    semaphore = services.sem
    interface = intf

    doc_digest = string_hash(str(system_doc.content))
    
    # ------------------------------------------------------------------
    # Shared artifacts for Phase 5
    # ------------------------------------------------------------------
    master_spec = SharedArtifact.create(
        store, MASTER_SPEC_NS + (doc_digest,), summary.contract.name, initial_content="",
    )
    registry = stub_registry

    # ------------------------------------------------------------------
    # Phase 2 + 5:  Per-component extraction → per-component batch CVL gen
    # ------------------------------------------------------------------

    prop_context = ctx.child(PROPERTIES_KEY)

    analysis_input = (system_doc, services.builder)

    results: list[GeneratedCVL] = []
    failures: list[PropertyFailure] = []

    # Phase 2: per-component property extraction
    @dataclass
    class _ComponentBatch:
        feat: ContractComponentInstance
        props: list[PropertyFormulation]
        feat_ctx: WorkflowContext[ComponentGroup]

    async def _analyze_component(component_idx: int) -> _ComponentBatch | None:
        feat = ContractComponentInstance(_contract=summary, ind=component_idx)
        name = f"{feat.contract.name}: {feat.component.name}"
        feat_ctx = prop_context.child(
            _component_cache_key(feat.component, summary.app.application_type),
            {
                "component": feat.component.model_dump(),
                "app_type": summary.app.application_type,
            },
        )

        props = await run_task(
            handler_factory,
            TaskInfo(f"bug-{summary.contract.name}-{component_idx}", name, "bug_analysis"),
            lambda: run_bug_analysis(feat_ctx, feat, analysis_input),
            semaphore,
        )

        if not props:
            return None
        return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)

    extraction_results = await asyncio.gather(*[
        _analyze_component(i) for i in range(len(summary.contract.components))
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
        stub_tools = registry.get_tools(contract_name)
        typecheck_tool = make_advisory_typecheck_tool(
            lambda: registry.read_stub(contract_name), interface, stub.solidity_identifier, solc_version,
        )

        publish = make_publish_tools(
            master_spec=master_spec,
            stub_read=lambda: registry.read_stub(contract_name),
            interface=interface,
            builder=services.cvl_research,
            ctx=batch_ctx,
            contract_id=stub.solidity_identifier,
            solc_version=solc_version
        )

        prompt_extras = [
            f"The current stub implementation of the {contract_name} is",
            stub_registry.read_stub(contract_name)
        ]

        prompt_extras.append("The interface of the contract containing this component is")
        prompt_extras.append(interface.name_to_interface[contract_name].content)

        for sib in batch.feat.ommer_contract:
            prompt_extras.extend([
                f"The interface of the {sib.name} contract is:",
                interface.name_to_interface[sib.name].content
            ])

        prompts = GenerationPrompt(
            cvl_prompt=TemplateInstantiation.create(
                templ=NoSourceGenerationPrompt,
                args={
                    "context": batch.feat
                }
            ),
            cvl_prompt_extras=[
                f"The current stub implementation of the {contract_name} is",
                stub_registry.read_stub(contract_name)
            ],
            feedback_prompt=TemplateInstantiation.create(
                templ=FeedbackPrompt,
                args={
                    "context": batch.feat,
                    "has_source": False
                }
            ),
            feedback_prompt_extras=(lambda: [
                f"The current typechecking stub for the {contract_name} stub is",
                stub_registry.read_stub(contract_name),
                "For reference, the system document for the application is",
                system_doc.content
            ])
        )

        env = GenerationEnv(
            prompt=prompts,
            cvl_authorship=services.cvl_research,
            cvl_research=services.cvl_research,
            extra_tools=[*stub_tools, typecheck_tool],
            output_tools=list(publish)
        )

        label = f"{contract_name} {batch.feat.component.name} ({len(batch.props)} properties)"
        return await run_task(
            handler_factory,
            TaskInfo(f"cvl-{contract_name}-{batch_idx}", label, "cvl_gen"),
            lambda: generate_batch_cvl(
                batch_ctx, batch.props, env, with_memory=True,
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
    return ContractResult(
        spec=master_spec.read_unsync() or "",
        failures=failures
    )



async def run_natspec_pipeline(
    system_doc: SystemDoc,
    solc_version: str,
    analysis_builder: PlainBuilder,
    cvl_research: CVLOnlyBuilder,
    ctx: WorkflowContext[None],
    store: BaseStore,
    handler_factory: HandlerFactory[Phase, None],
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
        analysis_builder: Builder for component analysis, interface/stub gen, registry (no domain tools).
        cvl_authorship: Builder for CVL generation agents (has CVL manual tools).
        cvl_research: Builder for CVL research sub-agents and merge agent.
        ctx: Root workflow context.
        store: BaseStore for shared artifacts and caching.
        handler_factory: Creates per-task ``(IOHandler, EventHandler)``
            pairs.  Called once per top-level agent invocation.
        max_concurrent: Maximum concurrent LLM agents.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis
    # ------------------------------------------------------------------
    summary = await run_task(
        handler_factory,
        TaskInfo("component-analysis", SYSTEM_DESC, "component_analysis"),
        lambda: run_component_analysis(ctx, system_doc, analysis_builder),
    )
    if summary is None:
        raise ValueError("Component analysis produced no result — is the system doc empty?")

    # ------------------------------------------------------------------
    # Phase 3: Interface generation
    # ------------------------------------------------------------------
    interface = await run_task(
        handler_factory,
        TaskInfo("interface-gen", INTERFACE_GEN_DESC, "interface_gen"),
        lambda: generate_interface(ctx, summary, system_doc, analysis_builder, solc_version),
    )

    # ------------------------------------------------------------------
    # Phase 4: Initial stub generation
    # ------------------------------------------------------------------
    async def gen_one_stub(
        contract_name: str
    ) -> tuple[str, StubDeclaration]:
        res = await run_task(
            handler_factory,
            TaskInfo(f"stub-gen-{contract_name}", f"Stub: {contract_name}", "stub_gen"),
            lambda: generate_stub(ctx, interface, contract_name, analysis_builder, solc_version),
        )
        return (contract_name, res)

    generated_stubs = await asyncio.gather(*[
        gen_one_stub(c.name) for c in summary.contract_components
    ])

    # ------------------------------------------------------------------
    # Shared artifacts for Phase 5
    # ------------------------------------------------------------------

    registry = StubRegistry.create(
        store, STUB_NS + (string_hash(str(system_doc.content)),), analysis_builder, ctx, interface, {
            k: c.content for (k, c) in generated_stubs
        }, solc_version,
    )

    serv = PipelineServices(
        sem=semaphore,
        builder=analysis_builder,
        cvl_research=cvl_research,
        factory=handler_factory,
        store=store
    )

    tasks : list[Awaitable[ContractResult]] = []

    name_to_stub = { nm: stub for (nm, stub) in generated_stubs }
    import logging
    logging.getLogger(__name__).debug(name_to_stub)


    for (ind, contract) in enumerate(summary.contract_components):
        contract_key = CacheKey[None, Contract](string_hash(contract.model_dump_json()))
        contract_ctx = ctx.child(contract_key, contract.model_dump())
        cont = analyze_single_contract(
            system_doc=system_doc,
            ctx=contract_ctx,
            services=serv,
            solc_version=solc_version,
            intf=interface,
            stub_registry=registry,
            summary=ContractInstance(ind=ind, app=summary),
            stub=name_to_stub[contract.name]
        )
        
        tasks.append(cont)
    results = await asyncio.gather(*tasks)

     
    to_ret : list[ContractFormulation] = []
    for (c, res) in zip(summary.contract_components, results):
        to_ret.append(ContractFormulation(
            spec=res.spec,
            failures=res.failures,
            interface=interface.name_to_interface[c.name],
            stub=name_to_stub[c.name],
            name=c.name
        ))
    return PipelineResult(
        app=summary,
        contracts=to_ret
    )
