"""
Auto-prove multi-agent pipeline orchestration.

Phases:
1. Harness setup — classify external contracts, generate harness files
2. Custom summaries — generate CVL summaries for SUMMARIZABLE contracts
3. Structural invariants — formulate and generate CVL for structural invariants
4. Component analysis
5. Per-component property extraction (parallel)
6. Per-component CVL generation (parallel, semaphore-bounded)
"""

import asyncio
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from composer.ui.multi_job_app import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.ui.autoprove_app import AutoProvePhase

from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties, CVLGeneration,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import CVLResource
from composer.spec.source.harness import run_setup
from composer.spec.source.system_analysis import run_component_analysis
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.source.summarizer import setup_summaries
from composer.spec.system_model import (
    HarnessedApplication, SourceExplicitContract,
    HarnessedExplicitContract, SourceExternalActor, HarnessDefinition
)
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.prover import CloudConfig
from composer.spec.source.prover import get_prover_tool
from composer.spec.source.struct_invariant import get_invariant_formulation
from composer.spec.source.author import batch_cvl_generation
from composer.spec.source.common_pipeline import run_generation_pipeline, AutoProveResult


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[None, Properties]("properties")
INV_CVL_KEY = CacheKey[None, GeneratedCVL]("invariant-cvl")



# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_autoprove_pipeline(
    llm: BaseChatModel,
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    *,
    cloud: CloudConfig | None = None,
    max_concurrent: int = 4,
) -> AutoProveResult:
    """Run the auto-prove multi-agent pipeline."""
    semaphore = asyncio.Semaphore(max_concurrent)

    s = await run_task(
        handler_factory,
        TaskInfo("system-analysis", "System Analysis", AutoProvePhase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(ctx, source_input, env=env)
    )

    if s is None:
        raise ValueError("we're fucked")

    setup = await run_task(
        handler_factory,
        TaskInfo("setup", "Auto Setup", AutoProvePhase.HARNESS),
        lambda: run_setup(
            ctx, source_input, env, s
        )
    )
    
    if setup is None:
        raise ValueError("we're still fucked")

    contract_to_harness : dict[str, list[HarnessDefinition]] = {}
    for c in setup.system_description.transitive_closure:
        if not c.harness_definition:
            continue
        if c.harness_definition.harness_of not in contract_to_harness:
            contract_to_harness[c.harness_definition.harness_of] = []
        contract_to_harness[c.harness_definition.harness_of].append(
            HarnessDefinition(
                name=c.name,
                path=c.path
            )
        )

    comp : list[SourceExternalActor | HarnessedExplicitContract] = []
    for c in s.components:
        if not isinstance(c, SourceExplicitContract):
            comp.append(c)
            continue
        comp.append(HarnessedExplicitContract(
            sort=c.sort,
            name=c.name,
            components=c.components,
            description=c.description,
            path=c.path,
            harnesses=contract_to_harness.get(c.name, [])
        ))


    harnessed_app : HarnessedApplication = HarnessedApplication(
        application_type=s.application_type,
        description=s.description,
        components=comp
    )

    # Build initial resources from PreAudit-generated summaries
    resources: list[CVLResource] = [
        CVLResource(
            import_path=str(setup.config.summaries_path),
            required=True,
            description="PreAudit-generated summaries",
            sort="import",
        ),
    ]

    if setup.system_description.erc20_contracts or setup.system_description.external_interfaces:
        summary_resource : CVLResource = await run_task(
            handler_factory,
            TaskInfo("summaries", "Custom Summaries", AutoProvePhase.SUMMARIES),
            lambda: setup_summaries(
                ctx=ctx,
                app=harnessed_app,
                config=setup,
                env=env,
                source=source_input
            )
        )
        resources.append(summary_resource)

    # Build prover tool (needs config from phase 1)
    prover_tool = get_prover_tool(
        llm, source_input.contract_name,
        source_input.project_root, cloud=cloud,
    )

    # ------------------------------------------------------------------
    # Phase 3: Structural invariants
    # ------------------------------------------------------------------
    invariants = await run_task(
        handler_factory,
        TaskInfo("invariants", "Structural Invariants", AutoProvePhase.INVARIANTS),
        lambda: get_invariant_formulation(ctx, source_input, env, harnessed_app),
    )

    if invariants.inv:
        inv_cvl_ctx = ctx.child(INV_CVL_KEY)
        cached_inv_cvl = await inv_cvl_ctx.cache_get(GeneratedCVL)

        if cached_inv_cvl is not None:
            inv_cvl = cached_inv_cvl
        else:
            inv_props = [
                PropertyFormulation(
                    methods="invariant",
                    description=inv.description,
                    sort="invariant",
                )
                for inv in invariants.inv
            ]

            inv_cvl = await run_task(
                handler_factory,
                TaskInfo("invariant-cvl", "Invariant CVL", AutoProvePhase.CVL_GEN),
                lambda: batch_cvl_generation(
                    ctx=inv_cvl_ctx.abstract(CVLGeneration),
                    component=None,
                    props=inv_props,
                    env=env,
                    init_config=setup.config.prover_config,
                    prover_tool=prover_tool,
                    resources=resources,
                    description="Structural invariant CVL",
                    source=source_input
                ),
            )
            await inv_cvl_ctx.cache_put(inv_cvl)

        inv_spec_name = "invariants.spec"
        (Path(source_input.project_root) / "certora" / inv_spec_name).write_text(inv_cvl.cvl)
        resources.append(CVLResource(
            import_path=inv_spec_name,
            required=False,
            description="Structural invariants that may be assumed as preconditions",
            sort="import",
        ))

    prop_context = ctx.child(PROPERTIES_KEY)

    res = await run_generation_pipeline(
        source_input=source_input,
        env=env,
        handler_factory=handler_factory,
        prop_context=prop_context,
        prover_config=setup.config.prover_config,
        prover_tool=prover_tool,
        resources=resources,
        semaphore=semaphore,
        summary=harnessed_app
    )
    return res
