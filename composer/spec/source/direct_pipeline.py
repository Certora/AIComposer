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

from langchain_core.language_models.chat_models import BaseChatModel

from composer.ui.multi_job_app import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.ui.autoprove_app import AutoProvePhase

from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties,
)
from composer.spec.gen_types import CVLResource
from composer.spec.source.system_analysis import run_component_analysis
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.system_model import (
    HarnessedApplication, SourceExplicitContract,
    HarnessedExplicitContract, SourceExternalActor, HarnessDefinition
)
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.prover import CloudConfig, get_prover_tool
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
    custom_summary_path: str,
    standard_summary_path: str,
    config_path: str,
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
        raise ValueError("System analysis failed")

    contract_to_harness : dict[str, list[HarnessDefinition]] = {}
    
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

    # Build initial resources from AutoSetup-generated summaries
    resources: list[CVLResource] = [
        CVLResource(
            import_path=str(standard_summary_path),
            required=True,
            description="AutoSetup-generated summaries",
            sort="import",
        ),
        CVLResource(
            import_path=str(custom_summary_path),
            required=False,
            description=f"Summaries specific to {source_input.contract_name}",
            sort="import"
        )
    ]

    import json
    config = json.load(open(config_path, "r"))

    # ------------------------------------------------------------------
    # Phase 3: Structural invariants
    # ------------------------------------------------------------------

    # Build prover tool (needs config from phase 1)
    prover_tool = get_prover_tool(
        llm, source_input.contract_name,
        source_input.project_root, cloud=cloud,
    )

    # ------------------------------------------------------------------
    # Phase 5: Per-component property extraction
    # ------------------------------------------------------------------
    prop_context = ctx.child(PROPERTIES_KEY)
    return await run_generation_pipeline(
        source_input=source_input,
        env=env,
        handler_factory=handler_factory,
        prop_context=prop_context,
        prover_config=config,
        prover_tool=prover_tool,
        resources=resources,
        semaphore=semaphore,
        summary=harnessed_app
    )
