"""
Auto-prove multi-agent pipeline orchestration.

Phases:
1. Component analysis
2. Harness creation — classify external contracts, generate harness files
3. In parallel, after harness creation:
     - AutoSetup (compilation config + summaries), then custom summaries
     - Structural-invariant formulation
     - Per-component property extraction ("bug analysis")
4. Staged CVL-generation join:
     - Stage 1: structural-invariant CVL (writes invariants.spec)
     - Stage 2: per-component CVL (parallel, imports invariants.spec as
       assumable preconditions)
"""

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.spec.source.autosetup import SetupSuccess
from composer.ui.autoprove_app import AutoProvePhase

from composer.input.files import Document
from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties, CVLGeneration,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import CVLResource, CERTORA_DIR, SPECS_DIR, certora_relative_to_project, under_project
from composer.spec.util import ensure_dir
from composer.spec.source.harness import run_harness_creation, run_autosetup_phase, ContractSetup
from composer.spec.source.system_analysis import run_component_analysis
from composer.spec.service_host import ServiceHost
from composer.spec.source.summarizer import setup_summaries
from composer.spec.system_model import (
    HarnessedApplication, SourceExplicitContract,
    HarnessedExplicitContract, SourceExternalActor, HarnessDefinition, SolidityIdentifier
)
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.prover import get_prover_tool, dump_final_conf
from composer.prover.core import ProverOptions
from composer.spec.source.struct_invariant import get_invariant_formulation
from composer.spec.source.author import batch_cvl_generation, GaveUp
from composer.spec.source.common_pipeline import extract_all_components, generate_all_component_cvl, AutoProveResult, dump_properties, dump_property_rules
from composer.spec.source.task_ids import (
    SYSTEM_ANALYSIS_TASK_ID, HARNESS_TASK_ID, AUTOSETUP_TASK_ID,
    SUMMARIES_TASK_ID, INVARIANTS_TASK_ID, INVARIANT_CVL_TASK_ID,
)


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
    env: ServiceHost,
    *,
    prover_opts: ProverOptions,
    max_concurrent: int = 4,
    interactive: bool,
    threat_model : Document | None = None,
    max_bug_rounds: int = 3,
) -> AutoProveResult:
    """Run the auto-prove multi-agent pipeline."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis
    # ------------------------------------------------------------------
    s = await run_task(
        handler_factory,
        TaskInfo(SYSTEM_ANALYSIS_TASK_ID, "System Analysis", AutoProvePhase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(ctx, source_input, env=env)
    )

    if s is None:
        raise ValueError("System analysis failed")

    # ------------------------------------------------------------------
    # Phase 2: Harness creation. Prerequisite for AutoSetup (which reads the
    # generated harness files) and for the invariant/bug branch (both consume
    # harnessed_app, built from harness creation + component analysis).
    # ------------------------------------------------------------------
    sys_desc = await run_task(
        handler_factory,
        TaskInfo(HARNESS_TASK_ID, "Harness Creation", AutoProvePhase.HARNESS),
        lambda: run_harness_creation(ctx, source_input, env, s),
    )

    contract_to_harness : dict[SolidityIdentifier, list[HarnessDefinition]] = {}
    for c in sys_desc.transitive_closure:
        if not c.harness_definition:
            continue
        if c.harness_definition.harness_of not in contract_to_harness:
            contract_to_harness[c.harness_definition.harness_of] = []
        contract_to_harness[c.harness_definition.harness_of].append(
            HarnessDefinition(
                name=c.solidity_identifier,
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
            solidity_identifier=c.solidity_identifier,
            components=c.components,
            description=c.description,
            path=c.path,
            harnesses=contract_to_harness.get(c.solidity_identifier, [])
        ))


    harnessed_app : HarnessedApplication = HarnessedApplication(
        application_type=s.application_type,
        description=s.description,
        components=comp
    )

    # Prover tool is stateless with respect to setup, so build it now; it is
    # shared by every CVL-generation call below.
    prover_tool = get_prover_tool(
        llm, source_input.contract_name,
        source_input.project_root, prover_opts=prover_opts,
    )

    # ------------------------------------------------------------------
    # Phase 3 (parallel branches, joined below):
    #   A) AutoSetup → custom summaries  (produces prover config + resources)
    #   B) structural-invariant formulation
    #   C) per-component property extraction ("bug analysis")
    # B and C are independent of A; they only need harnessed_app + source.
    # ------------------------------------------------------------------
    async def stream_autosetup() -> tuple[SetupSuccess, list[CVLResource]]:
        setup_config = await run_task(
            handler_factory,
            TaskInfo(AUTOSETUP_TASK_ID, "AutoSetup", AutoProvePhase.AUTOSETUP),
            lambda: run_autosetup_phase(ctx, source_input, sys_desc, s, prover_opts),
        )
        resources: list[CVLResource] = [
            CVLResource(
                path=certora_relative_to_project(setup_config.summaries_path),
                required=True,
                description="AutoSetup-generated summaries",
                sort="import",
            ),
        ]
        if sys_desc.erc20_contracts or sys_desc.external_interfaces:
            summary_resource = await run_task(
                handler_factory,
                TaskInfo(SUMMARIES_TASK_ID, "Custom Summaries", AutoProvePhase.SUMMARIES),
                lambda: setup_summaries(
                    ctx=ctx,
                    app=harnessed_app,
                    config=ContractSetup(system_description=sys_desc, config=setup_config),
                    env=env,
                    source=source_input
                )
            )
            resources.append(summary_resource)
        return setup_config, resources

    async def stream_invariants():
        return await run_task(
            handler_factory,
            TaskInfo(INVARIANTS_TASK_ID, "Structural Invariants", AutoProvePhase.INVARIANTS),
            lambda: get_invariant_formulation(ctx, source_input, env, harnessed_app),
        )

    async def stream_bugs():
        return await extract_all_components(
            source_input=source_input,
            prop_context=ctx.child(PROPERTIES_KEY),
            handler_factory=handler_factory,
            env=env,
            summary=harnessed_app,
            semaphore=semaphore,
            interactive=interactive,
            threat_model=threat_model,
            max_bug_rounds=max_bug_rounds,
        )

    (setup_config, resources), invariants, component_batches = await asyncio.gather(
        stream_autosetup(),
        stream_invariants(),
        stream_bugs(),
    )

    if not component_batches:
        raise ValueError("No properties extracted from any component.")

    certora_dir = under_project(source_input.project_root, CERTORA_DIR)

    # ------------------------------------------------------------------
    # Join, stage 1: structural-invariant CVL. Runs before the per-component
    # CVL so invariants.spec exists and can be imported as preconditions.
    # ------------------------------------------------------------------
    if invariants.inv:
        inv_task_id = INVARIANT_CVL_TASK_ID
        inv_cvl_ctx = ctx.child(INV_CVL_KEY)
        cached_inv_cvl = await inv_cvl_ctx.cache_get(GeneratedCVL)

        inv_props = [
            PropertyFormulation(
                title=inv.name,
                methods="invariant",
                description=inv.description,
                sort="invariant",
            )
            for inv in invariants.inv
        ]

        # Dump the analysis-phase invariant properties now that we have them.
        dump_properties(certora_dir, "invariants", inv_props)

        if cached_inv_cvl is not None:
            inv_cvl = cached_inv_cvl
        else:
            inv_cvl_result = await run_task(
                handler_factory,
                TaskInfo(inv_task_id, "Invariant CVL", AutoProvePhase.CVL_GEN),
                lambda: batch_cvl_generation(
                    ctx=inv_cvl_ctx.abstract(CVLGeneration),
                    component=None,
                    props=inv_props,
                    env=env,
                    init_config=setup_config.prover_config,
                    prover_tool=prover_tool,
                    resources=resources,
                    description="Structural invariant CVL",
                    source=source_input,
                    spec_dir=SPECS_DIR,
                ),
            )
            if isinstance(inv_cvl_result, GaveUp):
                raise RuntimeError(
                    f"Structural invariant CVL generation gave up: {inv_cvl_result.reason}"
                )
            inv_cvl = inv_cvl_result
            await inv_cvl_ctx.cache_put(inv_cvl)

        ensure_dir(certora_dir / "specs")  # absolute (project_root/certora/specs)
        # Canonical (project-root-relative) path of the persisted spec. The conf's
        # verify entry derives from it; the CVL import path is derived (relative to
        # certora/specs/) where the import is emitted.
        inv_spec_path = SPECS_DIR / "invariants.spec"
        under_project(source_input.project_root, inv_spec_path).write_text(inv_cvl.cvl)
        dump_property_rules(certora_dir, "invariants", inv_cvl.property_rules)
        dump_final_conf(
            project_root=source_input.project_root,
            main_contract=source_input.contract_name,
            task_id=inv_task_id,
            spec_path=inv_spec_path,
            conf=inv_cvl.conf,
        )
        # All three streams have already joined, so `resources` is no longer
        # shared with any running task: this append is race-free, and the
        # stage-2 component CVLs below will see invariants.spec.
        resources.append(CVLResource(
            path=inv_spec_path,
            required=False,
            description="Structural invariants that may be assumed as preconditions",
            sort="import",
        ))

    # ------------------------------------------------------------------
    # Join, stage 2: per-component CVL (parallel, semaphore-bounded). Imports
    # invariants.spec (if any) as assumable preconditions.
    # ------------------------------------------------------------------
    return await generate_all_component_cvl(
        source_input=source_input,
        component_batches=component_batches,
        handler_factory=handler_factory,
        env=env,
        prover_tool=prover_tool,
        prover_config=setup_config.prover_config,
        resources=resources,
        semaphore=semaphore,
    )
