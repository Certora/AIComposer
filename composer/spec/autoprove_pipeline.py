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
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.store.base import BaseStore

from composer.io.multi_job_app import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.io.autoprove_app import AutoProvePhase

from composer.spec.context import (
    WorkflowContext, SourceCode, SourceBuilder, CVLBuilder, CVLOnlyBuilder,
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
from composer.spec.cvl_generation import (
    GenerationEnv, GeneratedCVL, CVLResource, generate_batch_cvl,
)
from composer.spec.harness import setup_and_harness_agent, Configuration
from composer.spec.summarizer import setup_summaries
from composer.spec.struct_invariant import get_invariant_formulation
from composer.spec.prover import get_prover_tool, CloudConfig


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

PROPERTIES_KEY = CacheKey[None, Properties]("properties")
INV_CVL_KEY = CacheKey[None, GeneratedCVL]("invariant-cvl")


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
class AutoProveResult:
    n_components: int
    n_properties: int
    failures: list[str] = field(default_factory=list)

from logging import getLogger

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_autoprove_pipeline(
    system_doc: SourceCode,
    source_tools: SourceBuilder,
    cvl_authorship: CVLBuilder,
    cvl_research: CVLOnlyBuilder,
    ctx: WorkflowContext[None],
    store: BaseStore,
    handler_factory: HandlerFactory[AutoProvePhase],
    *,
    llm: BaseChatModel,
    cloud: CloudConfig | None = None,
    max_concurrent: int = 4,
) -> AutoProveResult:
    """Run the auto-prove multi-agent pipeline."""
    semaphore = asyncio.Semaphore(max_concurrent)
    analysis_input = (system_doc, source_tools)

    # ------------------------------------------------------------------
    # Phase 1: Harness setup
    # ------------------------------------------------------------------
    config: Configuration = await run_task(
        handler_factory,
        TaskInfo("harness-setup", "Harness Setup", AutoProvePhase.HARNESS),
        lambda: setup_and_harness_agent(ctx, system_doc, llm),
    )

    # Build initial resources from PreAudit-generated summaries
    resources: list[CVLResource] = [
        CVLResource(
            import_path=config.summaries_path,
            required=True,
            description="PreAudit-generated summaries",
            sort="import",
        ),
    ]

    # Build prover tool (needs config from phase 1)
    prover_tool = get_prover_tool(
        llm, config.config, system_doc.contract_name,
        system_doc.project_root, cloud=cloud, semaphore=semaphore,
    )

    # ------------------------------------------------------------------
    # Phase 2: Custom summaries
    # ------------------------------------------------------------------
    has_summarizable = any(
        c.l == "SUMMARIZABLE" for c in config.external_contracts
    )
    if has_summarizable:
        summary_resource: CVLResource = await run_task(
            handler_factory,
            TaskInfo("summaries", "Custom Summaries", AutoProvePhase.SUMMARIES),
            lambda: setup_summaries(ctx, system_doc, config, cvl_authorship, cvl_research),
        )
        resources.append(summary_resource)

    # ------------------------------------------------------------------
    # Phase 3: Structural invariants
    # ------------------------------------------------------------------
    invariants = await run_task(
        handler_factory,
        TaskInfo("invariants", "Structural Invariants", AutoProvePhase.INVARIANTS),
        lambda: get_invariant_formulation(ctx, system_doc, source_tools),
    )

    if invariants.inv:
        inv_cvl_ctx = ctx.child(INV_CVL_KEY)
        cached_inv_cvl = inv_cvl_ctx.cache_get(GeneratedCVL)

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

            inv_env = GenerationEnv(
                input=system_doc,
                cvl_authorship=cvl_authorship,
                cvl_research=cvl_research,
                source_tools=source_tools,
                resources=list(resources),
                prover_tool=prover_tool,
            )

            inv_cvl = await run_task(
                handler_factory,
                TaskInfo("invariant-cvl", "Invariant CVL", AutoProvePhase.CVL_GEN),
                lambda: generate_batch_cvl(
                    inv_cvl_ctx.abstract(CVLGeneration),
                    inv_props, None, inv_env,
                    with_memory=True,
                    description="Structural invariant CVL",
                ),
            )
            inv_cvl_ctx.cache_put(inv_cvl)

        inv_spec_name = "invariants.spec"
        (Path(system_doc.project_root) / "certora" / inv_spec_name).write_text(inv_cvl.cvl)
        resources.append(CVLResource(
            import_path=inv_spec_name,
            required=False,
            description="Structural invariants that may be assumed as preconditions",
            sort="import",
        ))

    # ------------------------------------------------------------------
    # Phase 4: Component analysis
    # ------------------------------------------------------------------
    summary = await run_task(
        handler_factory,
        TaskInfo("component-analysis", COMPONENT_ANALYSIS_DESC, AutoProvePhase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(ctx, analysis_input),
    )
    if summary is None:
        raise ValueError("Component analysis produced no result — is the system doc empty?")

    # ------------------------------------------------------------------
    # Phase 5: Per-component property extraction
    # ------------------------------------------------------------------
    prop_context = ctx.child(PROPERTIES_KEY)

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

        props = await run_task(
            handler_factory,
            TaskInfo(f"bug-{component_idx}", name, AutoProvePhase.BUG_ANALYSIS),
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

    # ------------------------------------------------------------------
    # Phase 6: Per-component CVL generation
    # ------------------------------------------------------------------
    async def _generate_batch(
        batch_idx: int,
        batch: _ComponentBatch,
    ) -> GeneratedCVL:
        batch_ctx = batch.feat_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        ).abstract(CVLGeneration)

        env = GenerationEnv(
            input=system_doc,
            cvl_authorship=cvl_authorship,
            cvl_research=cvl_research,
            source_tools=source_tools,
            resources=resources,
            prover_tool=prover_tool,
        )

        label = f"{batch.feat.component.name} ({len(batch.props)} properties)"
        return await run_task(
            handler_factory,
            TaskInfo(f"cvl-{batch_idx}", label, AutoProvePhase.CVL_GEN),
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

    failures: list[str] = []
    n_properties = 0
    for batch, result in zip(component_batches, generation_results):
        n_properties += len(batch.props)
        if isinstance(result, BaseException):
            failures.append(f"{batch.feat.component.name}: {result}")
        elif isinstance(result, GeneratedCVL) and result.commentary.startswith("GAVE_UP:"):
            failures.append(f"{batch.feat.component.name}: {result.commentary}")

    return AutoProveResult(
        n_components=len(component_batches),
        n_properties=n_properties,
        failures=failures,
    )
