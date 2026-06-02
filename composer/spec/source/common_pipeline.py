
import asyncio
from dataclasses import dataclass, field
import pathlib

from langchain_core.tools import BaseTool

from composer.io.multi_job import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.ui.autoprove_app import AutoProvePhase

from composer.input.files import Document
from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties, ComponentGroup, CVLGeneration,
)
from composer.spec.util import string_hash, slugify_filename
from composer.spec.prop_inference import run_property_inference
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import CVLResource
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.system_model import (
    ContractComponentInstance, HarnessedApplication, ContractInstance
)
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.author import batch_cvl_generation, GaveUp, BatchGeneratedCVLResult
from composer.spec.source.prover import dump_final_conf

PROPERTIES_KEY = CacheKey[None, Properties]("properties")
INV_CVL_KEY = CacheKey[None, GeneratedCVL]("invariant-cvl")


def _component_cache_key(
    component: ContractComponentInstance,
) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([component.app.model_dump_json(), str(component.ind), str(component._contract.ind)])
    return CacheKey(string_hash(combined))


def _batch_cache_key(props: list[PropertyFormulation]) -> CacheKey[ComponentGroup, GeneratedCVL]:
    combined = "|".join(p.model_dump_json() for p in props)
    return CacheKey(string_hash(combined))

@dataclass
class AutoProveResult:
    n_components: int
    n_properties: int
    failures: list[str] = field(default_factory=list)

async def run_generation_pipeline(
    source_input: SourceCode,
    prop_context: WorkflowContext[Properties],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    summary: HarnessedApplication,
    semaphore: asyncio.Semaphore,
    resources: list[CVLResource],
    prover_tool: BaseTool,
    prover_config: dict,
    interactive: bool,
    threat_model: Document | None,
    max_bug_rounds: int = 3,
) -> AutoProveResult:

    contract_instance : ContractInstance

    ind = -1

    for i, c in enumerate(summary.contract_components):
        if c.name == source_input.contract_name:
            ind = i
            break
    if ind == -1:
        raise ValueError("Component not found")

    contract_instance = ContractInstance(
        ind, app=summary
    )

    # ------------------------------------------------------------------
    # Phase 5: Per-component property extraction
    # ------------------------------------------------------------------
    @dataclass
    class _ComponentBatch:
        feat: ContractComponentInstance
        props: list[PropertyFormulation]
        feat_ctx: WorkflowContext[ComponentGroup]

    async def _analyze_component(component_idx: int) -> _ComponentBatch | None:
        feat = ContractComponentInstance(_contract=contract_instance, ind=component_idx)
        name = feat.component.name
        feat_ctx = await prop_context.child(
            _component_cache_key(feat),
            {
                "component": feat.component.model_dump(),
            },
        )

        props = await run_task(
            handler_factory,
            TaskInfo(f"bug-{component_idx}", name, AutoProvePhase.BUG_ANALYSIS),
            lambda conv: run_property_inference(feat_ctx, env, feat, refinement=conv if interactive else None, threat_model=threat_model, max_rounds=max_bug_rounds),
            semaphore,
        )

        if not props:
            return None
        return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)

    extraction_results = await asyncio.gather(*[
        _analyze_component(i) for i in range(len(contract_instance.contract.components))
    ])

    component_batches = [b for b in extraction_results if b is not None]

    if not component_batches:
        raise ValueError("No properties extracted from any component.")

    raw_slugs = [slugify_filename(b.feat.component.name) for b in component_batches]
    slug_counts: dict[str, int] = {}
    for s in raw_slugs:
        slug_counts[s] = slug_counts.get(s, 0) + 1
    batch_filename_bases = [
        f"{s}_{b.feat.ind}" if slug_counts[s] > 1 else s
        for s, b in zip(raw_slugs, component_batches)
    ]

    # ------------------------------------------------------------------
    # Phase 6: Per-component CVL generation
    # ------------------------------------------------------------------
    async def _generate_batch(
        task_id: str,
        batch: _ComponentBatch,
    ) -> BatchGeneratedCVLResult:
        batch_child = await batch.feat_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        )
        if (cached := await batch_child.cache_get(GeneratedCVL)) is not None:
            return cached
        batch_ctx = batch_child.abstract(CVLGeneration)

        label = f"{batch.feat.component.name} ({len(batch.props)} properties)"
        res = await run_task(
            handler_factory,
            TaskInfo(task_id, label, AutoProvePhase.CVL_GEN),
            lambda: batch_cvl_generation(
                ctx=batch_ctx,
                init_config=prover_config,
                component=batch.feat,
                env=env,
                props=batch.props,
                prover_tool=prover_tool,
                resources=resources,
                description=label,
                source=source_input
            ),
            semaphore,
        )
        if isinstance(res, GeneratedCVL):
            await batch_child.cache_put(res)
        return res

    async def _generate_and_write_batch(
        i: int, batch: _ComponentBatch
    ) -> BatchGeneratedCVLResult:
        task_id = f"cvl-{i}"
        res = await _generate_batch(task_id=task_id, batch=batch)
        if isinstance(res, GaveUp):
            return res
        certora_dir = pathlib.Path(source_input.project_root) / "certora"
        specs_dir = certora_dir / "specs"
        specs_dir.mkdir(exist_ok=True, parents=True)
        base = batch_filename_bases[i]
        spec_name = pathlib.Path(f"autospec_{base}.spec")
        (specs_dir / spec_name).write_text(res.cvl)
        (certora_dir / f"autospec_{base}.commentary.md").write_text(res.commentary)
        dump_final_conf(
            project_root=source_input.project_root,
            main_contract=source_input.contract_name,
            task_id=task_id,
            spec_name=spec_name,
            conf=res.conf,
        )
        return res

    generation_results = await asyncio.gather(
        *[
            _generate_and_write_batch(i, batch)
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
        elif isinstance(result, GaveUp):
            failures.append(f"{batch.feat.component.name}: GAVE_UP: {result.reason}")

    return AutoProveResult(
        n_components=len(component_batches),
        n_properties=n_properties,
        failures=failures,
    )
