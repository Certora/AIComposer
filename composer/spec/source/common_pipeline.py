
import asyncio
from dataclasses import dataclass, field
import json
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
from composer.spec.util import string_hash, ensure_dir
from composer.spec.prop_inference import run_property_inference
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import CVLResource, CERTORA_DIR, SPECS_DIR, under_project
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.system_model import (
    ContractComponentInstance, HarnessedApplication, ContractInstance
)
from composer.spec.cvl_generation import GeneratedCVL, PropertyRuleMapping
from composer.spec.source.author import batch_cvl_generation, GaveUp, BatchGeneratedCVLResult
from composer.spec.source.prover import dump_final_conf
from composer.spec.source.task_ids import bug_analysis_task_id, cvl_gen_task_id

PROPERTIES_KEY = CacheKey[None, Properties]("properties")
INV_CVL_KEY = CacheKey[None, GeneratedCVL]("invariant-cvl")


def dump_properties(
    certora_dir: pathlib.Path,
    spec_stem: str,
    props: list[PropertyFormulation],
) -> None:
    """Write the analysis-phase properties (title, sort, methods, description) to
    ``properties/{spec_stem}.properties.json`` under ``certora_dir``, accompanying
    ``{spec_stem}.spec``. ``title`` is the cross-reference key used by
    ``{spec_stem}.property_rules.json``."""
    properties_dir = ensure_dir(certora_dir / "properties")
    properties_dump = [prop.model_dump() for prop in props]
    (properties_dir / f"{spec_stem}.properties.json").write_text(
        json.dumps(properties_dump, indent=2)
    )


def dump_property_rules(
    certora_dir: pathlib.Path,
    spec_stem: str,
    property_rules: list[PropertyRuleMapping],
) -> None:
    """Write the property->rules mapping ``{property title: [rule names]}`` to
    ``properties/{spec_stem}.property_rules.json`` under ``certora_dir``, accompanying
    ``{spec_stem}.spec``. Titles are unique (enforced at extraction) and validated against
    the batch at completion."""
    properties_dir = ensure_dir(certora_dir / "properties")
    mapping = {m.property_title: m.rules for m in property_rules}
    (properties_dir / f"{spec_stem}.property_rules.json").write_text(
        json.dumps(mapping, indent=2)
    )


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

@dataclass
class _ComponentBatch:
    feat: ContractComponentInstance
    props: list[PropertyFormulation]
    feat_ctx: WorkflowContext[ComponentGroup]


async def extract_all_components(
    *,
    source_input: SourceCode,
    prop_context: WorkflowContext[Properties],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    summary: HarnessedApplication,
    semaphore: asyncio.Semaphore,
    interactive: bool,
    threat_model: Document | None,
    max_bug_rounds: int = 3,
) -> list[_ComponentBatch]:
    """Phase 5 — per-component property extraction ("bug analysis").

    Runs ``run_property_inference`` for every component in parallel
    (semaphore-bounded) and dumps the analysis-phase properties. Returns the
    batches that yielded properties; an empty list means nothing was extracted
    (the caller decides how to react).
    """
    ind = -1
    for i, c in enumerate(summary.contract_components):
        if c.name == source_input.contract_name:
            ind = i
            break
    if ind == -1:
        raise ValueError("Component not found")

    contract_instance = ContractInstance(ind, app=summary)

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
            TaskInfo(bug_analysis_task_id(component_idx, feat.slugified_name), name, AutoProvePhase.BUG_ANALYSIS),
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
        return []

    # Dump the analysis-phase properties for each component now that the
    # extraction phase is complete.
    certora_dir = under_project(source_input.project_root, CERTORA_DIR)
    for batch in component_batches:
        dump_properties(certora_dir, f"autospec_{batch.feat.slugified_name}", batch.props)

    return component_batches


async def generate_all_component_cvl(
    *,
    source_input: SourceCode,
    component_batches: list[_ComponentBatch],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    prover_tool: BaseTool,
    prover_config: dict,
    resources: list[CVLResource],
    semaphore: asyncio.Semaphore,
) -> AutoProveResult:
    """Phase 6 — per-component CVL generation.

    Generates and writes a spec for each extracted batch in parallel
    (semaphore-bounded). ``resources`` is consumed read-only; callers that want
    the structural invariants assumed as preconditions must include
    ``invariants.spec`` in ``resources`` before calling.
    """
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
                source=source_input,
                spec_dir=SPECS_DIR,
            ),
            semaphore,
        )
        if isinstance(res, GeneratedCVL):
            await batch_child.cache_put(res)
        return res

    async def _generate_and_write_batch(
        batch: _ComponentBatch
    ) -> BatchGeneratedCVLResult:
        task_id = cvl_gen_task_id(batch.feat.ind, batch.feat.slugified_name)
        res = await _generate_batch(task_id=task_id, batch=batch)
        if isinstance(res, GaveUp):
            return res
        certora_dir = under_project(source_input.project_root, CERTORA_DIR)
        specs_dir = ensure_dir(certora_dir / "specs")  # absolute (project_root/certora/specs)
        properties_dir = ensure_dir(certora_dir / "properties")
        base = batch.feat.slugified_name
        spec_name = pathlib.Path(f"autospec_{base}.spec")
        (specs_dir / spec_name).write_text(res.cvl)
        # Canonical (project-root-relative) path of the persisted spec, used for
        # the conf's verify entry.
        spec_path = SPECS_DIR / spec_name
        (properties_dir / f"autospec_{base}.commentary.md").write_text(res.commentary)
        dump_property_rules(certora_dir, f"autospec_{base}", res.property_rules)
        dump_final_conf(
            project_root=source_input.project_root,
            main_contract=source_input.contract_name,
            task_id=task_id,
            spec_path=spec_path,
            conf=res.conf,
        )
        return res

    generation_results = await asyncio.gather(
        *[
            _generate_and_write_batch(batch)
            for batch in component_batches
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
    """Property extraction followed by CVL generation for every component.

    Thin wrapper over ``extract_all_components`` + ``generate_all_component_cvl``
    for callers that run the two phases back-to-back (e.g. ``direct_pipeline``).
    The staged pipeline calls the two halves directly so it can interleave other
    phases (autosetup, invariant CVL) between them.
    """
    component_batches = await extract_all_components(
        source_input=source_input,
        prop_context=prop_context,
        handler_factory=handler_factory,
        env=env,
        summary=summary,
        semaphore=semaphore,
        interactive=interactive,
        threat_model=threat_model,
        max_bug_rounds=max_bug_rounds,
    )
    if not component_batches:
        raise ValueError("No properties extracted from any component.")
    return await generate_all_component_cvl(
        source_input=source_input,
        component_batches=component_batches,
        handler_factory=handler_factory,
        env=env,
        prover_tool=prover_tool,
        prover_config=prover_config,
        resources=resources,
        semaphore=semaphore,
    )
