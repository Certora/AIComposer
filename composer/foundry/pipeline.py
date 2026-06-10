"""Foundry test-generation pipeline.

Orchestrates the same component-analysis + per-component property
inference scaffolding the autoprove pipeline uses, but routes the
per-component generation step into ``batch_foundry_test_generation``
(foundry ``.t.sol`` output, ``forge test`` gating) instead of CVL+prover.

Reused as-is from existing infrastructure (NOT modified):

* ``composer.spec.system_analysis.run_component_analysis`` — produces an
  ``Application``-typed model of the system from the design doc + source.
* ``composer.spec.prop_inference.run_property_inference`` — per-component
  property extraction.
* ``composer.spec.context.WorkflowContext`` / ``SourceCode`` / cache keys.

Writes one ``.t.sol`` file per component under ``<project>/test/``
(named ``composer_<component>.t.sol``) on success. Skipped batches and
give-ups are reported as failures in the result.
"""

import asyncio
import pathlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from composer.foundry.author import (
    BatchFoundryResult, GaveUp, GeneratedFoundryTest, batch_foundry_test_generation,
)
from composer.foundry.env import FoundryEnv
from composer.spec.context import (
    CVLGeneration, CacheKey, ComponentGroup, Properties, SourceCode, WorkflowContext,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.prop_inference import run_property_inference
from composer.spec.system_analysis import run_component_analysis


# Backend-guidance string fed into the property-analysis prompt — describes
# what kinds of properties are / aren't a fit for foundry's verification
# surface so the extraction agent doesn't propose properties that are
# unrealistic to formalize as ``forge test`` runs.
FOUNDRY_BACKEND_GUIDANCE: str = """\
These properties will be checked using Foundry. Foundy,
as a unit testing/fuzzing framework, cannot *prove*
universally quantified properties or invariants.
However, it can approximate these properties (via fuzz tests
and the like) and *refutations* of these universal properties
(surfaced by failures of Foundry tests) are extremely valuable.

Accordingly, you *should* freely write universally quantified properties
without taking into considerations the fundamental limitations of Foundry
as a verification backend. Do *not* artificially restrict the
space of properties you write simply because Foundry cannot *definitively*
prove them to be true; as mentioned above, the approximation of the
property is still valuable.

However, a handful of categories are genuinely a poor fit for Foundry:

1. Properties that reference off-chain events (key compromise, phishing,
   social-engineering attacks, oracle manipulation outside the test's
   modeled actors).
2. Properties whose only meaningful content is hash-collision
   resistance — "no two inputs ever collide" is unprovable by
   sampling. (Note: signature validity, signer authorization, and
   similar crypto-adjacent properties are NOT in this category.)

In addition, due to the advent of checked arithmetic, properties that
assert no overflow are uninteresting. Properties implied by the type
system (a uint256 being non-negative, etc.) are also uninteresting.
"""
from composer.spec.system_model import (
    ContractComponentInstance, ContractInstance, SourceApplication,
)
from composer.spec.util import slugify_filename, string_hash

from composer.io.multi_job import HandlerFactory, TaskInfo, run_task
from composer.ui.autoprove_app import AutoProvePhase


# ---------------------------------------------------------------------------
# Cache keys (parallel to common_pipeline's)
# ---------------------------------------------------------------------------

SOURCE_ANALYSIS_KEY = CacheKey[None, SourceApplication]("foundry-source-analysis")
PROPERTIES_KEY = CacheKey[None, Properties]("foundry-properties")


def _component_cache_key(
    component: ContractComponentInstance,
) -> CacheKey[Properties, ComponentGroup]:
    combined = "|".join([
        component.app.model_dump_json(),
        str(component.ind),
        str(component._contract.ind),
    ])
    return CacheKey(string_hash(combined))


def _batch_cache_key(props: list[PropertyFormulation]) -> CacheKey[ComponentGroup, GeneratedFoundryTest]:
    combined = "|".join(p.model_dump_json() for p in props)
    return CacheKey(string_hash(combined))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class FoundryPipelineResult:
    n_components: int
    n_properties: int
    written: list[pathlib.Path] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_foundry_pipeline(
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: FoundryEnv,
    *,
    max_concurrent: int = 4,
    max_bug_rounds: int = 3,
    interactive: bool = False,
    forge_binary: str = "forge",
    forge_timeout_s: int = 600,
) -> FoundryPipelineResult:
    """Run the foundry test-generation pipeline against an existing project.

    ``source_input.project_root`` must point at a configured foundry project
    (``foundry.toml`` + ``lib/forge-std`` + the contracts under test). The
    pipeline does NOT modify ``foundry.toml`` / ``lib/`` / ``src/`` — only
    writes generated ``.t.sol`` files under ``<project>/test/``.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis. SourceApplication so the model carries
    # paths back from the design doc + source-tree exploration.
    # ------------------------------------------------------------------
    summary = await run_task(
        handler_factory,
        TaskInfo("system-analysis", "System Analysis", AutoProvePhase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(
            ty=SourceApplication,
            child_ctxt=ctx.child(SOURCE_ANALYSIS_KEY),
            input=source_input,
            env=env,
            extra_input=[
                f"The main entry point of this application has been "
                f"explicitly identified as {source_input.contract_name} at "
                f"relative path {source_input.relative_path}",
            ],
        ),
    )
    if summary is None:
        raise ValueError("Component analysis produced no result.")

    # Locate the main contract in the analyzed application.
    main_ind = -1
    for i, c in enumerate(summary.contract_components):
        if c.name == source_input.contract_name:
            main_ind = i
            break
    if main_ind == -1:
        raise ValueError(
            f"Contract {source_input.contract_name!r} not found in the "
            "analyzed application."
        )
    contract_instance = ContractInstance(ind=main_ind, app=summary)

    prop_context = ctx.child(PROPERTIES_KEY)

    # ------------------------------------------------------------------
    # Phase 2: Per-component property extraction.
    # ------------------------------------------------------------------
    @dataclass
    class _ComponentBatch:
        feat: ContractComponentInstance
        props: list[PropertyFormulation]
        feat_ctx: WorkflowContext[ComponentGroup]

    async def _extract(idx: int) -> _ComponentBatch | None:
        feat = ContractComponentInstance(_contract=contract_instance, ind=idx)
        feat_ctx = await prop_context.child(
            _component_cache_key(feat),
            {"component": feat.component.model_dump()},
        )
        props = await run_task(
            handler_factory,
            TaskInfo(
                f"bug-{idx}", feat.component.name, AutoProvePhase.BUG_ANALYSIS,
            ),
            lambda conv: run_property_inference(
                feat_ctx,
                env,
                feat,
                refinement=conv if interactive else None,
                max_rounds=max_bug_rounds,
                backend_guidance=FOUNDRY_BACKEND_GUIDANCE,
            ),
            semaphore,
        )
        if not props:
            return None
        return _ComponentBatch(feat=feat, props=props, feat_ctx=feat_ctx)

    extraction = await asyncio.gather(*[
        _extract(i)
        for i in range(len(contract_instance.contract.components))
    ])
    batches = [b for b in extraction if b is not None]
    if not batches:
        raise ValueError("No properties extracted from any component.")

    # Filename slugs — disambiguate collisions with the index.
    raw_slugs = [slugify_filename(b.feat.component.name) for b in batches]
    slug_counts: dict[str, int] = {}
    for s in raw_slugs:
        slug_counts[s] = slug_counts.get(s, 0) + 1
    bases = [
        f"{s}_{b.feat.ind}" if slug_counts[s] > 1 else s
        for s, b in zip(raw_slugs, batches)
    ]

    # ------------------------------------------------------------------
    # Phase 3: Per-component foundry test generation.
    # ------------------------------------------------------------------
    test_dir = pathlib.Path(source_input.project_root) / "test"
    test_dir.mkdir(exist_ok=True)

    forge_runner_sem = asyncio.Semaphore(1)

    async def _generate(i: int, batch: _ComponentBatch) -> BatchFoundryResult:
        batch_child = await batch.feat_ctx.child(
            _batch_cache_key(batch.props),
            {"properties": [p.model_dump() for p in batch.props]},
        )
        if (cached := await batch_child.cache_get(GeneratedFoundryTest)) is not None:
            return cached
        batch_ctx = batch_child.abstract(CVLGeneration)

        label = f"{batch.feat.component.name} ({len(batch.props)} properties)"

        res = await run_task(
            handler_factory,
            TaskInfo(f"foundry-{i}", label, AutoProvePhase.CVL_GEN),
            lambda: batch_foundry_test_generation(
                ctx=batch_ctx,
                project_root=source_input.project_root,
                contract_name=source_input.contract_name,
                props=batch.props,
                component=batch.feat,
                env=env,
                description=label,
                forge_binary=forge_binary,
                forge_timeout_s=forge_timeout_s,
                forge_sem = forge_runner_sem
            ),
            semaphore,
        )
        if isinstance(res, GeneratedFoundryTest):
            await batch_child.cache_put(res)
        return res

    async def _generate_and_write(
        i: int, batch: _ComponentBatch,
    ) -> tuple[BatchFoundryResult, pathlib.Path | None]:
        res = await _generate(i, batch)
        if isinstance(res, GaveUp):
            return res, None
        out_path = test_dir / f"composer_{bases[i]}.t.sol"
        out_path.write_text(res.test_source)
        return res, out_path

    results = await asyncio.gather(
        *[_generate_and_write(i, b) for i, b in enumerate(batches)],
        return_exceptions=True,
    )

    written: list[pathlib.Path] = []
    failures: list[str] = []
    n_properties = 0
    for batch, result in zip(batches, results):
        n_properties += len(batch.props)
        if isinstance(result, BaseException):
            failures.append(f"{batch.feat.component.name}: {result}")
            continue
        res, path = result
        if isinstance(res, GaveUp):
            failures.append(
                f"{batch.feat.component.name}: GAVE_UP: {res.reason}"
            )
        elif path is not None:
            written.append(path)

    return FoundryPipelineResult(
        n_components=len(batches),
        n_properties=n_properties,
        written=written,
        failures=failures,
    )


type FoundryPipelineExecutor = Callable[
    [HandlerFactory[AutoProvePhase, None]], Awaitable[FoundryPipelineResult],
]
