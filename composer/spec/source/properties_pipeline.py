"""
Known-properties auto-prove pipeline.

A fork of ``run_autoprove_pipeline`` (``pipeline.py``) for the case where the
properties to prove are *already known* (supplied as a YAML file) rather than
inferred from the source. The differences from the inference pipeline:

* per-component "bug analysis" / property inference is skipped;
* a new **Formalize Properties** phase maps each known property onto a discovered
  component (``formalize_properties``);
* ``--skip-setup`` replaces autosetup + harness creation with a user-supplied
  ``.conf`` and a single summary import (mirrors ``direct_pipeline.py``).

System component analysis and structural-invariant discovery/generation are kept
from the full pipeline. Coverage is tracked per ``property_id`` end-to-end: a
property lost at the formalize phase (no component match) or at the CVL phase
(agent skip / batch gave up / error) lands in a single consolidated
``certora/properties/uncovered_properties.json`` and in the returned
``AutoProveResult.uncovered``.
"""

import asyncio
import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import (
    TaskInfo, HandlerFactory, run_task,
)
from composer.spec.source.autosetup import SetupSuccess
from composer.ui.autoprove_app import AutoProvePhase

from composer.spec.context import (
    WorkflowContext, SourceCode, CacheKey, Properties, CVLGeneration,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.gen_types import (
    CVLResource, CERTORA_DIR, SPECS_DIR, certora_relative_to_project, under_project,
)
from composer.spec.util import ensure_dir
from composer.spec.source.harness import run_harness_creation, run_autosetup_phase, ContractSetup
from composer.spec.source.system_analysis import run_component_analysis
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.source.summarizer import setup_summaries
from composer.spec.system_model import (
    HarnessedApplication, SourceExplicitContract, SourceApplication,
    HarnessedExplicitContract, SourceExternalActor, HarnessDefinition,
)
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.source.prover import get_prover_tool, dump_final_conf
from composer.prover.core import ProverOptions
from composer.spec.source.struct_invariant import get_invariant_formulation
from composer.spec.source.author import batch_cvl_generation, GaveUp
from composer.spec.source.common_pipeline import (
    generate_all_component_cvl, AutoProveResult, UncoveredProperty, Unmatched,
    dump_properties, dump_property_rules, _ComponentBatch,
)
from composer.spec.source.formalize_properties import formalize_properties, UnmatchedProperty
from composer.spec.source.known_properties import KnownProperties
from composer.spec.source.task_ids import (
    SYSTEM_ANALYSIS_TASK_ID, HARNESS_TASK_ID, AUTOSETUP_TASK_ID,
    SUMMARIES_TASK_ID, INVARIANTS_TASK_ID, INVARIANT_CVL_TASK_ID, FORMALIZE_TASK_ID,
)

_logger = logging.getLogger(__name__)

PROPERTIES_KEY = CacheKey[None, Properties]("properties")
INV_CVL_KEY = CacheKey[None, GeneratedCVL]("invariant-cvl")


def _dump_uncovered(
    project_root: str,
    known: KnownProperties,
    uncovered: list[UncoveredProperty],
) -> None:
    """Write the consolidated coverage report
    ``certora/properties/uncovered_properties.json`` =
    ``[{property_id, property_desc, component, reason}]``. ``component`` is
    derived from the reason variant (formalize unmatched ⇒ no component)."""
    desc_by_id = {p.property_id: p.property_desc for p in known.properties}
    payload: list[dict[str, str | None]] = []
    for uc in uncovered:
        component = None if isinstance(uc.reason, Unmatched) else uc.reason.feat.component.name
        payload.append({
            "property_id": uc.property_id,
            "property_desc": desc_by_id.get(uc.property_id, ""),
            "component": component,
            "reason": str(uc.reason),
        })
    properties_dir = ensure_dir(under_project(project_root, CERTORA_DIR) / "properties")
    (properties_dir / "uncovered_properties.json").write_text(json.dumps(payload, indent=2))


def _build_harnessed_app(
    s: SourceApplication,
    contract_to_harness: dict[str, list[HarnessDefinition]],
) -> HarnessedApplication:
    """Build a ``HarnessedApplication`` from a component analysis, attaching the
    given harnesses to each explicit contract (empty dict ⇒ no harnesses)."""
    comp: list[SourceExternalActor | HarnessedExplicitContract] = []
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
            harnesses=contract_to_harness.get(c.name, []),
        ))
    return HarnessedApplication(
        application_type=s.application_type,
        description=s.description,
        components=comp,
    )


async def run_properties_pipeline(
    llm: BaseChatModel,
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    env: SourceEnvironment,
    *,
    known_properties: KnownProperties,
    prover_opts: ProverOptions,
    max_concurrent: int = 4,
    skip_setup: bool = False,
    conf_path: str | None = None,
    summary_path: str | None = None,
) -> AutoProveResult:
    """Run the known-properties auto-prove pipeline."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Phase 1: Component analysis
    # ------------------------------------------------------------------
    s = await run_task(
        handler_factory,
        TaskInfo(SYSTEM_ANALYSIS_TASK_ID, "System Analysis", AutoProvePhase.COMPONENT_ANALYSIS),
        lambda: run_component_analysis(ctx, source_input, env=env),
    )
    if s is None:
        raise ValueError("System analysis failed")

    # ------------------------------------------------------------------
    # Phase 2: Harness + harnessed_app. Default mode runs harness creation
    # (required by autosetup, which reads harness files from disk); --skip-setup
    # builds the harnessed app with empty harness lists.
    # ------------------------------------------------------------------
    sys_desc = None
    if not skip_setup:
        sys_desc = await run_task(
            handler_factory,
            TaskInfo(HARNESS_TASK_ID, "Harness Creation", AutoProvePhase.HARNESS),
            lambda: run_harness_creation(ctx, source_input, env, s),
        )
        contract_to_harness: dict[str, list[HarnessDefinition]] = {}
        for c in sys_desc.transitive_closure:
            if not c.harness_definition:
                continue
            contract_to_harness.setdefault(c.harness_definition.harness_of, []).append(
                HarnessDefinition(name=c.name, path=c.path)
            )
        harnessed_app = _build_harnessed_app(s, contract_to_harness)
    else:
        harnessed_app = _build_harnessed_app(s, {})

    # Prover tool is stateless with respect to setup; build it once and share.
    prover_tool = get_prover_tool(
        llm, source_input.contract_name,
        source_input.project_root, prover_opts=prover_opts,
    )

    # ------------------------------------------------------------------
    # Phase 3: setup -> prover_config + resources.
    #   default:      autosetup (+ custom summaries)
    #   --skip-setup: user-supplied .conf + single summary import
    # ------------------------------------------------------------------
    async def stream_autosetup() -> tuple[dict, list[CVLResource]]:
        assert sys_desc is not None
        sd = sys_desc  # non-None alias so the narrowing reaches the nested closures
        setup_config = await run_task(
            handler_factory,
            TaskInfo(AUTOSETUP_TASK_ID, "AutoSetup", AutoProvePhase.AUTOSETUP),
            lambda: run_autosetup_phase(ctx, source_input, sd, s, prover_opts),
        )
        res: list[CVLResource] = [
            CVLResource(
                path=certora_relative_to_project(setup_config.summaries_path),
                required=True,
                description="AutoSetup-generated summaries",
                sort="import",
            ),
        ]
        if sd.erc20_contracts or sd.external_interfaces:
            summary_resource = await run_task(
                handler_factory,
                TaskInfo(SUMMARIES_TASK_ID, "Custom Summaries", AutoProvePhase.SUMMARIES),
                lambda: setup_summaries(
                    ctx=ctx,
                    app=harnessed_app,
                    config=ContractSetup(system_description=sd, config=setup_config),
                    env=env,
                    source=source_input,
                ),
            )
            res.append(summary_resource)
        return setup_config.prover_config, res

    async def stream_invariants():
        return await run_task(
            handler_factory,
            TaskInfo(INVARIANTS_TASK_ID, "Structural Invariants", AutoProvePhase.INVARIANTS),
            lambda: get_invariant_formulation(ctx, source_input, env, harnessed_app),
        )

    async def stream_formalize() -> tuple[list[_ComponentBatch], list[UnmatchedProperty]]:
        return await run_task(
            handler_factory,
            TaskInfo(FORMALIZE_TASK_ID, "Formalize Properties", AutoProvePhase.FORMALIZE),
            lambda: formalize_properties(
                ctx, source_input, env, harnessed_app, known_properties,
                ctx.child(PROPERTIES_KEY),
            ),
        )

    if not skip_setup:
        (prover_config, resources), invariants, (component_batches, unmatched) = await asyncio.gather(
            stream_autosetup(),
            stream_invariants(),
            stream_formalize(),
        )
    else:
        # User-facing validation lives at CLI parse time (see _properties_entry_point
        # in autoprove_common.py). This is only a caller invariant / type narrowing.
        assert conf_path is not None and summary_path is not None, (
            "--skip-setup requires both a conf and a summary path"
        )
        prover_config = json.load(open(conf_path, "r"))
        resources = [
            CVLResource(
                path=certora_relative_to_project(summary_path),
                required=True,
                description="User-supplied summaries",
                sort="import",
            ),
        ]
        invariants, (component_batches, unmatched) = await asyncio.gather(
            stream_invariants(),
            stream_formalize(),
        )

    certora_dir = under_project(source_input.project_root, CERTORA_DIR)
    uncovered = [
        UncoveredProperty(property_id=up.property_id, reason=Unmatched(up.reason))
        for up in unmatched
    ]

    # If nothing matched, there is no CVL to generate: report the unmatched
    # properties as uncovered and stop here (instead of raising).
    if not component_batches:
        _logger.warning("No properties were mapped to a component; skipping CVL generation.")
        _dump_uncovered(source_input.project_root, known_properties, uncovered)
        return AutoProveResult(n_components=0, n_properties=0, uncovered=uncovered)

    # ------------------------------------------------------------------
    # Stage 1: structural-invariant CVL (writes invariants.spec, imported as
    # assumable preconditions by the per-component CVLs).
    # ------------------------------------------------------------------
    if invariants.inv:
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
        dump_properties(certora_dir, "invariants", inv_props)

        if cached_inv_cvl is not None:
            inv_cvl = cached_inv_cvl
        else:
            inv_cvl_result = await run_task(
                handler_factory,
                TaskInfo(INVARIANT_CVL_TASK_ID, "Invariant CVL", AutoProvePhase.CVL_GEN),
                lambda: batch_cvl_generation(
                    ctx=inv_cvl_ctx.abstract(CVLGeneration),
                    component=None,
                    props=inv_props,
                    env=env,
                    init_config=prover_config,
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

        ensure_dir(certora_dir / "specs")
        inv_spec_path = SPECS_DIR / "invariants.spec"
        under_project(source_input.project_root, inv_spec_path).write_text(inv_cvl.cvl)
        dump_property_rules(certora_dir, "invariants", inv_cvl.property_rules)
        dump_final_conf(
            project_root=source_input.project_root,
            main_contract=source_input.contract_name,
            task_id=INVARIANT_CVL_TASK_ID,
            spec_path=inv_spec_path,
            conf=inv_cvl.conf,
        )
        resources.append(CVLResource(
            path=inv_spec_path,
            required=False,
            description="Structural invariants that may be assumed as preconditions",
            sort="import",
        ))

    # ------------------------------------------------------------------
    # Stage 2: per-component CVL (parallel, semaphore-bounded).
    # ------------------------------------------------------------------
    result = await generate_all_component_cvl(
        source_input=source_input,
        component_batches=component_batches,
        handler_factory=handler_factory,
        env=env,
        prover_tool=prover_tool,
        prover_config=prover_config,
        resources=resources,
        semaphore=semaphore,
    )

    # ------------------------------------------------------------------
    # Consolidate coverage: merge formalize-stage unmatched + CVL-stage uncovered
    # into the single report and the returned result.
    # ------------------------------------------------------------------
    result.uncovered.extend(uncovered)
    _dump_uncovered(source_input.project_root, known_properties, result.uncovered)
    return result
