"""
Known-properties auto-prove pipeline.

Shares the staged orchestrator (``common_pipeline.run_staged_pipeline``) with the
inference pipeline. The differences this module supplies:

* the property source is **Formalize Properties** (``formalize_properties``), a
  pure mapping of each known property onto a discovered component, rather than
  per-component "bug analysis";
* ``--skip-setup`` selects ``UserSuppliedMode`` (a user-supplied ``.conf`` + a
  single summary import) instead of ``AutosetupMode``;
* a coverage reporter dumps the consolidated
  ``certora/properties/uncovered_properties.json``.

Coverage is tracked per ``property_id`` end-to-end: a property lost at the
formalize phase (no component match) or at the CVL phase (agent skip / batch gave
up / error) lands in that single report and in the returned
``AutoProveResult.uncovered``.
"""

import asyncio
import json

from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import TaskInfo, HandlerFactory, run_task
from composer.ui.autoprove_app import AutoProvePhase

from composer.spec.context import WorkflowContext, SourceCode, Properties
from composer.spec.gen_types import CERTORA_DIR, under_project
from composer.spec.util import ensure_dir
from composer.spec.source.source_env import ServiceHost
from composer.spec.system_model import (
    HarnessedApplication, ContractInstance, ContractComponentInstance,
)
from composer.prover.core import ProverOptions
from composer.spec.source.common_pipeline import (
    AutoProveResult, AutosetupMode, SetupMode, UserSuppliedMode, UncoveredProperty,
    Unmatched, _ComponentBatch, _main_contract_index, build_component_batch,
    run_staged_pipeline,
)
from composer.spec.source.formalize_properties import formalize_properties
from composer.spec.source.known_properties import KnownProperties
from composer.spec.source.task_ids import FORMALIZE_TASK_ID


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
            "property_desc": desc_by_id[uc.property_id],
            "component": component,
            "reason": str(uc.reason),
        })
    properties_dir = ensure_dir(under_project(project_root, CERTORA_DIR) / "properties")
    (properties_dir / "uncovered_properties.json").write_text(json.dumps(payload, indent=2))


class FormalizeSource:
    """Property source for the known-properties pipeline: maps each known
    property onto a component (pure mapping), then builds the per-component
    batches. Unmatched known properties are surfaced as uncovered."""

    def __init__(self, known_properties: KnownProperties):
        self._known = known_properties

    async def __call__(
        self,
        *,
        ctx: WorkflowContext[None],
        source_input: SourceCode,
        prop_context: WorkflowContext[Properties],
        handler_factory: HandlerFactory[AutoProvePhase, None],
        env: ServiceHost,
        harnessed_app: HarnessedApplication,
        semaphore: asyncio.Semaphore,
    ) -> tuple[list[_ComponentBatch], list[UncoveredProperty]]:
        props_by_component, unmatched = await run_task(
            handler_factory,
            TaskInfo(FORMALIZE_TASK_ID, "Formalize Properties", AutoProvePhase.FORMALIZE),
            lambda: formalize_properties(ctx, source_input, env, harnessed_app, self._known),
        )

        # Resolve each ComponentRef to a ContractComponentInstance and wire up the
        # per-component batch contexts.
        contract_instance = ContractInstance(
            _main_contract_index(harnessed_app, source_input.contract_name), app=harnessed_app
        )
        batches = [
            await build_component_batch(
                prop_context=prop_context,
                feat=ContractComponentInstance(_contract=contract_instance, ind=ref.index),
                props=props,
            )
            for ref, props in props_by_component.items()
        ]
        uncovered = [
            UncoveredProperty(property_id=up.property_id, reason=Unmatched(up.reason))
            for up in unmatched
        ]
        return batches, uncovered


async def run_properties_pipeline(
    llm: BaseChatModel,
    source_input: SourceCode,
    ctx: WorkflowContext[None],
    handler_factory: HandlerFactory[AutoProvePhase, None],
    host: ServiceHost,
    *,
    known_properties: KnownProperties,
    prover_opts: ProverOptions,
    max_concurrent: int = 4,
    skip_setup: bool = False,
    conf_path: str | None = None,
    summary_path: str | None = None,
) -> AutoProveResult:
    """Run the known-properties auto-prove pipeline."""
    setup_mode: SetupMode
    if skip_setup:
        assert conf_path is not None and summary_path is not None, (
            "--skip-setup requires both a conf and a summary path"
        )
        setup_mode = UserSuppliedMode(conf_path=conf_path, summary_path=summary_path)
    else:
        setup_mode = AutosetupMode()

    return await run_staged_pipeline(
        llm, source_input, ctx, handler_factory, host,
        prover_opts=prover_opts,
        max_concurrent=max_concurrent,
        setup_mode=setup_mode,
        property_source=FormalizeSource(known_properties),
        coverage_reporter=lambda result, project_root: _dump_uncovered(
            project_root, known_properties, result.uncovered
        ),
    )
