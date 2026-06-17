"""
Auto-prove multi-agent pipeline orchestration (property *inference*).

The shared staged pipeline lives in ``common_pipeline.run_staged_pipeline``. This
module supplies the inference-specific property source — per-component "bug
analysis" via ``extract_all_components`` — and the thin public wrapper.
"""

import asyncio
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from composer.io.multi_job import HandlerFactory
from composer.ui.autoprove_app import AutoProvePhase

from composer.input.files import Document
from composer.spec.context import WorkflowContext, SourceCode, Properties
from composer.spec.service_host import ServiceHost
from composer.spec.system_model import HarnessedApplication
from composer.prover.core import ProverOptions
from composer.spec.source.common_pipeline import (
    AutoProveResult, AutosetupMode, UncoveredProperty, _ComponentBatch,
    extract_all_components, run_staged_pipeline,
)


@dataclass(frozen=True)
class BugAnalysisSource:
    """Property source for the inference pipeline: runs per-component "bug
    analysis" over every component. The inference source never leaves a property
    unmatched, so it returns an empty uncovered list."""
    interactive: bool
    threat_model: Document | None
    max_bug_rounds: int = 3

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
        batches = await extract_all_components(
            source_input=source_input,
            prop_context=prop_context,
            handler_factory=handler_factory,
            env=env,
            summary=harnessed_app,
            semaphore=semaphore,
            interactive=self.interactive,
            threat_model=self.threat_model,
            max_bug_rounds=self.max_bug_rounds,
        )
        return batches, []


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
    threat_model: Document | None = None,
    max_bug_rounds: int = 3,
) -> AutoProveResult:
    """Run the auto-prove multi-agent pipeline (property inference)."""
    return await run_staged_pipeline(
        llm, source_input, ctx, handler_factory, env,
        prover_opts=prover_opts,
        max_concurrent=max_concurrent,
        setup_mode=AutosetupMode(),
        property_source=BugAnalysisSource(
            interactive=interactive,
            threat_model=threat_model,
            max_bug_rounds=max_bug_rounds,
        ),
    )
