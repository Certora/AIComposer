"""Entry point for the auto-prove multi-agent pipeline TUI."""

import argparse
import pathlib
from contextlib import asynccontextmanager
from typing import cast, AsyncIterator, Protocol, Callable, Awaitable

from composer.spec.context import get_document_input
from composer.spec.source.autoprove_services import (
    BaseAutoProveArgs, add_base_autoprove_args, autoprove_services,
)
from composer.spec.source.pipeline import run_autoprove_pipeline, AutoProveResult
from composer.ui.autoprove_app import AutoProvePhase

from composer.io.multi_job import HandlerFactory


class AutoProveArgs(BaseAutoProveArgs, Protocol):
    threat_model: str
    max_bug_rounds: int


type Executor = Callable[[HandlerFactory[AutoProvePhase, None]], Awaitable[AutoProveResult]]


@asynccontextmanager
async def _entry_point() -> AsyncIterator[Executor]:
    parser = argparse.ArgumentParser(description="Auto-prove multi-agent pipeline TUI")
    add_base_autoprove_args(parser)
    parser.add_argument(
        "--threat-model", type=str, default=None,
        help="Path to a 'thread' model (text or pdf) with which to seed the property extraction process",
    )
    parser.add_argument(
        "--max-bug-rounds", type=int, default=3,
        help="Maximum number of bug-extraction rounds run per component during property analysis (default: 3)",
    )

    args = cast(AutoProveArgs, parser.parse_args())

    threat_model = (
        get_document_input(pathlib.Path(threat_path))
        if (threat_path := args.threat_model) is not None
        else None
    )

    async with autoprove_services(args, parser) as svc:
        async def runner(handler: HandlerFactory[AutoProvePhase, None]) -> AutoProveResult:
            return await run_autoprove_pipeline(
                llm=svc.llm,
                ctx=svc.ctx,
                source_input=svc.system_doc,
                env=svc.source_env,
                handler_factory=handler,
                prover_opts=svc.prover_opts,
                max_concurrent=args.max_concurrent,
                interactive=args.interactive,
                threat_model=threat_model,
                max_bug_rounds=args.max_bug_rounds,
            )

        yield runner
