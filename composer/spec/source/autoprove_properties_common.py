"""Entry point shared by ``console_autoprove_properties`` and
``tui_autoprove_properties``.

CLI surface:
  - positional: ``project_root``, ``main_contract`` (``path:ContractName``),
    ``system_doc``
  - ``--properties-path`` *(required)* — file with the user's plain-English
    property list (text or PDF).
  - ``--config-path`` *(optional)* — single ``.conf`` to use directly; when
    supplied, AutoSetup is skipped.
  - plus the shared flags from ``add_base_autoprove_args`` (``--cloud``,
    ``--prover-extra-args``, ``--max-concurrent``, ``--cache-ns``,
    ``--memory-ns``, ``--interactive``, model/RAG options).
"""

import argparse
import pathlib
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Protocol, cast

from composer.io.multi_job import HandlerFactory
from composer.spec.context import get_document_input
from composer.spec.source.autoprove_services import (
    BaseAutoProveArgs, add_base_autoprove_args, autoprove_services,
)
from composer.spec.source.common_pipeline import AutoProveResult
from composer.spec.source.properties_pipeline import run_properties_pipeline
from composer.ui.autoprove_app import AutoProvePhase


class AutoProvePropertiesArgs(BaseAutoProveArgs, Protocol):
    properties_path: str
    config_path: str | None


type Executor = Callable[[HandlerFactory[AutoProvePhase, None]], Awaitable[AutoProveResult]]


@asynccontextmanager
async def _entry_point() -> AsyncIterator[Executor]:
    parser = argparse.ArgumentParser(
        description="Properties-driven auto-prove pipeline (single conf)"
    )
    add_base_autoprove_args(parser)
    parser.add_argument(
        "--properties-path", required=True,
        help="Path to the user's plain-English property list (text or PDF).",
    )
    parser.add_argument(
        "--config-path", default=None,
        help="Optional path to a single .conf file. If supplied, AutoSetup is skipped.",
    )

    args = cast(AutoProvePropertiesArgs, parser.parse_args())

    if args.config_path is not None and not args.config_path.endswith(".conf"):
        parser.error(f"--config-path must point to a .conf file, got {args.config_path}")

    properties_path = pathlib.Path(args.properties_path)
    properties = get_document_input(properties_path)
    if properties is None:
        parser.error(f"cannot read {properties_path}")

    async with autoprove_services(args, parser, thread_prefix="autoprove_properties") as svc:
        config_path: str | None = None
        if args.config_path is not None:
            resolved = (svc.project_root / args.config_path).resolve() \
                if not pathlib.Path(args.config_path).is_absolute() \
                else pathlib.Path(args.config_path).resolve()
            if not resolved.is_file():
                parser.error(f"--config-path {resolved} does not exist")
            config_path = str(resolved)

        async def runner(handler: HandlerFactory[AutoProvePhase, None]) -> AutoProveResult:
            return await run_properties_pipeline(
                llm=svc.llm,
                ctx=svc.ctx,
                source_input=svc.system_doc,
                env=svc.source_env,
                handler_factory=handler,
                properties=properties,
                config_path=config_path,
                prover_opts=svc.prover_opts,
                max_concurrent=args.max_concurrent,
                interactive=args.interactive,
            )

        yield runner
