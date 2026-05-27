"""Entry point for the codegen workflow — console mode (no TUI).

Modern replacement for the top-level ``main.py`` wrapper. Registered
as the ``console-codegen`` script in ``[project.scripts]``."""

import composer.bind as _

import asyncio
import logging
import sys

from composer.input.parsing import fresh_workflow_argument_parser
from composer.workflow.provider import provider_for
from composer.assistant.codegen_launch import upload_input
from composer.workflow.executor import execute_ai_composer_workflow
from composer.ui.console import ConsoleHandler


async def _main() -> int:
    parser = fresh_workflow_argument_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    print("Reading input files...")
    input_data = await upload_input(args, provider_for(args.model))

    print("Starting AI Composer workflow...")
    result = await execute_ai_composer_workflow(
        handler=ConsoleHandler(capture_prover_output=args.prover_capture_output),
        input=input_data,
        workflow_options=args,
    )
    return result.exit_code


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
