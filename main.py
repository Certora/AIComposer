"""DEPRECATED. Use ``console-codegen`` (registered in ``[project.scripts]``).

This entry continues to work and intentionally retains the legacy
``ConsoleHandler`` (per-line prover stdout streaming, full state-update
dumps, checkpoint id printing). The modern ``console-codegen`` script
runs the same workflow against the autoprove-styled
``CodegenConsoleHandler`` instead.
"""

import composer.bind as _

import asyncio
import sys
import warnings

from composer.input.parsing import fresh_workflow_argument_parser
from composer.workflow.services import create_llm
from composer.input.files import upload_input
from composer.workflow.executor import execute_ai_composer_workflow
from composer.ui.console import ConsoleHandler
from composer.diagnostics.debug import setup_logging, dump_fs


warnings.warn(
    "main.py is deprecated; use the `console-codegen` console script for the "
    "modern handler, or import `composer.ui.console.ConsoleHandler` directly "
    "if you specifically need the legacy logging shape.",
    DeprecationWarning,
    stacklevel=2,
)


async def main() -> int:
    """Legacy console entry. Uses the older ``ConsoleHandler`` deliberately;
    callers wanting the autoprove-styled output should run ``console-codegen``.
    """
    parser = fresh_workflow_argument_parser(sys.argv[1:])
    args = parser.parse_args()

    setup_logging(args.debug)

    llm = create_llm(args)

    if args.debug_fs:
        if not args.checkpoint_id or not args.thread_id:
            print("Need to provide checkpoint-id and thread-id")
            return 1
        return await dump_fs(
            args.debug_fs, thread_id=args.thread_id, checkpoint_id=args.checkpoint_id
        )

    print("Reading input files...")

    input_data = await upload_input(args)

    print("Starting AI Composer workflow...")
    result = await execute_ai_composer_workflow(
        handler=ConsoleHandler(capture_prover_output=args.prover_capture_output),
        llm=llm,
        input=input_data,
        workflow_options=args,
        memory_namespace=args.memory_namespace,
        resume_work_key=args.resume_work_key,
    )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
