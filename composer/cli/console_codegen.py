"""Entry point for the codegen workflow — fancy console mode (no TUI).

Mirrors ``console-autoprove`` in shape: parses the codegen CLI surface
via ``fresh_workflow_argument_parser`` and runs the workflow with the
new ``CodegenConsoleHandler`` (autoprove-styled logging + the legacy
``ConsoleHandler``'s human-interaction prompts).

The legacy ``main.py`` continues to use the older ``ConsoleHandler``
directly; this entry is the modern, "tooled" replacement.
"""

import composer.bind as _

import asyncio
import sys

from composer.input.parsing import fresh_workflow_argument_parser
from composer.workflow.services import create_llm
from composer.input.files import upload_input
from composer.workflow.executor import execute_ai_composer_workflow
from composer.ui.codegen_console import CodegenConsoleHandler
from composer.diagnostics.debug import setup_logging


async def _main() -> int:
    parser = fresh_workflow_argument_parser(sys.argv[1:])
    args = parser.parse_args()

    setup_logging(args.debug)

    llm = create_llm(args)

    print("Reading input files...")
    input_data = await upload_input(args)

    # Effective output folder for the disk write at end-of-run:
    # explicit ``--output-folder`` wins; otherwise fall back to
    # ``--source-root`` (the from-source workspace is the natural target);
    # otherwise ``None`` and the handler prompts.
    output_folder = args.output_folder or input_data.source_root

    print("Starting AI Composer workflow...")
    result = await execute_ai_composer_workflow(
        handler=CodegenConsoleHandler(
            capture_prover_output=args.prover_capture_output,
            output_folder=output_folder,
        ),
        llm=llm,
        input=input_data,
        workflow_options=args,
        memory_namespace=args.memory_namespace,
        resume_work_key=args.resume_work_key,
    )
    return result.exit_code


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
