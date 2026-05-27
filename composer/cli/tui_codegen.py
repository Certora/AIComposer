"""Entry point for the codegen workflow — TUI mode (Textual).

Modern replacement for the top-level ``tui_main.py`` wrapper.
Registered as the ``tui-codegen`` script in ``[project.scripts]``."""

import composer.bind as _

import asyncio
import logging
import sys

from composer.input.parsing import fresh_workflow_argument_parser
from composer.workflow.provider import provider_for
from composer.assistant.codegen_launch import upload_input
from composer.workflow.executor import execute_ai_composer_workflow
from composer.ui.codegen_rich import CodeGenRichApp
from composer.ui.ide_bridge import IDEBridge
from composer.ui.tool_display import tool_context


async def _main() -> int:
    parser = fresh_workflow_argument_parser()
    parser.add_argument(  # type: ignore
        "--show-checkpoints",
        action="store_true",
        help="Show checkpoint IDs inline in the event log",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    input_data = await upload_input(args, provider_for(args.model))

    ide = await IDEBridge.connect()

    app = CodeGenRichApp(show_checkpoints=args.show_checkpoints, ide=ide)  # type: ignore

    async def work() -> None:
        app.result = await execute_ai_composer_workflow(
            handler=app,
            input=input_data,
            workflow_options=args,
        )

    app.set_work(work)
    with tool_context():
        await app.run_async()

    if ide is not None:
        await ide.close()

    return app.exit_code


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
