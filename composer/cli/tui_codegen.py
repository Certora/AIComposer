"""Entry point for the codegen TUI workflow.

Replaces the legacy top-level ``tui_main.py``. Both entry points share
the same parser shape (``fresh_workflow_argument_parser`` plus
``--show-checkpoints``) so existing scripts keep working when they
swap one for the other.
"""

import composer.bind as _

import asyncio
import sys

from langchain_core.language_models.chat_models import BaseChatModel

from composer.input.parsing import fresh_workflow_argument_parser, CommandLineArgs
from composer.workflow.services import create_llm
from composer.input.files import upload_input, InputData
from composer.workflow.executor import execute_ai_composer_workflow
from composer.ui.codegen_rich import CodeGenRichApp
from composer.ui.ide_bridge import IDEBridge
from composer.diagnostics.debug import setup_logging, dump_fs
from composer.ui.tool_display import tool_context


async def run() -> int:
    parser = fresh_workflow_argument_parser(sys.argv[1:])
    parser.add_argument(  # type: ignore
        "--show-checkpoints",
        action="store_true",
        help="Show checkpoint IDs inline in the event log",
    )
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

    input_data = await upload_input(args)

    async with IDEBridge.connect() as ide:
        return await _runner(llm, input_data, args, ide)


async def _runner(
    llm: BaseChatModel,
    input_data: InputData,
    args: CommandLineArgs,
    ide: IDEBridge | None,
) -> int:
    app = CodeGenRichApp(show_checkpoints=args.show_checkpoints, ide=ide)  # type: ignore

    async def work() -> None:
        app.result = await execute_ai_composer_workflow(
            handler=app,
            llm=llm,
            input=input_data,
            workflow_options=args,
            memory_namespace=args.memory_namespace,
            resume_work_key=args.resume_work_key,
        )

    app.set_work(work)
    with tool_context():
        await app.run_async()

    return app.exit_code


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
