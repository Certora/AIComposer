import composer.certora as _

import asyncio

from composer.input.parsing import fresh_workflow_argument_parser
from composer.workflow.factories import create_llm
from composer.input.files import upload_input
from composer.workflow.executor import execute_ai_composer_workflow
from composer.io.codegen_rich import CodeGenRichApp
from composer.io.ide_bridge import IDEBridge
from composer.diagnostics.debug import setup_logging, dump_fs


async def main() -> int:
    """TUI entry point for the AI Composer tool."""
    parser = fresh_workflow_argument_parser()
    parser.add_argument("--show-checkpoints", action="store_true", #type: ignore
                        help="Show checkpoint IDs inline in the event log")
    args = parser.parse_args()

    setup_logging(args.debug)

    llm = create_llm(args)

    if args.debug_fs:
        if not args.checkpoint_id or not args.thread_id:
            print("Need to provide checkpoint-id and thread-id")
            return 1
        return dump_fs(args, llm)

    input_data = upload_input(args)

    ide = await IDEBridge.connect()

    app = CodeGenRichApp(show_checkpoints=args.show_checkpoints, ide=ide) #type: ignore

    async def work():
        app.result = await execute_ai_composer_workflow(
            handler=app,
            llm=llm,
            input=input_data,
            workflow_options=args
        )

    app.set_work(work)
    await app.run_async()

    if ide is not None:
        await ide.close()

    return app.exit_code


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
