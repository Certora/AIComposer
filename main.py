import composer.certora as _

from composer.input.parsing import fresh_workflow_argument_parser
from composer.workflow.factories import create_llm
from composer.input.files import upload_input
from composer.workflow.executor import execute_ai_composer_workflow
from composer.diagnostics.debug import setup_logging, dump_fs

def main() -> int:
    """Main entry point for the AI Composer tool."""
    parser = fresh_workflow_argument_parser()
    args = parser.parse_args()

    setup_logging(args.debug)

    llm = create_llm(args)

    if args.debug_fs:
        if not args.checkpoint_id or not args.thread_id:
            print("Need to provide checkpoint-id and thread-id")
            return 1
        return dump_fs(args, llm)

    print("Reading input files...")

    input_data = upload_input(args)

    print("Starting AI Composer workflow...")
    return execute_ai_composer_workflow(
        llm=llm,
        input=input_data,
        workflow_options=args
    )

if __name__ == "__main__":
    import sys
    sys.exit(main())