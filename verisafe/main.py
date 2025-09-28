import os
import sys
import anthropic

if __name__ == "__main__":
    import pathlib
    verisafe_dir = str(pathlib.Path(__file__).parent.parent.absolute())
    if verisafe_dir not in sys.path:
        sys.path.append(verisafe_dir)

env = os.environ.get("CERTORA")
if env is None:
    raise RuntimeError("CERTORA environment variable not set")
if env not in sys.path:
    sys.path.append(env)

from verisafe.input.parsing import setup_argument_parser
from verisafe.workflow.factories import create_llm
from verisafe.input.files import upload_input
from verisafe.workflow.executor import execute_cryptosafe_workflow
from verisafe.diagnostics.debug import setup_logging, dump_fs

def main() -> int:
    """Main entry point for the CryptoSafe tool."""
    parser = setup_argument_parser()
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

    print("Starting CryptoSafe workflow...")
    return execute_cryptosafe_workflow(
        llm=llm,
        input=input_data,
        workflow_options=args
    )

if __name__ == "__main__":
    import sys
    # client = anthropic.Anthropic()
    # for f in client.beta.files.list():
    #     client.beta.files.delete(f.id)
    sys.exit(main())