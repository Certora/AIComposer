import argparse
from typing import TypeVar, Protocol, cast
from verisafe.rag.db import DEFAULT_CONNECTION
from verisafe.input.types import CommandLineArgs

ArgNS = TypeVar("ArgNS", covariant=True)

class TypedArgumentParser(Protocol[ArgNS]):
    def parse_args(self) -> ArgNS:
        ...

def setup_argument_parser() -> TypedArgumentParser[CommandLineArgs]:
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(description="Certora CryptoSafe Tool for Smart Contract Security")
    parser.add_argument("spec_file", help="Specification file for the smart contract")
    parser.add_argument("interface_file", help="The interface file for the smart contract")
    parser.add_argument("system_doc", help="A text document describing the system")
    parser.add_argument("project_root", help="The root directory of the project")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model to use for code generation (default: claude-sonnet-4-20250514)")
    parser.add_argument("--tokens", type=int, default=10_000,
                        help="Token budget for code generation (default: 10,000)")
    parser.add_argument("--thinking-tokens", type=int, default=2048,
                        help="Token budget for thinking (default: 2048)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging output")
    parser.add_argument("--thread-id", help="The thread id to use for execution")
    parser.add_argument("--checkpoint-id", help="The checkpoint id to resume a workflow from")

    parser.add_argument("--debug-fs", action="store", help="Dump the virtual FS to the provided folder and exit. Requires thread-id and checkpoint-id")

    # Database configuration for CVL manual search
    parser.add_argument("--db-host", default=DEFAULT_CONNECTION["host"], help="Database host for CVL manual search")
    parser.add_argument("--db-port", type=int, default=DEFAULT_CONNECTION["port"], help="Database port for CVL manual search")
    parser.add_argument("--db-name", default=DEFAULT_CONNECTION["database"], help="Database name for CVL manual search")
    parser.add_argument("--db-user", default=DEFAULT_CONNECTION["user"], help="Database user for CVL manual search")
    parser.add_argument("--db-password", default=DEFAULT_CONNECTION["password"], help="Database password for CVL manual search")

    parser.add_argument("--audit-db", help="Path for the Sqlite database for audit results")

    # prover options
    parser.add_argument("--prover-capture-output", action=argparse.BooleanOptionalAction, default=True, help="Whether to capture the stdout/stderr of the prover")
    parser.add_argument("--prover-keep-folders", action="store_true", help="Keep the temporary folders after the prover runs instead of deleting them")

    parser.add_argument("--debug-prompt-override", help="Append this text to the final prompt for debugging instructions to the LLM")
    parser.add_argument("--recursion-limit", type=int, help="The number of iterations of the graph to allow", default=50)

    
    return cast(TypedArgumentParser[CommandLineArgs], parser)

