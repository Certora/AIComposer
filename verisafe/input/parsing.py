import argparse
from typing import TypeVar, Protocol, cast
from verisafe.rag.db import DEFAULT_CONNECTION as RAGDB_DEFAULT_CONNECTION
from verisafe.input.types import CommandLineArgs, ResumeArgs

ArgNS = TypeVar("ArgNS", covariant=True)

class TypedArgumentParser(Protocol[ArgNS]):
    def parse_args(self) -> ArgNS:
        ...

def _final_resume_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("updated_system", help="The new system document, if any. If not provided, the original system doc is used", nargs='?')

def _common_options(parser: argparse.ArgumentParser) -> None:
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

    # Summarization options
    parser.add_argument("--summarization-threshold", type=int, help="The number of messages that triggers summarization")

    # Database configuration for CVL manual search
    parser.add_argument("--rag-db", default=RAGDB_DEFAULT_CONNECTION, help="Database connection string for CVL manual search")

    parser.add_argument("--audit-db", help="Database connection string for audit results, given as: postgresql://user:password@localhost:5432/db_name")

    # prover options
    parser.add_argument("--prover-capture-output", action=argparse.BooleanOptionalAction, default=True, help="Whether to capture the stdout/stderr of the prover")
    parser.add_argument("--prover-keep-folders", action="store_true", help="Keep the temporary folders after the prover runs instead of deleting them")

    parser.add_argument("--debug-prompt-override", help="Append this text to the final prompt for debugging instructions to the LLM")
    parser.add_argument("--recursion-limit", type=int, help="The number of iterations of the graph to allow", default=50)
    parser.add_argument("--memory-tool", action="store_true", help="Enable Anthropic's memory tool")


def fresh_workflow_argument_parser() -> TypedArgumentParser[CommandLineArgs]:
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(description="Certora CryptoSafe Tool for Smart Contract Security")
    parser.add_argument("spec_file", help="Specification file for the smart contract")
    parser.add_argument("interface_file", help="The interface file for the smart contract")
    parser.add_argument("system_doc", help="A text document describing the system")
    _common_options(parser)

    return cast(TypedArgumentParser[CommandLineArgs], parser)


def _common_resume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--commentary", default=None, help="Commentary describing the changes to the system. If prefixed with @, assumed to be a filename from which the commentary is read")
    parser.add_argument("src_thread_id", help="The thread id from which to resume the workflow")


def resume_workflow_parser() -> TypedArgumentParser[ResumeArgs]:
    parser = argparse.ArgumentParser()
    _common_options(parser)
    sub_parse = parser.add_subparsers(dest="command", required=True)
    materialize_args = sub_parse.add_parser("materialize", help="Materialize the complete VFS from a run")
    materialize_args.add_argument("src_thread_id", help="The thread id for which to dump the VFS")
    materialize_args.add_argument("target", help="The target directory")

    resume_id_args = sub_parse.add_parser("resume-id")
    _common_resume_args(resume_id_args)
    resume_id_args.add_argument("new_spec_file", help="")
    resume_id_args.add_argument("new_spec", help="The path to the new spec file.")
    _final_resume_option(resume_id_args)

    resume_fs_args = sub_parse.add_parser("resume-dir")
    _common_resume_args(resume_fs_args)
    resume_fs_args.add_argument("working_dir", help="Path to the directory that is the new root of the VFS to use during the workflow")
    _final_resume_option(resume_fs_args)

    return cast(TypedArgumentParser[ResumeArgs], parser)
