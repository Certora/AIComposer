import argparse
from typing import TypeVar, Protocol, cast, Annotated, get_type_hints, get_origin, Any, get_args, Union
from composer.audit.db import DEFAULT_CONNECTION as AUDITDB_DEFAULT_CONNECTION
from composer.input.types import CommandLineArgs, ResumeArgs, Arg, OptionalArg, RAGDBOptions, ModelOptions, LangraphOptions

ArgNS = TypeVar("ArgNS", covariant=True)

class TypedArgumentParser(Protocol[ArgNS]):
    def parse_args(self) -> ArgNS:
        ...

def add_protocol_args(parser: argparse.ArgumentParser, protocol: type, feature_flags: set[Any] | None = None) -> None:
    """
    Introspect a Protocol and add its fields as arguments to an ArgumentParser.
    
    Args:
        parser: The ArgumentParser to configure
        protocol: A Protocol class with Annotated fields containing Arg metadata
    """
    # Get type hints with include_extras=True to preserve Annotated metadata
    hints = get_type_hints(protocol, include_extras=True)
    
    for name, type_hint in hints.items():
        # Extract Arg metadata from Annotated
        arg_spec = _extract_arg_metadata(type_hint)
        
        if arg_spec is None:
            continue

        if isinstance(arg_spec, Arg) and arg_spec.feature_flag is not None:
            feature_flag_enabled = feature_flags is not None and arg_spec.feature_flag[0] in feature_flags
            if not feature_flag_enabled:
                parser.set_defaults(**{name: arg_spec.feature_flag[1]})
                continue

        arg_kwargs = {}

        help_str : str
        match arg_spec:
            case Arg(help=h, default=d):
                help_str = h.format(default=str(d))
                arg_kwargs["default"] = d
            case OptionalArg(help=h):
                help_str = h

        arg_kwargs["help"] = help_str
        
        # Get the actual type (strip Annotated wrapper)
        actual_type = _get_actual_type(type_hint, expect_optional=isinstance(arg_spec, OptionalArg))

        assert actual_type is not None
        
        if actual_type == bool:
            arg_kwargs["action"] = "store_true"
        elif actual_type != str:
            arg_kwargs["type"] = actual_type
        
        # Add argument
        arg_name = f"--{name.replace('_', '-')}"
        parser.add_argument(arg_name, **arg_kwargs) #type: ignore


def _extract_arg_metadata(type_hint: Any) -> Arg | OptionalArg | None:
    """Extract Arg metadata from an Annotated type hint."""
    origin = get_origin(type_hint)
    
    # Check if this is an Annotated type
    if origin is Annotated:
        args = get_args(type_hint)
        # args[0] is the actual type, args[1:] are metadata
        for metadata in args[1:]:
            if isinstance(metadata, Arg) or isinstance(metadata, OptionalArg):
                return metadata
    
    return None


def _get_actual_type(type_hint: Any, expect_optional: bool = False) -> type[int] | type[str] | type[float] | type[float] | None:
    """Extract the actual type from an Annotated or Optional type hint."""
    origin = get_origin(type_hint)
    
    if origin is not Annotated:
        raise ValueError(f"Passed type hint: {type_hint} is not an annotated type")
    
    annot_type = get_args(type_hint)[0]
    if expect_optional:
        if get_origin(annot_type) is not Union:
            raise ValueError(f"Misconfiguration: an optional arg MUST wrap an Optional[T]` {annot_type}")
        union_args = get_args(annot_type)
        assert len(union_args) == 2, f"{annot_type} does not appear to be an optional"
        if union_args[0] is not type(None) and union_args[1] is not type(None):
            raise ValueError(f"{annot_type} does not appear to be an optional")
        annot_type = union_args[0] if union_args[1] is type(None) else union_args[1]
    
    if annot_type in (int, str, float, bool):
        return annot_type
    
    return None


def _final_resume_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("updated_system", help="The new system document, if any. If not provided, the original system doc is used", nargs='?')

def _common_options(parser: argparse.ArgumentParser) -> None:
    add_protocol_args(parser, ModelOptions, feature_flags=set(["memory"]))
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, LangraphOptions)

    parser.add_argument("--debug", action="store_true",
                    help="Enable debug logging output")


    parser.add_argument("--debug-fs", help="Dump the virtual FS to the provided folder and exit. Requires thread-id and checkpoint-id")

    # Summarization options
    parser.add_argument("--summarization-threshold", type=int, help="The number of messages that triggers summarization")

    parser.add_argument("--audit-db", help="Database connection string for audit results, given as: postgresql://user:password@localhost:5432/db_name", default=AUDITDB_DEFAULT_CONNECTION)

    # prover options
    parser.add_argument("--prover-capture-output", action=argparse.BooleanOptionalAction, default=True, help="Whether to capture the stdout/stderr of the prover")
    parser.add_argument("--prover-keep-folders", action="store_true", help="Keep the temporary folders after the prover runs instead of deleting them")

    parser.add_argument("--debug-prompt-override", help="Append this text to the final prompt for debugging instructions to the LLM")
    parser.add_argument("--requirements-oracle", action="append", help="Use existing files to automatically answer questions during requirement generation")
    parser.add_argument("--set-reqs", help="The name of a file containing a list of additional requirements fed in as the implementation requirements. If " \
    "this option starts with '@', taken to be the thread id of another run whose requirements should be copied.")
    parser.add_argument("--skip-reqs", action="store_true", help="If provided, no natural language requirements are added, and requirement judgment is skipped.")


def fresh_workflow_argument_parser() -> TypedArgumentParser[CommandLineArgs]:
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(description="Certora AI Composer for Smart Contract Generation")
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
    resume_id_args.add_argument("new_spec", help="The path to the new spec file.")
    _final_resume_option(resume_id_args)

    resume_fs_args = sub_parse.add_parser("resume-dir")
    _common_resume_args(resume_fs_args)
    resume_fs_args.add_argument("working_dir", help="Path to the directory that is the new root of the VFS to use during the workflow")
    _final_resume_option(resume_fs_args)

    return cast(TypedArgumentParser[ResumeArgs], parser)
