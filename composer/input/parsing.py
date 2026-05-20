import argparse
import json
import pathlib
from typing import TypeVar, Protocol, cast, Annotated, get_type_hints, get_origin, Any, get_args, Union


def _parse_prover_conf(path: str) -> dict:
    """argparse ``type=`` callback for ``--prover-conf``: resolve the
    given path to the loaded JSON dict at parse time so downstream
    consumers see a dict everywhere, never a path string."""
    return json.loads(pathlib.Path(path).read_text())
from composer.input.types import CommandLineArgs, ResumeArgs, Arg, OptionalArg, RAGDBOptions, ModelOptions, LanggraphOptions

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
    add_protocol_args(parser, LanggraphOptions)

    parser.add_argument("--debug", action="store_true",
                    help="Enable debug logging output")


    parser.add_argument("--debug-fs", help="Dump the virtual FS to the provided folder and exit. Requires thread-id and checkpoint-id")

    # Summarization options
    parser.add_argument("--summarization-threshold", type=int, help="The number of messages that triggers summarization")

    # prover options
    parser.add_argument("--prover-capture-output", action=argparse.BooleanOptionalAction, default=True, help="Whether to capture the stdout/stderr of the prover")
    parser.add_argument("--prover-keep-folders", action="store_true", help="Keep the temporary folders after the prover runs instead of deleting them")
    parser.add_argument("--local-prover", action="store_true", help="Run the prover locally instead of in the cloud")

    parser.add_argument("--debug-prompt-override", help="Append this text to the final prompt for debugging instructions to the LLM")
    parser.add_argument("--requirements-oracle", action="append", help="Use existing files to automatically answer questions during requirement generation")
    parser.add_argument("--set-reqs", help="The name of a file containing a list of additional requirements fed in as the implementation requirements. If " \
    "this option starts with '@', taken to be the thread id of another run whose requirements should be copied.")
    parser.add_argument("--skip-reqs", action="store_true", help="If provided, no natural language requirements are added, and requirement judgment is skipped.")

    parser.add_argument("--cache-namespace", dest="cache_namespace", default=None,
                        help="Namespace for cross-run caching of derived artifacts "
                             "(requirements extraction). Combined with a content hash of "
                             "the run's inputs so edits invalidate automatically. "
                             "Leave unset to disable caching entirely.")

    parser.add_argument("--description", dest="description", default=None,
                        help="Free-form label recorded on the audit run_meta slot. Lets "
                             "you find this run later by description rather than chasing "
                             "thread ids after a crash.")

    parser.add_argument("--resume-work-key", dest="resume_work_key", default=None,
                        help="Recovery key from a previously crashed run. When set, the "
                             "in-progress VFS snapshot from that run (working spec + dirty "
                             "buffer) is overlaid onto the new run before execution. "
                             "Surfaced in the crash result of a previous launch.")

    parser.add_argument("--memory-namespace", dest="memory_namespace", default=None,
                        help="Namespace for the agent's persistent memory. When set, memory "
                             "persists across thread changes (including crashes and relaunches); "
                             "pair with --resume-work-key (and optionally --thread-id) to give "
                             "the relaunched agent visibility into its own prior notes. Defaults "
                             "to the thread id when unset, which makes memory thread-local.")

    parser.add_argument("--prover-conf", default=None, type=_parse_prover_conf,
                        help="Path to a Certora config JSON file whose keys (packages, link, solc_args, "
                             "optimistic_loop, rule_sanity, etc.) are merged into every prover/typecheck "
                             "invocation. The file is loaded at argparse time so the value reaches the "
                             "rest of the pipeline as a dict. Dynamic keys (files, verify, solc) are "
                             "always set by the pipeline and override whatever is in this dict.")


def _has_input_json_flag(argv: list[str]) -> bool:
    """True if ``--input-json`` (with or without an attached ``=value``)
    appears anywhere in ``argv``. Used by the parser builder to decide
    whether to require the legacy positional triad."""
    for tok in argv:
        if tok == "--input-json" or tok.startswith("--input-json="):
            return True
    return False


def fresh_workflow_argument_parser(argv: list[str]) -> TypedArgumentParser[CommandLineArgs]:
    """Configure command line argument parser.

    Two input modes — the parser shape varies based on whether ``argv``
    contains ``--input-json``:

      1. Legacy triad (single spec): positional ``spec_file``
         ``interface_file`` ``system_doc``, REQUIRED. Good for one-off
         single-spec runs. Used when ``--input-json`` is not in ``argv``.
      2. JSON input (``--input-json path.json``): describes a contract
         task with multiple specs, optional contract name, implementation
         path, source root, and per-task prover config. The positional
         triad is NOT registered on the parser in this mode (so argparse
         won't reject extra args or demand them); the namespace still
         exposes ``spec_file`` / ``interface_file`` / ``system_doc`` as
         ``None`` via ``set_defaults`` so downstream consumers can read
         them uniformly.

    ``--input-json`` itself is registered unconditionally so it always
    appears under ``--help``, regardless of which mode is selected.

    Args:
        argv: The argv list (typically ``sys.argv[1:]``) — peeked at to
            choose between the two parser shapes. Pre-parse, so we can
            shape the parser before argparse sees the args.
    """
    parser = argparse.ArgumentParser(description="Certora AI Composer for Smart Contract Generation")

    # Always register --input-json so it shows up in --help regardless of
    # which mode the current invocation selected.
    parser.add_argument("--input-json", default=None,
                        help="Path to a JSON file describing the contract task (multi-spec mode). "
                             "When set, the positional spec/interface/system-doc triad is omitted "
                             "from this invocation's parser shape. "
                             "Schema: {name?, interface, implementation_path?, specs[], system_doc, "
                             "source_root?, prover_conf?}. Relative paths resolve against the JSON "
                             "file's directory.")

    if _has_input_json_flag(argv):
        # JSON mode: don't register the positionals (would conflict with
        # the JSON file's contents and confuse --help). Surface the
        # attributes as ``None`` so downstream code can read them
        # uniformly without ``hasattr`` guards.
        parser.set_defaults(
            spec_file=None,
            interface_file=None,
            system_doc=None,
            contract_name=None,
            implementation_path=None
        )
    else:
        # Legacy triad mode: positionals are REQUIRED; argparse enforces.
        parser.add_argument("spec_file",
                            help="Specification file for the smart contract (legacy single-spec mode)")
        parser.add_argument("interface_file",
                            help="The interface file for the smart contract (legacy single-spec mode)")
        parser.add_argument("system_doc",
                            help="A text document describing the system (legacy single-spec mode)")
        
        parser.add_argument("--contract-name", default=None,
                        help="Solidity identifier of the contract being implemented. Informational; "
                             "agents may still choose their own target_contract per prover run. "
                             "In JSON mode, taken from the JSON's name field instead.")
        parser.add_argument("--implementation-path", default=None,
                            help="Suggested VFS path for the generated Solidity implementation. "
                                "In JSON mode, taken from the JSON's implementation_path field instead.")


    parser.add_argument("--source-root", default=None,
                        help="Path to an existing codebase to use as the VFS underlay. "
                             "When set, agents see existing files read-only and can layer new files on top. "
                             "In JSON mode, this overrides the JSON's source_root field instead")
    parser.add_argument("--output-folder", dest="output_folder", default=None,
                        help="Directory to write the generated source files into on success. "
                             "VFS paths land underneath this directory (e.g. a VFS path of "
                             "``certora/MyContract.sol`` becomes ``<output-folder>/certora/MyContract.sol``). "
                             "Defaults to ``--source-root`` when that is set; otherwise the console handler "
                             "prompts for a target at the end of the run.")
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
    resume_id_args.add_argument(
        "new_spec",
        nargs='+',
        help=(
            "Path(s) to the new spec file(s) to apply on resume. If the prior "
            "run had exactly one spec, a bare file path works (it's mapped to "
            "that single spec's VFS path). If the prior run had multiple specs, "
            "each argument must use the form ``<vfs_path>=<local_file>`` to "
            "disambiguate. Multiple arguments allowed."
        ),
    )
    _final_resume_option(resume_id_args)

    resume_fs_args = sub_parse.add_parser("resume-dir")
    _common_resume_args(resume_fs_args)
    resume_fs_args.add_argument("working_dir", help="Path to the directory that is the new root of the VFS to use during the workflow")
    _final_resume_option(resume_fs_args)

    return cast(TypedArgumentParser[ResumeArgs], parser)
