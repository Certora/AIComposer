from typing import Optional, Protocol, Literal, Any, Annotated
from composer.rag.db import DEFAULT_CONNECTION as RAGDB_DEFAULT_CONNECTION
import pathlib
from dataclasses import dataclass


@dataclass
class BasicArg:
    help: str

@dataclass
class Arg(BasicArg):
    default: Any | None
    feature_flag: tuple[Any, Any] | None = None

@dataclass
class OptionalArg(BasicArg):
    pass


# ---------------------------------------------------------------------------
# File-content Protocols
# ---------------------------------------------------------------------------


class InputFileLike(Protocol):
    """A file that may or may not be representable as a text string.

    ``string_contents`` returns ``None`` for files whose bytes aren't
    text (e.g. PDF system documents). Callers that need a guaranteed
    string body should narrow to ``TextInputFile`` at the type level
    rather than testing for ``None`` at every call site.

    For LLM ingestion, ``to_document_dict`` is the universal mechanism —
    a content block of whatever shape the implementation has access to
    (Files-API reference, inline text, inline base64). The caller
    doesn't care which.
    """

    @property
    def basename(self) -> str:
        ...

    @property
    def bytes_contents(self) -> bytes:
        ...

    @property
    def string_contents(self) -> Optional[str]:
        """The file's text body, or ``None`` if the file is binary."""
        ...

    def to_document_dict(self) -> dict:
        ...


class TextInputFile(InputFileLike, Protocol):
    """An ``InputFileLike`` known statically to be text — ``string_contents``
    is guaranteed non-None. Use this parameter type for specs, interfaces,
    and other places where binary input would be a programming error."""

    @property
    def string_contents(self) -> str:
        ...

class RAGDBOptions(Protocol):
    # database options
    rag_db: Annotated[str, Arg(
        help="Database connection string for CVL manual search",
        default=RAGDB_DEFAULT_CONNECTION
    )]

class LanggraphOptions(Protocol):
    checkpoint_id: Annotated[Optional[str], OptionalArg(help="The checkpoint id to resume a workflow from")]
    thread_id: Annotated[Optional[str], OptionalArg(help="The checkpoint id to resume a workflow from")]
    recursion_limit: Annotated[int, Arg(
        help="The number of iterations of the graph to allow (default: {default}",
        default=50
    )]


class WorkflowOptions(RAGDBOptions, LanggraphOptions, Protocol):
    prover_capture_output: bool
    prover_keep_folders: bool
    local_prover: bool

    debug_prompt_override: Optional[str]

    recursion_limit: int
    summarization_threshold: Optional[int]

    requirements_oracle: list[str]
    set_reqs: Optional[str]
    skip_reqs: bool

    # Pre-parsed at the CLI boundary: ``--prover-conf <path>`` resolves
    # to a dict at argparse-time (``type=`` callback in
    # ``input/parsing.py``), so consumers always see the merged dict
    # form here, never a path string. ``None`` means no overrides.
    prover_conf: Optional[dict]


class ModelOptionsBase(Protocol):
    """Read-only view of model options. thinking_tokens may be None to disable thinking."""
    @property
    def model(self) -> str: ...
    @property
    def tokens(self) -> int: ...
    @property
    def thinking_tokens(self) -> int | None: ...
    @property
    def memory_tool(self) -> bool: ...

    @property
    def interleaved_thinking(self) -> bool: ...



class ModelOptions(Protocol):
    model: Annotated[str, Arg(
        help="Model to use for code generation (default: {default})", default="claude-opus-4-6"
        )]
    tokens: Annotated[int, Arg(
        help="Token budget for code generation (default: {default})",
        default=10_000
    )]
    thinking_tokens: Annotated[int, Arg(
        help="Token budget for thinking (default: {default})",
        default=2048
    )]
    memory_tool: Annotated[bool, Arg(
        help="Enable Anthropic's memory tool",
        default=None,
        feature_flag=("memory", True) # default to use if this is not exposed on command line
    )]
    interleaved_thinking: Annotated[bool, Arg(
        help="Enable interleaved thinking mode (default: {default})",
        default=False
    )]

class UploadPaths(Protocol):
    """Legacy CLI triad (optional, single-spec): one spec, one interface, one
    system doc. All fields may be ``None`` when ``--input-json`` is supplied
    via ``InputJSONPath``.
    """
    spec_file: Optional[str]
    interface_file: Optional[str]
    system_doc: Optional[str]
    source_root: Optional[str]
    contract_name: Optional[str]
    implementation_path: Optional[str]


class InputJSONPath(Protocol):
    """Alternative CLI shape: a single JSON file describing the contract task.
    Mutually exclusive with the legacy triad."""
    input_json: Optional[str]


class CommandLineArgs(WorkflowOptions, ModelOptions, UploadPaths, InputJSONPath, Protocol):
    debug_fs: str

    debug: bool

class ResumeArgs(WorkflowOptions, ModelOptions, Protocol):
    # common
    src_thread_id: str
    command: Literal["materialize", "resume-dir", "resume-id"]

    # materialize
    target: str

    # common resume
    commentary: Optional[str]
    updated_system: Optional[str]

    # resume-id: list of new spec entries. Bare path → mapped to the prior
    # run's single registered spec (error if the prior run had multiple).
    # ``<vfs_path>=<local_file>`` → explicitly targets a specific spec.
    new_spec: list[str]

    # resume-fs
    working_dir: str


@dataclass
class SpecInput:
    """A single spec file paired with the VFS path at which it should be
    materialized inside the workflow's virtual filesystem."""
    file: TextInputFile
    vfs_path: str


@dataclass
class InputData:
    """Normalized codegen workflow input for a single contract task.

    Carries one or more specs (all describing the same contract), the contract's
    interface, the surrounding system document, and optional metadata (contract
    name, expected implementation path, per-task prover config). The VFS path
    for each spec is resolved at ``upload_input`` time so downstream code
    (executor, prover tool, propose_spec_change) can key off a stable location.

    Greenfield inputs land at ``certora/<basename>``; inputs given with a
    ``source_root`` land at their workspace-relative path.
    """
    specs: list[SpecInput]
    # ``system_doc`` may be binary (e.g. PDF); ``intf`` is always text
    # (Solidity interface).
    system_doc: InputFileLike
    intf: TextInputFile
    kickstart_context: str | None
    source_root: Optional[str] = None
    contract_name: Optional[str] = None
    implementation_path: Optional[str] = None
    # ``prover_conf`` is *not* on ``InputData`` — prover overrides are
    # an orthogonal runtime concern and travel via the workflow-options
    # channel (CLI ``--prover-conf`` / ``CommonCodeGen.prover_conf``)
    # straight to the executor, not on the input bundle.

    @property
    def spec_vfs_paths(self) -> list[str]:
        return [s.vfs_path for s in self.specs]


class ResumeInput(Protocol):
    @property
    def comments(self) -> Optional[str]:
        ...

    @property
    def new_system(self) -> Optional[InputFileLike]:
        ...

    @property
    def thread_id(self) -> str:
        ...

@dataclass
class ResumeIdData:
    thread_id: str
    # Mapping from VFS path → new spec content. Only paths present here are
    # updated on resume; other specs keep their prior state. For single-spec
    # resumes from legacy CLI callers, this dict carries one entry. Specs
    # are guaranteed text — the construction site uploads via
    # ``FileUploader.upload_text_file_if_needed`` so the type system
    # carries the guarantee.
    new_specs: dict[str, TextInputFile]
    comments: Optional[str]
    new_system: Optional[InputFileLike]

@dataclass
class ResumeFSData:
    thread_id: str
    file_path: str
    comments: Optional[str]
    new_system: Optional[InputFileLike]
