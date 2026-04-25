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


@dataclass
class UploadedFile:
    """
    Represents a file uploaded with claude's file API
    """
    file_id: str
    basename: str

    path: str

    def to_document_dict(self) -> dict:
        """Convert to document dictionary format for LangGraph"""
        return {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": self.file_id
            }
        }

    def read(self) -> str:
        with open(self.path, 'r') as f:
            return f.read()
        
    @property
    def string_contents(self) -> str:
        return self.read()

    @property
    def bytes_contents(self) -> bytes:
        with open(self.path, 'rb') as f:
            return f.read()

class InMemoryFile:
    def __init__(self, name: str, contents: str | bytes):
        self.basname = name
        self.bytes_contents = contents if isinstance(contents, bytes) else contents.encode("utf-8")

class NativeFS:
    def __init__(self, p: pathlib.Path):
        self.where = p

    @property
    def bytes_contents(self) -> bytes:
        return self.where.read_bytes()
    
    @property
    def basename(self) -> str:
        return self.where.name
    
    @property
    def string_contents(self) -> str:
        return self.where.read_text()
    
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

    prover_conf: Optional[str]


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
    file: UploadedFile
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
    system_doc: UploadedFile
    intf: UploadedFile
    source_root: Optional[str] = None
    contract_name: Optional[str] = None
    implementation_path: Optional[str] = None
    prover_conf: Optional[dict] = None

    @property
    def spec_vfs_paths(self) -> list[str]:
        return [s.vfs_path for s in self.specs]


class ResumeInput(Protocol):
    @property
    def comments(self) -> Optional[str]:
        ...

    @property
    def new_system(self) -> Optional[NativeFS]:
        ...

    @property
    def thread_id(self) -> str:
        ...

@dataclass
class ResumeIdData:
    thread_id: str
    # Mapping from VFS path → new spec content. Only paths present here are
    # updated on resume; other specs keep their prior state. For single-spec
    # resumes from legacy CLI callers, this dict carries one entry.
    new_specs: dict[str, NativeFS]
    comments: Optional[str]
    new_system: Optional[NativeFS]

@dataclass
class ResumeFSData:
    thread_id: str
    file_path: str
    comments: Optional[str]
    new_system: Optional[NativeFS]
