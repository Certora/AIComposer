from typing import Optional, Protocol, Literal
import pathlib
from dataclasses import dataclass

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
    rag_db: str

class WorkflowOptions(RAGDBOptions):
    prover_capture_output: bool
    prover_keep_folders: bool

    debug_prompt_override: Optional[str]

    checkpoint_id: Optional[str]
    thread_id: Optional[str]
    recursion_limit: int
    audit_db: str
    summarization_threshold: Optional[int]

    requirements_oracle: list[str]
    set_reqs: Optional[str]
    skip_reqs: bool


class ModelOptions(Protocol):
    model: str
    tokens: int
    thinking_tokens: int
    memory_tool: bool

class CommandLineArgs(WorkflowOptions, ModelOptions):
    spec_file: str
    interface_file: str
    system_doc: str
    debug_fs: str

    debug: bool

class ResumeArgs(WorkflowOptions, ModelOptions):
    # common
    src_thread_id: str
    command: Literal["materialize", "resume-dir", "resume-id"]

    # materialize
    target: str

    # common resume
    commentary: Optional[str]
    updated_system: Optional[str]

    # resume-id
    new_spec: str

    # resume-fs
    working_dir: str


@dataclass
class InputData:
    """
    Represents all of the file inputs provided by the user after uploading
    """
    spec: UploadedFile
    system_doc: UploadedFile
    intf: UploadedFile


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
    new_spec: NativeFS
    comments: Optional[str]
    new_system: Optional[NativeFS]

@dataclass
class ResumeFSData:
    thread_id: str
    file_path: str
    comments: Optional[str]
    new_system: Optional[NativeFS]
