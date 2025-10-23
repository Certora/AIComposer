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
    def bytes_contents(self) -> bytes:
        with open(self.path, 'rb') as f:
            return f.read()


class WorkflowOptions(Protocol):
    # database options
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str

    prover_capture_output: bool
    prover_keep_folders: bool

    debug_prompt_override: Optional[str]

    checkpoint_id: Optional[str]
    thread_id: Optional[str]
    recursion_limit: int
    audit_db: Optional[str]
    summarization_threshold: Optional[int]


class ModelOptions(Protocol):
    model: str
    tokens: int
    thinking_tokens: int

class CommandLineArgs(WorkflowOptions, ModelOptions):
    spec_file: str
    interface_file: str
    system_doc: str
    debug_fs: str

    debug: bool

@dataclass
class InputData:
    """
    Represents all of the file inputs provided by the user after uploading
    """
    spec: UploadedFile
    system_doc: UploadedFile
    intf: UploadedFile
