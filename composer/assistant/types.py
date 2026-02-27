from dataclasses import dataclass, field
from pathlib import Path
from typing import NotRequired, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState

from composer.io.ide_bridge import IDEBridge


class OrchestratorState(MessagesState):
    result: NotRequired[str]


@dataclass
class OrchestratorContext:
    workspace: Path
    ide: IDEBridge | None
    llm: BaseChatModel


@dataclass
class CodegenWorkflowArgs:
    """Satisfies WorkflowOptions protocol for programmatic invocation."""
    prover_capture_output: bool = True
    prover_keep_folders: bool = False
    debug_prompt_override: Optional[str] = None
    recursion_limit: int = 50
    audit_db: str = ""
    summarization_threshold: Optional[int] = None
    requirements_oracle: list[str] = field(default_factory=list)
    set_reqs: Optional[str] = None
    skip_reqs: bool = False
    rag_db: str = ""
    checkpoint_id: Optional[str] = None
    thread_id: Optional[str] = None
    model: str = "claude-opus-4-6"
    tokens: int = 10_000
    thinking_tokens: int = 2048
    memory_tool: bool = True


@dataclass
class NatSpecWorkflowArgs:
    """Satisfies NatSpecArgs protocol for programmatic invocation."""
    input_file: str = ""
    model: str = "claude-opus-4-6"
    tokens: int = 10_000
    thinking_tokens: int = 2048
    memory_tool: bool = True
    rag_db: str = ""
    checkpoint_id: Optional[str] = None
    thread_id: Optional[str] = None
    recursion_limit: int = 50
