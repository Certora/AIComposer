from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

from composer.ui.ide_bridge import IDEBridge
from composer.assistant.launch_args import (
    LaunchCodegenArgs,
    LaunchNatSpecArgs,
    LaunchResumeArgs,
)


class OrchestratorState(MessagesState):
    result: NotRequired[str]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorModelConfig:
    """Typed configuration for LLM + service options.

    Satisfies both ``ModelOptions`` and ``RAGDBOptions`` protocols.
    """
    model: str
    tokens: int
    thinking_tokens: int
    memory_tool: bool
    rag_db: str


@dataclass
class OrchestratorContext:
    workspace: Path
    ide: IDEBridge | None
    llm: BaseChatModel
    config: OrchestratorModelConfig


# ---------------------------------------------------------------------------
# Interrupt payloads (discriminated union)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConversationTurn:
    """Conversation-mode interrupt: LLM emitted a message with no tool calls."""
    message: AIMessage


type ConfirmLaunch = LaunchCodegenArgs | LaunchNatSpecArgs | LaunchResumeArgs
type OrchestratorInterrupt = ConversationTurn | ConfirmLaunch
