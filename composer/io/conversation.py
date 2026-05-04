from typing import Protocol, TypedDict
from langgraph.types import Checkpointer
from langchain_core.language_models.chat_models import BaseChatModel
from dataclasses import dataclass

@dataclass
class ThinkingStart:
    pass

@dataclass
class ToolStart:
    tool_name: str
    tid: str

@dataclass
class ToolComplete:
    tid: str

@dataclass
class AIYapping:
    yap_content: str

type ProgressPayload = ToolComplete | ToolStart | ThinkingStart | AIYapping

class ConversationClient(Protocol):
    async def human_turn(
        self, ai_response: str | None
    ) -> str:
        ...

    def progress_update(
        self, progress: ProgressPayload
    ):
        ...