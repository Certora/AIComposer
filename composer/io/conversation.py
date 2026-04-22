from typing import Protocol, Callable, AsyncContextManager
from dataclasses import dataclass, field
from rich.console import RenderableType

from langchain_core.messages.tool import ToolCall

type ConversationContextProvider = Callable[[RenderableType], AsyncContextManager[ConversationClient]]


@dataclass
class ThinkingStart:
    pass


@dataclass
class ToolBatch:
    """A single AI turn's tool calls.

    One ``ToolBatch`` per LangGraph ``AIMessage`` that contained tool
    calls — grouping across batches is handled by the renderer's
    consecutive-group state and is reset on human turns.
    """
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ToolComplete:
    tid: str


@dataclass
class AIYapping:
    yap_content: str

@dataclass
class StateUpdate:
    state_display: RenderableType

type ProgressPayload = ToolComplete | ToolBatch | ThinkingStart | AIYapping | StateUpdate


class ConversationClient(Protocol):
    async def human_turn(
        self, ai_response: str | None
    ) -> str:
        ...

    def progress_update(
        self, progress: ProgressPayload
    ):
        ...
