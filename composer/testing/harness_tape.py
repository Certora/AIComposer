from typing import Any, Callable, Sequence, override
import random
import asyncio

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

class HarnessFakeLLM(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` tolerant of the specific shape of attribute
    access the codegen workflow performs on the bound LLM.

    Two compatibility shims:

    * ``thinking`` — ``composer.workflow.meta.create_resume_commentary``
      calls ``llm.copy(update={"thinking": None})``. Pydantic v2 tolerates
      unknown keys but prints less predictably; declaring the field makes
      the copy a no-op explicitly.
    * ``betas`` — ``composer.workflow.executor`` does
      ``getattr(llm, "betas")``. An empty list keeps the memory-tool
      beta branch off, so the main codegen agent's tool list matches
      what the tape expects.
    """

    thinking: Any = None
    betas: list[str] = []

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        return self
    
    async def ainvoke(
        self,
        input: PromptValue | str | Sequence[BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]],
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any
    ) -> AIMessage:
        delay = random.random() * 3.0 + 1.0
        await asyncio.sleep(delay)
        return await super().ainvoke(input, config, stop=stop, **kwargs)
