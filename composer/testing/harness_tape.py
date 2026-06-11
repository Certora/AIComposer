from typing import Any, Callable, Sequence, override
import random
import asyncio

from pydantic import Field
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from composer.diagnostics.timing import get_current_task_id


def _prompt_preview(model_input: Any) -> str:
    """A short, safe description of the incoming prompt, to make a missing or
    mis-lane'd tape entry easy to locate when authoring."""
    try:
        if isinstance(model_input, PromptValue):
            msgs: list[Any] = list(model_input.to_messages())
        elif isinstance(model_input, (list, tuple)):
            msgs = list(model_input)
        else:
            return repr(model_input)[:160]
        if not msgs:
            return "<empty prompt>"
        last = msgs[-1]
        content = getattr(last, "content", last)
        return f"{type(last).__name__}: {str(content)[:160]}"
    except Exception:
        return "<unpreviewable prompt>"


class HarnessFakeLLM(FakeMessagesListChatModel):
    """``FakeMessagesListChatModel`` tolerant of the specific shape of attribute
    access the codegen workflow performs on the bound LLM, with per-lane tape
    routing.

    Two compatibility shims:

    * ``thinking`` — ``composer.workflow.meta.create_resume_commentary``
      calls ``llm.copy(update={"thinking": None})``. Pydantic v2 tolerates
      unknown keys but prints less predictably; declaring the field makes
      the copy a no-op explicitly.
    * ``betas`` — ``composer.workflow.executor`` does
      ``getattr(llm, "betas")``. An empty list keeps the memory-tool
      beta branch off, so the main codegen agent's tool list matches
      what the tape expects.

    Lane routing: each call is served from the per-lane cursor for the active
    ``run_task`` ``task_id`` (``composer.diagnostics.timing.get_current_task_id``).
    The task_id is read in the async ``ainvoke`` body, where the ContextVar that
    ``run_task`` set is visible (reading it inside the synchronous ``_generate``,
    which the base runs in an executor thread, would not see it). This keeps the
    tape deterministic even though the pipeline runs phases concurrently.
    """

    thinking: Any = None
    betas: list[str] = []
    # The base requires `responses`, but lane routing serves from `lanes` and
    # never reads it; default it so callers construct with `lanes=` alone.
    responses: list[BaseMessage] = Field(default_factory=list)
    # task_id -> ordered scripted responses for that lane.
    lanes: dict[str, list[BaseMessage]] = Field(default_factory=dict)
    # task_id -> next index. Mutated in place; each instance owns its own dict.
    lane_cursors: dict[str, int] = Field(default_factory=dict, exclude=True)

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
    ) -> BaseMessage:
        # Simulate LLM latency to keep the TUI from filling all at once to give some ability to judge the "feel" of the UI.
        await asyncio.sleep(random.random() * 1.5 + 1.0)

        task_id = get_current_task_id()
        if task_id is None:
            raise RuntimeError(
                "HarnessFakeLLM: LLM call outside any run_task scope, so it "
                "cannot be routed to a tape lane. "
                f"Prompt -> {_prompt_preview(input)}"
            )
        lane = self.lanes.get(task_id)
        if lane is None:
            raise RuntimeError(
                f"HarnessFakeLLM: no tape lane for task_id {task_id!r}. "
                f"Known lanes: {sorted(self.lanes)}. "
                f"Prompt -> {_prompt_preview(input)}"
            )
        i = self.lane_cursors.get(task_id, 0)
        if i >= len(lane):
            raise RuntimeError(
                f"HarnessFakeLLM: tape lane {task_id!r} exhausted after "
                f"{len(lane)} response(s) — the pipeline issued an extra call in "
                f"this phase. Prompt -> {_prompt_preview(input)}"
            )
        self.lane_cursors[task_id] = i + 1
        return lane[i]
