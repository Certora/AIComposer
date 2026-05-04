"""Explicit thinking and rough draft tools for multi-step reasoning workflows.

Ported from composer/spec/cvl_generation.py and composer/spec/draft.py on
the jtoman/auto-prover branch.
"""

from typing import cast, overload, override
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.types import Command
from pydantic import Field
from composer.ui.tool_display import tool_display_of, CommonTools

from graphcore.tools.schemas import WithImplementation, WithInjectedId, WithInjectedState

class RoughDraftState(TypedDict):
    memory: str | None
    did_read: bool


class _RoughDraftWithMessages(RoughDraftState, MessagesState):
    """Bound for the reminder-injecting overload — the guard reads the
    messages channel and the HumanMessage rides on the same channel."""


@overload
def get_rough_draft_tools[ST: RoughDraftState](
    ty: type[ST],
) -> list[BaseTool]: ...


@overload
def get_rough_draft_tools[ST: _RoughDraftWithMessages](
    ty: type[ST],
    *,
    review_reminder: str,
) -> list[BaseTool]: ...


def get_rough_draft_tools[ST](
    ty: type[ST],
    *,
    review_reminder: str | None = None,
) -> list[BaseTool]:
    """Build the (write_rough_draft, read_rough_draft) tool pair.

    ``review_reminder`` is an optional prompt fragment surfaced as a
    ``<system-reminder>`` HumanMessage immediately after the rough draft
    is delivered to the agent on a ``read_rough_draft`` call. Use it to
    re-state, at the moment of review, what specifically the agent should
    be checking — the failure modes the validator will reject, the
    shape requirements of the result, etc. This is the same head-of-recent-
    context lever the prover-violation reminder uses; appending the cue
    at read time defeats the long-context drift where the agent reviews
    its draft without remembering the original reviewing criteria.

    The reminder rides alongside the tool result as a separate HumanMessage,
    which means ``read_rough_draft`` MUST be the only tool call in its turn —
    otherwise the appended user-role content breaks the tool_use ↔
    tool_result pairing for any siblings. ``GetMemory.run`` enforces this
    via a ``state["messages"][-1].tool_calls`` check, mirroring the
    parallel-prover guard in ``CertoraProverTool``. The overload bound
    requires ``ST`` to also satisfy ``MessagesState`` whenever a reminder
    is provided so that channel access is statically valid; the impl
    casts unsafely on that promise.
    """
    @tool_display_of(CommonTools.read_rough_draft)
    class GetMemory(WithInjectedState[ST], WithImplementation[Command | str], WithInjectedId):
        """
        Retrieve the rough draft of the feedback
        """
        @override
        def run(self) -> str | Command:
            if review_reminder is not None:
                # Overloads guarantee ST extends MessagesState here.
                msg_channel = cast(MessagesState, self.state)["messages"]
                last = msg_channel[-1]
                if isinstance(last, AIMessage):
                    tcs = last.tool_calls
                    if any(tc["id"] == self.tool_call_id for tc in tcs) and len(tcs) > 1:
                        return (
                            "Error: read_rough_draft must be the only tool call "
                            "in its turn. Re-issue this call alone."
                        )
            mem_state = cast(RoughDraftState, self.state)
            mem = mem_state["memory"]
            if mem is None:
                return "Rough draft not yet written"
            messages: list = [
                ToolMessage(tool_call_id=self.tool_call_id, content=mem),
            ]
            if review_reminder is not None:
                messages.append(HumanMessage(
                    content=f"<system-reminder>\n{review_reminder}\n</system-reminder>",
                    display_tag="rough_draft_review_reminder",
                ))
            return Command(update={
                "messages": messages,
                "did_read": True,
            })

    @tool_display_of(CommonTools.write_rough_draft)
    class SetMemory(WithInjectedId, WithImplementation[Command]):
        """
        Write your rough draft for review
        """
        rough_draft: str = Field(description="The new rough draft of your feedback")

        @override
        def run(self) -> Command:
            return Command(update={
                "memory": self.rough_draft,
                "did_read": False,
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content="Success")]
            })

    return [SetMemory.as_tool("write_rough_draft"), GetMemory.as_tool("read_rough_draft")]
