"""Explicit thinking and rough draft tools for multi-step reasoning workflows.

Ported from composer/spec/cvl_generation.py and composer/spec/draft.py on
the jtoman/auto-prover branch.
"""

from typing import override
from typing_extensions import TypedDict

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import Field
from composer.ui.tool_display import tool_display_of, CommonTools

from graphcore.tools.schemas import WithImplementation, WithInjectedId, WithInjectedState

class RoughDraftState(TypedDict):
    memory: str | None
    did_read: bool


def get_rough_draft_tools[ST: RoughDraftState](
    ty: type[ST],
) -> list[BaseTool]:
    @tool_display_of(CommonTools.read_rough_draft)
    class GetMemory(WithInjectedState[ST], WithImplementation[Command | str], WithInjectedId):
        """
        Retrieve the rough draft of the feedback
        """
        @override
        def run(self) -> str | Command:
            mem = self.state["memory"]
            if mem is None:
                return "Rough draft not yet written"
            return Command(update={
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content=mem)],
                "did_read": True
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
