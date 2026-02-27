"""Explicit thinking and rough draft tools for multi-step reasoning workflows.

Ported from composer/spec/cvl_generation.py and composer/spec/draft.py on
the jtoman/auto-prover branch.
"""

from typing import TypedDict, NotRequired, override

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import Field

from graphcore.tools.schemas import WithImplementation, WithInjectedId, WithInjectedState


class ExplicitThinking(
    WithImplementation[Command],
    WithInjectedId
):
    """Use this tool to record your reasoning. It will not execute any actions
    or retrieve any information — it only logs your thought for future reference.

    Use it when you need to:
    - Synthesize findings after gathering source files or documentation
    - Plan an implementation approach before writing or modifying code
    - Analyze a prover violation before deciding on a fix
    - Evaluate tradeoffs between multiple strategies
    - Verify that your planned changes satisfy all requirements and constraints

    Do NOT use it when:
    - The next step is obvious (e.g., fetching a file, running a test)
    - You are simply executing a known plan step by step
    - You have not yet gathered the information needed to reason usefully

    IMPORTANT: you may not call this tool in parallel with other tools.
    """
    thought: str = Field(
        description=(
            "Your structured reasoning. Include: "
            "what you have learned so far, "
            "what constraints or requirements apply, "
            "what approach you are considering and why, "
            "and any risks or edge cases to watch for."
        )
    )

    @override
    def run(self) -> Command:
        return Command(update={"messages": [
            ToolMessage(tool_call_id=self.tool_call_id, content="Thought recorded."),
            HumanMessage(
                content="Now, consider your current thought process and carefully evaluate how to proceed.",
                display_tag="thinking_nudge"
            ),
        ]})


explicit_thinking = ExplicitThinking.as_tool("extended_reasoning")


class RoughDraftProtocol(TypedDict):
    memory: NotRequired[str]
    did_read: NotRequired[bool]


def get_rough_draft_tools[ST: RoughDraftProtocol](
    ty: type[ST],
) -> list[BaseTool]:
    class GetMemory(WithInjectedState[ST], WithImplementation[Command | str], WithInjectedId):
        """
        Retrieve the rough draft of the feedback
        """
        @override
        def run(self) -> str | Command:
            mem = self.state.get("memory", None)
            if mem is None:
                return "Rough draft not yet written"
            return Command(update={
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content=mem)],
                "did_read": True
            })

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
