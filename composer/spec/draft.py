from typing import TypedDict, NotRequired, override
from graphcore.tools.schemas import WithInjectedId, WithInjectedState, WithImplementation

from langgraph.types import Command

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, BaseTool

from pydantic import Field

class Protocol(TypedDict):
    memory: NotRequired[str]
    did_read: NotRequired[bool]

def get_rough_draft_tools[ST: Protocol](
    ty: type[ST]
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
        rough_draft : str = Field(description="The new rough draft of your feedback")

        @override
        def run(self) -> Command:
            return Command(update={
                "memory": self.rough_draft,
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content="Success")]
            })
    
    return [SetMemory.as_tool("write_rough_draft"), GetMemory.as_tool("read_rough_draft")]

