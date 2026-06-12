"""Shared ``give_up`` escape-hatch tool for agent loops.

Calling ``give_up`` sets the graph's ``result`` output key to the supplied
reason and marks ``failed=True`` in state, terminating the loop *without* going
through the result validator. Agents use it to bail out of unrecoverable or
precondition failures (e.g. the project does not compile) instead of retrying
indefinitely.

The state the agent runs in must declare ``result`` (the output key) and a
``failed: bool | None`` field; callers inspect ``failed`` after the loop to
distinguish a give-up from a normal completion.
"""

from typing import override

from pydantic import Field
from langgraph.types import Command

from graphcore.graph import tool_state_update
from graphcore.tools.schemas import WithImplementation, WithInjectedId
from composer.ui.tool_display import tool_display


@tool_display(
    label=lambda p: f"Giving up: {p['reason']}",
    result=None,
)
class GiveUpTool(WithImplementation[Command], WithInjectedId):
    """
    Call this tool to give up on the current task.

    This should only ever be called as a LAST RESORT when you have exhausted all other
    mechanisms to complete your task, or when the task is blocked by a precondition /
    environment failure that you cannot fix (for example, the project does not compile).
    """
    reason: str = Field(description="The reason for giving up on your task")

    @override
    def run(self) -> Command:
        return tool_state_update(
            self.tool_call_id,
            "Accepted",
            failed=True,
            result=self.reason,
        )
