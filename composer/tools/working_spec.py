from typing import override

from langgraph.types import Command, interrupt

from pydantic import Field

from graphcore.graph import tool_state_update
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId

from composer.core.state import AIComposerState
from composer.human.types import ProposalType
from composer.cvl.tools import maybe_update_cvl

class ReadWorkingSpec(WithImplementation[str], WithInjectedState[AIComposerState], WithInjectedId):
    """
    Read the contents of your working spec.
    """
    @override
    def run(self) -> str:
        if not self.state["working_spec"]:
            return "No working spec written"
        return self.state["working_spec"]
    
class WriteWorkingSpec(WithImplementation[Command | str], WithInjectedId):
    """
    Write a new version of your working spec. If the new version is not syntactically correct this tool call will be rejected
    with an error message.
    """
    new_cvl: str = Field(description="The new working spec. Should be a complete, self-contained, syntatically correct CVL file (do NOT submit a 'patch')")

    @override
    def run(self) -> str | Command:
        return maybe_update_cvl(
            tool_call_id=self.tool_call_id,
            pp=self.new_cvl,
            spec_key="working_spec"
        )
    
class CommitWorkingSpec(WithImplementation[Command | str], WithInjectedId, WithInjectedState[AIComposerState]):
    """
    Call this tool to ask a human reviewer to approve "committing" your working spec to the "master" copy.

    You should only use this tool after you have run the prover with sufficient rigor to confirm that the changes present
    in the spec are correct and pass formal verification. In addition, the changes present here should be the minimal possible
    changes to ensure the formal verification passes. Do *NOT* rewrite entire portions of the specification or make large scale changes
    unless the user has explicitly approved these changes via the human_in_the_loop tool. Do *NOT* request changes that significantly
    weaken the specification or otherwise trivialize it.

    NB once the working spec has been committed to the VFS, it is discarded.
    """
    explanation: str = \
    Field(description="An explanation to the human reviewer as to why you think the changes in the working spec"
            "this change is necessary and why it is safe or sound to apply it.")

    @override
    def run(self) -> Command | str:
        if not self.state["working_spec"]:
            return "No working spec set."
        work_spec = self.state["working_spec"]
        proposal : ProposalType = {
            "type": "proposal",
            "current_spec": self.state["vfs"]["rules.spec"],
            "proposed_spec": work_spec,
            "explanation": self.explanation
        }

        response = interrupt(proposal)
        assert isinstance(response, str)
        if response.startswith("ACCEPTED"):
            return tool_state_update(
                tool_call_id=self.tool_call_id,
                content="Accepted",
                working_spec=None,
                vfs={"rules.spec", work_spec}
            )
        return response
