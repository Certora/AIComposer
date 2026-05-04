from typing import override

from langgraph.runtime import get_runtime
from langgraph.types import Command, interrupt

from pydantic import Field

from graphcore.graph import tool_state_update
from graphcore.tools.schemas import (
    WithAsyncDependencies, WithImplementation, WithInjectedState, WithInjectedId,
)

from composer.core.context import AIComposerContext
from composer.core.state import AIComposerState
from composer.human.types import ProposalType
from composer.cvl.tools import maybe_update_cvl
from composer.spec.proposal_store import ProposalStore

from composer.ui.tool_display import tool_display

@tool_display("Read working spec", None)
class ReadWorkingSpec(WithImplementation[str], WithInjectedState[AIComposerState], WithInjectedId):
    """
    Read the contents of your working spec draft. The working spec is a single
    transient draft — it has no VFS path until you commit it.
    """
    @override
    def run(self) -> str:
        ws = self.state["working_spec"]
        if not ws:
            return "No working spec written"
        return ws

@tool_display("Write working spec", None)
class WriteWorkingSpec(WithImplementation[Command | str], WithInjectedId):
    """
    Write a new version of your working spec draft from raw CVL text.
    Only one working spec can exist at a time; calling this tool replaces
    any previous draft. The draft is transient and has no VFS path —
    that's assigned at commit time (with human review).

    For applying a `cex_remediation` proposal, prefer
    `apply_remediation_proposal(proposal_key)` — it fetches the full
    proposed text by key so you don't re-emit it. Reach for this tool
    when you need to tweak a remediator proposal (e.g. fix a typecheck
    error in the proposed CVL) or for any other case where you have raw
    CVL text in hand.

    If the new version is not syntactically correct, this tool call will
    be rejected with an error message.
    """
    new_cvl: str = Field(description="The new working spec. Should be a complete, self-contained, syntactically correct CVL file (do NOT submit a 'patch')")

    @override
    def run(self) -> str | Command:
        return maybe_update_cvl(
            tool_call_id=self.tool_call_id,
            pp=self.new_cvl,
            spec_key="working_spec",
        )


@tool_display("Apply remediation proposal", None)
class ApplyRemediationProposal(WithAsyncDependencies[Command | str, ProposalStore], WithInjectedId):
    """
    Stage a `cex_remediation` proposal as your working spec draft. Pass
    the `proposal_key` that the remediator returned in its "Apply via"
    line; this tool looks up the full proposed CVL text and stores it as
    the transient working draft, replacing any previous draft.

    Use this for the common case of "remediator returned a proposal, I
    accept it as-is." For the case where you need to tweak the proposal
    (typecheck fix, small adjustment), reach for `write_working_spec`
    with the modified text.

    The draft has no VFS path until you commit it via
    `commit_working_spec` (which goes through human review). If the
    proposed CVL fails the typecheck, this tool call will be rejected
    with an error message.
    """
    proposal_key: str = Field(
        description=(
            "The opaque proposal_key printed in `cex_remediation`'s output "
            "under 'Apply via'. The full proposed spec text is stored under "
            "this key; this tool fetches and stages it."
        ),
    )

    @override
    async def run(self) -> str | Command:
        with self.tool_deps() as proposal_store:
            full_cvl = await proposal_store.lookup(self.proposal_key)
            if full_cvl is None:
                return (
                    f"No proposal found for proposal_key {self.proposal_key!r}. "
                    f"Re-invoke `cex_remediation` to get a fresh key; old keys "
                    f"do not survive across remediation calls."
                )
            return maybe_update_cvl(
                tool_call_id=self.tool_call_id,
                pp=full_cvl,
                spec_key="working_spec",
            )

@tool_display(
    lambda p: (
        f"Committing working spec to {p['target_path']}"
        if p.get("target_path") else "Committing working spec"
    ),
    None,
)
class CommitWorkingSpec(WithImplementation[Command | str], WithInjectedId, WithInjectedState[AIComposerState]):
    """
    Call this tool to ask a human reviewer to approve "committing" your working
    spec to a specific spec file in the VFS. Conceptually: ``mv
    <working_spec> <target_path>`` — the transient draft becomes the committed
    content at ``target_path``.

    You should only use this tool after you have run the prover with sufficient
    rigor (via ``use_working_spec=True``) to confirm that the changes present
    in the spec are correct and pass formal verification. In addition, the
    changes present here should be the minimal possible changes to ensure the
    formal verification passes. Do *NOT* rewrite entire portions of the
    specification or make large scale changes unless the user has explicitly
    approved these changes via the human_in_the_loop tool. Do *NOT* request
    changes that significantly weaken the specification or otherwise trivialize
    it.

    NB once the working spec has been committed to the VFS, it is discarded.
    Committing does NOT by itself produce a prover verification stamp — after
    a successful commit you still need to run ``certora_prover`` against the
    committed spec (with ``use_working_spec=False`` and the same
    ``target_spec=target_path``) to record the stamp.
    """
    target_path: str = Field(description=(
        "The VFS path of the spec file this working draft should become. "
        "Must be one of the registered spec files for this task (use "
        "``list_files`` to see which spec paths exist). The working draft is "
        "written to this path on acceptance."
    ))
    explanation: str = Field(description=(
        "An explanation to the human reviewer as to why you think this change "
        "is necessary and why it is safe or sound to apply it."
    ))

    @override
    def run(self) -> Command | str:
        work_spec = self.state["working_spec"]
        if not work_spec:
            return "No working spec set."
        current = self.state.get("vfs", {}).get(self.target_path)
        if current is None:
            return (
                f"Target path {self.target_path!r} is not a registered spec file "
                f"in the VFS. Use list_files to see available paths."
            )
        proposal : ProposalType = {
            "type": "proposal",
            "current_spec": current,
            "proposed_spec": work_spec,
            "explanation": self.explanation,
        }

        response = interrupt(proposal)
        assert isinstance(response, str)
        if response.startswith("ACCEPTED"):
            return tool_state_update(
                tool_call_id=self.tool_call_id,
                content="Accepted",
                working_spec=None,
                vfs={self.target_path: work_spec},
            )
        return response
