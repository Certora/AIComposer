from typing import Annotated
from graphcore.graph import WithToolCallId
from pydantic import Field

from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.runtime import get_runtime

from graphcore.graph import WithToolCallId, tool_return

from composer.human.types import ProposalType
from composer.core.state import AIComposerState
from composer.core.context import AIComposerContext
from composer.ui.tool_display import tool_display



class SpecChangeProposalArgs(WithToolCallId):
    """
    Propose a change to one of the specification files. There are two
    legitimate use cases for calling this tool.

    You may use this tool to propose the addition of *summaries*, by updating
    (or adding) the `methods` block, and adding any ancilliary definitions.

    Summaries are used to (effectively) replace the body of a function with
    the given "summary". This summary may be a declaration that the return
    value of the function should be treated as nondeterministic, a havoc, or a
    CVL expression.

    Summaries are typically used when a function is too difficult to reason
    about directly; this complexity may come from non-linear operations,
    complex inter-contract interactions, or bitwise operations.

    When possible, the summary should be *sound*, that is, the behavior that
    replaces the function should *over-approximate* the function
    implementation. For example of an *unsound* summary, summarizing a
    function called `mulDiv(uint,uint)` to simply return 0 is unsound. A
    *sound* summary for `mulDiv(uint,uint)` would be to return a
    non-deterministic number, as this admits "more" behaviors than the exact
    `mulDiv` implementation.

    However, there may be cases where a summary *can* be used to elide
    behavior that is not relevant to the properties being proven. For example,
    if the property being verified relates to interest fee calculation, and a
    function simply emits logs, it may be appropriate to "summarize away" the
    log emission function.

    The other use case is when the user has approved any other change
    discussed via the human_in_the_loop tool. For example, if the human agrees
    that the spec is ambiguous, or needs changing, use this tool to propose
    the minimal necessary change to the spec file.

    A human will review this request, and either respond with `ACCEPTED`,
    `REJECTED: ...`, or `REFINE: ...`. `ACCEPTED` means the proposed spec file
    will be used for all future Certora Prover runs. `REJECTED` means the
    proposed change should be discarded, with an explanation as to why the
    change is not appropriate. You should incorporate this explanation when
    considering proposed changes. `REFINE` indicates that the change proposal
    should be adjusted according to the given feedback and then re-proposed.
    """

    target_path: str = Field(description=(
        "The VFS path of the spec file being changed. Must be one of the "
        "registered spec files (use ``list_files`` to see what's available). "
        "The proposal replaces the entire contents at this path on acceptance."
    ))

    proposed_spec: str = Field(description=(
        "The new version of the spec file at ``target_path`` to use going "
        "forward. The proposed spec file *MUST* be syntactically valid and "
        "complete. Do *NOT* provide just the changes — provide the *entire* "
        "file *after* your proposed changes would be applied."
    ))

    explanation: str = Field(description=(
        "An explanation to the human reviewer as to why you think this change "
        "is necessary and why it is safe or sound to apply it."
    ))

    state: Annotated[AIComposerState, InjectedState]


@tool_display(
    lambda p: (
        f"Proposing spec change to {p.get('target_path', '?')}: {p['explanation']}"
        if p.get("explanation") else f"Proposing spec change to {p.get('target_path', '?')}"
    ),
    None,
)
@tool(args_schema=SpecChangeProposalArgs)
def propose_spec_change(
    target_path: str,
    proposed_spec: str,
    explanation: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[AIComposerState, InjectedState]
) -> Command:
    ctxt = get_runtime(AIComposerContext)
    vfs_access = ctxt.context.vfs_materializer
    curr_spec = vfs_access.get(state, target_path)
    if curr_spec is None:
        return tool_return(
            tool_call_id=tool_call_id,
            content=(
                f"Target path {target_path!r} is not a registered spec file "
                f"in the VFS. Use list_files to see available paths."
            ),
        )
    human_response = interrupt(ProposalType(
        type="proposal",
        proposed_spec=proposed_spec,
        current_spec=curr_spec.decode("utf-8"),
        explanation=explanation
    ))
    assert isinstance(human_response, str)
    if human_response.startswith("ACCEPTED"):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=human_response
                    )
                ],
                "vfs": {
                    target_path: proposed_spec
                }
            }
        )
    return tool_return(
        tool_call_id=tool_call_id,
        content=human_response
    )