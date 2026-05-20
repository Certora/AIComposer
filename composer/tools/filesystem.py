import re

from graphcore.graph import WithToolCallId, tool_output
from pydantic import Field
from typing import Dict, Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage


# Any path ending in ``.spec`` — the agent is never allowed to rewrite a spec
# file via the generic put_file tool. Spec mutations must go through
# ``propose_spec_change`` (needs human approval) or ``write_working_spec`` +
# ``commit_working_spec`` (drafts then requests approval).
_SPEC_FILE_RE = re.compile(r"^.+\.spec$")


class PutFileArgs(WithToolCallId):
    """
    Put file contents onto the virtual file system used by the Certora prover.

    IMPORTANT: You may not use this tool to add, rewrite, or create specification
    (``*.spec``) files. All spec mutations must go through ``propose_spec_change``
    (for committed edits) or ``write_working_spec`` + ``commit_working_spec``
    (for iterative drafts).
    """

    files: Dict[str, str] = \
        Field(description="A dictionary associating RELATIVE pathnames to the contents to store at those path names. The provided contents "
              "are durably stored into the virtual filesystem. Any files contents with the same path named stored in previous tool calls are overwritten. "
              "By convention, every Solidity placed into the virtual filesystem should contain exactly one contract/interface/library definitions. "
              "Further, the name of the contract/interface/library defined in that file should match the name of the solidity source file sans extension. "
              "For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`")


@tool(args_schema=PutFileArgs)
def put_file(
    files: Dict[str, str],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    blocked = [p for p in files if _SPEC_FILE_RE.match(p)]
    if blocked:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id=tool_call_id,
                        content=(
                            f"You may not mutate or create spec files via this tool. "
                            f"Blocked paths: {blocked}. All spec mutations must go "
                            f"through the propose_spec_change tool (for committed edits) "
                            f"or the write_working_spec / commit_working_spec flow (for "
                            f"iterative drafts)."
                        ),
                    )
                ]
            }
        )
    return tool_output(
        tool_call_id=tool_call_id,
        res={"vfs": files}
    )