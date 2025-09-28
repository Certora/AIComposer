from graphcore.graph import WithToolCallId, tool_output
from pydantic import Field
from typing import Dict, Annotated, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

class PutFileArgs(WithToolCallId):
    """
    Put file contents onto the virtual file system used by the Certora prover.

    IMPORTANT: You may not use this tool to update the specification, nor should you attempt to
    add new specification files.
    """

    files: Dict[str, str] = \
        Field(default_factory=dict, description="A dictionary associating RELATIVE pathnames to the contents to store at those path names. The provided contents "
              "are durably stored into the virtual filesystem. Any files contents with the same path named stored in previous tool calls are overwritten. "
              "By convention, every Solidity placed into the virtual filesystem should contain exactly one contract/interface/library defitions. "
              "Further, the name of the contract/interface/library defined in that file should name the name of the solidity source file sans extension. "
              "For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`")


@tool(args_schema=PutFileArgs)
def put_file(
    tool_call_id: Annotated[str, InjectedToolCallId],
    files: Optional[Dict[str, str]] = None
) -> Command:
    if not files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id = tool_call_id,
                        content=(
                            "The put_file tool requires a 'files' mapping of relative path -> contents. "
                            "Example: {\n  \"src/MyContract.sol\": \"<solidity source>\"\n}. "
                            "You may not mutate 'rules.spec' or 'test.t.sol' via this tool."
                        )
                    )
                ]
            }
        )
    if "rules.spec" in files or "test.t.sol" in files:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        tool_call_id = tool_call_id,
                        content="You may not mutate the rules.spec or test.t.sol files via this tool; " \
                                "all mutations must go through the propose_spec_change tool"
                    )
                ]
            }
        )
    return tool_output(
        tool_call_id=tool_call_id,
        res={"virtual_fs": files}
    )