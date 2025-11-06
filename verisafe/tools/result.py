from typing import Annotated, List
from pydantic import Field

from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from langgraph.runtime import get_runtime

from graphcore.graph import WithToolCallId, tool_output, tool_return

from verisafe.core.context import CryptoContext, compute_state_digest
from verisafe.core.state import CryptoStateGen, ResultStateSchema

class ResultStateArgSchema(WithToolCallId):
    """
    Used to communicate when the generated code is complete and satisfies all of the rules in specification,
    with any approved summaries.
    """

    source: List[str] = \
        Field(description="The relative filenames in the virtual FS to present to the user. IMPORTANT: "
              "the filenames here must have been populated by prior put_file tool calls")
    comments: str = \
        Field(description="Any comments or notes on the generated implementation, and a summary of your reasoning, along with any lessons "
              "learned from iterating with the prover.")

    state: Annotated[CryptoStateGen, InjectedState]

@tool(args_schema=ResultStateArgSchema)
def code_result(
    source: List[str],
    comments: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[CryptoStateGen, InjectedState]
) -> Command:
    ctxt = get_runtime(CryptoContext).context
    digest = compute_state_digest(c=ctxt, state=state)
    m = state.get("validation", {})
    for req_v in ctxt.required_validations:
        if req_v not in m or digest != m[req_v]:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=f"Result completion REJECTED; it appears you failed to satisfy the {req_v} requirement"
                        ),
                        HumanMessage(
                            content="You have apparently become confused about the status of your task. Evaluate the current "
                            "state of your implementation, enumerate any unaddressed feedback, and create a TODO list to address "
                            "that feedback."
                        )
                    ]
                }
            )
    return tool_output(
        tool_call_id=tool_call_id,
        res={
            "generated_code": ResultStateSchema(
                comments=comments,
                source=source
            )
        }
    )
