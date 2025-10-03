from typing import Annotated, List
from graphcore.graph import WithToolCallId, tool_output
from verisafe.core.state import CryptoStateGen, ResultStateSchema
from pydantic import Field
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command

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
    return tool_output(
        tool_call_id=tool_call_id,
        res={
            "generated_code": ResultStateSchema(
                comments=comments,
                source=source
            )
        }
    )
