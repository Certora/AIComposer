from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from langgraph.runtime import get_runtime

from graphcore.tools.results import result_tool_generator

from verisafe.core.context import CryptoContext, compute_state_digest
from verisafe.core.state import CryptoStateGen, ResultStateSchema

def check_completion(
    state: CryptoStateGen,
    sch: ResultStateSchema,
    tool_call_id: str
) -> Command | None:
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
    return None

code_result = result_tool_generator("generated_code", ResultStateSchema,
"""
Used to communicate when the generated code is complete and satisfies all of the rules in specification.
""",
    (CryptoStateGen, check_completion)
)
