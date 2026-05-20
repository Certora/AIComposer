from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from langgraph.runtime import get_runtime

from graphcore.tools.results import result_tool_generator

from composer.core.context import AIComposerContext, compute_state_digest
from composer.core.state import AIComposerState, ResultStateSchema
from composer.core.validation import Validation


def _describe(v: Validation) -> str:
    """Render a ``Validation`` key as a short prose phrase for the LLM."""
    match v.domain:
        case "prover":
            return f"prover verification of `{v.data}`"
        case "reqs":
            return "natural-language requirements judging"


def check_completion(
    state: AIComposerState,
    sch: ResultStateSchema,
    tool_call_id: str
) -> Command | None:
    ctxt = get_runtime(AIComposerContext).context
    digest = compute_state_digest(state=state)
    m = state.get("validation", {})
    for req_v in ctxt.required_validations:
        k = str(req_v)
        if k not in m or digest != m[k]:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            tool_call_id=tool_call_id,
                            content=f"Result completion REJECTED; it appears you failed to satisfy the {_describe(req_v)} requirement"
                        ),
                        HumanMessage(
                            content="You have apparently become confused about the status of your task. Evaluate the current "
                            "state of your implementation, enumerate any unaddressed feedback, and create a TODO list to address "
                            "that feedback.",
                            display_tag="scolding"
                        )
                    ]
                }
            )
    return None

code_result = result_tool_generator("generated_code", ResultStateSchema,
"""
Used to communicate when the generated code is complete and satisfies all of the rules in specification.
""",
    (AIComposerState, check_completion)
)
