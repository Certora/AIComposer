from typing import Optional, Annotated
from graphcore.graph import WithToolCallId
from pydantic import Field

from graphcore.graph import WithToolCallId, tool_return
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command, interrupt
from verisafe.human.types import QuestionType


class HumanInTheLoopArg(WithToolCallId):
    """
    Ask the user for help to debug some specific problem. Within the toolbox provided, this should be considered
    a last resort, but still a valid way to get "unstuck".

    A non-exhaustive list of the types of issues for which you should ask assistance include the following:
    1. Persistent timeouts despite attempts at rewriting and/or summarization
    2. Help in proposing a summary for an internal function
    3. Disambiguating or clarifying the informative specification
    4. Resolving apparent conflicts between the normative and informative specifications
    5. Unexplained Certora Prover errors that you are certain are not due to malformed inputs
    6. Undocumented CVL behavior, features, or syntax
    7. To request modifying the original specification if this is considered necessary.

    To emphasize, the above are just guidelines, and this is a free form method to request assistance.

    If the human response begins with FOLLOWUP, interpret their response as a request for clarification.
    Formulate an answer to the request, and reinvoke this tool, adjusting the question as appropriate.
    """
    question: str = Field(description="The exact question or request for assistance to pose to the human")
    context: str = \
        Field(description="Any context to give for the question, including what you have tried, "
              "what didn't work, and in case of questions about the spec, any quotes from the relevant documentation.")
    code: Optional[str] = \
        Field(description="Any code snippet(s) that might be relevant to the question. " \
              "IMPORTANT: do NOT escape newlines or other special characters; this string will be directly printed to a terminal.")

@tool(args_schema=HumanInTheLoopArg)
def human_in_the_loop(
    question: str,
    context: str,
    code: Optional[str],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    response = interrupt(QuestionType(
        type="question",
        context=context,
        code=code,
        question=question
    ))
    return tool_return(tool_call_id=tool_call_id, content=f"Human response: {response}")
