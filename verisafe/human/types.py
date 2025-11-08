from typing import Literal, Optional, TypedDict, Annotated, Union
from pydantic import Discriminator

class ProposalType(TypedDict):
    """
    Type for a proposed rule spec change to present to the user
    """
    type: Literal["proposal"]
    proposed_spec: str
    current_spec: str
    explanation: str


class QuestionType(TypedDict):
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

    type: Literal["question"]
    context: Annotated[str, "Any context to give for the question, including what you have tried, "
              "what didn't work, and in case of questions about the spec, any quotes from the relevant documentation."]
    question: Annotated[str, "The exact question or request for assistance to pose to the human"]
    code: Annotated[Optional[str], "Any code snippet(s) that might be relevant to the question. " \
              "IMPORTANT: do NOT escape newlines or other special characters; this string will be directly printed to a terminal."]

class RequirementRelaxationType(TypedDict):
    """
Ask the user to relax one of the requirements and remove its validation from the task completion criteria.

You should ONLY use this tool if you are certain that the failure to meet a requirement is in ERROR.
The requirements are part of the normative specification for the implementation. You should not use this
tool to attempt to circumvent them.

The user response will begin with either ACCEPTED or REJECTED. If 'ACCEPTED', the requirement may
be ignored in future validations. If REJECTED, the user considers both the requirement valid and the judgment
of the requirements oracle correct. In this case, reflect on the user's answer to understand what might
need to change in your implementation.

IMPORTANT: After invoking this tool, you MUST invoke the requirements oracle again to reflect this
change to the requirements.
    """

    type: Literal["req_relaxation"]
    context: Annotated[str, "Context for the disputed requirement, including any relevant experience from invoking the prover, prior" \
    " interactions with the user, or references to the system document."]
    req_number: Annotated[int, "The requirement number, taken from the original list"]
    req_text: Annotated[str, "The requirement text (without the numeric label)"]
    judgment: Annotated[str, "The text from the requirement oracle explaining its decision."]
    explanation: Annotated[str, "An explanation for why the requirement should be dropped/ignored. You may either argue that the " \
    "oracle mischaracterized the implementation, or that the original requirement is itself incorrect."]


# Type of the interrupt payload
HumanInteractionType = Annotated[Union[ProposalType, QuestionType, RequirementRelaxationType], Discriminator("type")]

