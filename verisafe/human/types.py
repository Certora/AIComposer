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
    Type for a question posed to the user
    """
    type: Literal["question"]
    context: str
    question: str
    code: Optional[str]

class RequirementRelaxationType(TypedDict):
    type: Literal["req_relaxation"]
    context: str
    req_number: int
    req_text: str
    judgment: str
    explanation: str


# Type of the interrupt payload
HumanInteractionType = Annotated[Union[ProposalType, QuestionType, RequirementRelaxationType], Discriminator("type")]

