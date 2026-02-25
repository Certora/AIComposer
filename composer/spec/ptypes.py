from typing import Annotated, NotRequired
from pydantic import BaseModel, Field
from composer.core.state import merge_validation

from graphcore.graph import MessagesState, FlowInput

class NatSpecInput(FlowInput):
    curr_intf: None
    curr_spec: None
    validations: dict[str, str]

class Result(BaseModel):
    expected_solc: str = Field(description="The expected solc version. Should match that used for interface updates and " \
    "spec type checking.")
    expected_contract_name: str = Field(description="The name of the contract which is expected to implement this interface")
    implementation_notes: str = Field(description="Any other notes that are relevant to an implementer of this spec.")

class NatSpecState(MessagesState):
    validations: Annotated[dict[str, str], merge_validation]
    curr_intf: str | None
    curr_spec: str | None
    result: NotRequired[Result]
