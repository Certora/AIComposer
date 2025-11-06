from typing import NotRequired
from pydantic import BaseModel
from langgraph.graph import MessagesState

from langgraph.graph import MessagesState

from graphcore.tools.vfs import VFSState

class ResultStateSchema(BaseModel):
    source: list[str]
    comments: str

def merge_validation(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    to_ret = left.copy()
    for (k, v) in right.items():
        to_ret[k] = v
    return to_ret

class CryptoStateGen(VFSState, MessagesState):
    generated_code: NotRequired[ResultStateSchema]
    validation: NotRequired[dict[str, str]]

