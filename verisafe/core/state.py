from typing import NotRequired
from pydantic import BaseModel
from langgraph.graph import MessagesState

from graphcore.tools.vfs import VFSState

class ResultStateSchema(BaseModel):
    source: list[str]
    comments: str

class CryptoStateGen(VFSState, MessagesState):
    generated_code: NotRequired[ResultStateSchema]

