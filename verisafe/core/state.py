from typing import NotRequired
from pydantic import BaseModel

from graphcore.tools.vfs import VFSState

class ResultStateSchema(BaseModel):
    source: dict[str, str]
    comments: str

class CryptoStateGen(VFSState):
    generated_code: NotRequired[ResultStateSchema]

