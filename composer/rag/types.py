from typing import List, TypedDict
from dataclasses import dataclass

@dataclass
class ManualRef:
    headers: List[str]
    content: str
    similarity: float

@dataclass
class BlockChunk:
    headers: List[str]
    part: int
    code_refs: List[str]
    chunk: str
