from typing import TypedDict
from dataclasses import dataclass

@dataclass
class ManualRef:
    headers: list[str]
    content: str
    similarity: float

@dataclass
class ManualSectionHit:
    headers: list[str]
    relevance: float

@dataclass
class BlockChunk:
    headers: list[str]
    part: int
    code_refs: list[str]
    chunk: str
