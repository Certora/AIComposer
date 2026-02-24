from typing import Protocol, Literal, TypedDict
from dataclasses import dataclass

Ecosystem = Literal["evm", "soroban", "move", "solana"]

class TokenUsageDict(TypedDict, total=False):
    """Dictionary for accumulating token usage across LLM calls."""
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int

class AnalysisArgs(Protocol):
    @property
    def folder(self) -> str:
        ...

    @property
    def rule(self) -> str:
        ...

    @property
    def method(self) -> str | None:
        ...

    @property
    def quiet(self) -> bool:
        ...

    @property
    def recursion_limit(self) -> int:
        ...
    
    @property
    def thread_id(self) -> str | None:
        ...

    @property
    def checkpoint_id(self) -> str | None:
        ...

    @property
    def thinking_tokens(self) -> int:
        ...
    
    @property
    def tokens(self) -> int:
        ...
    
    @property
    def ecosystem(self) -> Ecosystem:
        ...

    @property
    def rag_db(self) -> str:
        ...

@dataclass(frozen=True)
class AnalysisArgsD():
    folder: str
    rule: str
    quiet: bool
    recursion_limit: int
    thinking_tokens: int
    tokens: int
    ecosystem: Ecosystem
    rag_db: str
    method: str | None = None
    thread_id: str | None = None
    checkpoint_id: str | None = None

def __typechecker_stub(s: AnalysisArgs):
    pass

def __typechecker_sanity(s: AnalysisArgsD):
    __typechecker_stub(s)
