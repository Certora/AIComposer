from typing import Protocol, Literal, TypedDict
from dataclasses import dataclass

from composer.input.types import ModelOptions, LangraphOptions, RAGDBOptions

Ecosystem = Literal["evm", "soroban", "move", "solana"]

class TokenUsageDict(TypedDict, total=False):
    """Dictionary for accumulating token usage across LLM calls."""
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int

class AnalysisArgs(ModelOptions, LangraphOptions, RAGDBOptions, Protocol):
    folder: str
    rule: str
    method: str | None
    quiet: bool
    ecosystem: Ecosystem

@dataclass
class AnalysisArgsD():
    folder: str
    rule: str
    quiet: bool
    recursion_limit: int
    thinking_tokens: int
    tokens: int
    ecosystem: Ecosystem
    rag_db: str
    model: str
    method: str | None = None
    thread_id: str | None = None
    checkpoint_id: str | None = None
    memory_tool: bool = False

def __typechecker_stub(s: AnalysisArgs):
    pass

def __typechecker_sanity(s: AnalysisArgsD):
    __typechecker_stub(s)
