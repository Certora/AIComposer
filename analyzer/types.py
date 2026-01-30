from dataclasses import dataclass
from typing import Protocol, Literal

from pydantic import BaseModel, Field

Ecosystem = Literal["evm", "soroban", "move", "solana"]


class DefaultAnalysisResult(BaseModel):
    result: str = Field(
        description="The textual analysis explaining the counterexample. "
                    "You MAY use markdown in your output."
    )


@dataclass(frozen=True)
class ResultToolConfig:
    schema: type[BaseModel]
    doc: str


DEFAULT_RESULT_TOOL_CONFIG = ResultToolConfig(
    schema=DefaultAnalysisResult,
    doc="Tool to communicate the result of your analysis.",
)


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

    @property
    def output(self) -> str | None:
        ...

    @property
    def result_tool_config(self) -> ResultToolConfig:
        ...