from typing import Protocol, Literal

from composer.input.types import ModelOptions, LangraphOptions, RAGDBOptions

Ecosystem = Literal["evm", "soroban", "move", "solana"]

class AnalysisArgs(ModelOptions, LangraphOptions, RAGDBOptions):
    folder: str
    rule: str
    method: str | None
    quiet: bool
    ecosystem: Ecosystem

