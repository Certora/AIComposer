from typing import Protocol
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Protocols and standalone params
# ---------------------------------------------------------------------------

class ProjectParamProtocol(Protocol):
    """Minimum context needed for harness analysis.

    ``SourceCode`` satisfies this structurally.
    """
    @property
    def project_root(self) -> str: ...
    @property
    def contract_name(self) -> str: ...
    @property
    def relative_path(self) -> str: ...


@dataclass
class LLMParams:
    """Parameters to construct an LLM via ``create_llm`` for standalone use.

    Satisfies the ``ModelOptions`` protocol.
    """
    model: str
    tokens: int
    thinking_tokens: int
    memory_tool: bool

