from typing import Protocol

class AnalysisArgs(Protocol):
    folder: str
    rule: str
    method: str | None
    quiet: bool