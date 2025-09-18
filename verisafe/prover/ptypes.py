from typing import Literal, Optional
from dataclasses import dataclass

StatusCodes = Literal["VERIFIED", "VIOLATED", "TIMEOUT", "ERROR"]

@dataclass
class RuleResult:
    """
    Rule result parsed out of SandboxedRunResult.
    name is the name of the rule, status is the status of the rule.
    If status == VIOLATED, then cex_dump is non-null, and will contain the XML representation
    of the CEX
    """
    name: str
    cex_dump: Optional[str]
    status: StatusCodes