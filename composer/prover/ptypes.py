from typing import Literal, Optional, TypeVar
from dataclasses import dataclass

StatusCodes = Literal["VERIFIED", "VIOLATED", "TIMEOUT", "ERROR", "SANITY_FAILED"]

class _Missing:
    pass

_MISSING = _Missing()

_T = TypeVar('_T')

def _default_or(
    curr: _T,
    update: _Missing | _T
) -> _T:
    if isinstance(update, _Missing):
        return curr
    else:
        return update

@dataclass
class RulePath:
    rule: str
    contract: Optional[str] = None
    method: Optional[str] = None
    sanity: bool = False

    def copy(
            self,
            rule : str | _Missing = _MISSING,
            contract : str | None | _Missing = _MISSING,
            method : str | None | _Missing = _MISSING,
            sanity : bool | _Missing = _MISSING
    ) -> 'RulePath':
        return RulePath(
            rule=_default_or(self.rule, rule),
            contract=_default_or(self.contract, contract),
            method=_default_or(self.method, method),
            sanity=_default_or(self.sanity, sanity)
        )
    def pprint(self) -> str:
        if self.contract is not None:
            if self.method is None:
                return f"{self.rule} in contract {self.contract}"

        if self.method is not None:
            return f"{self.rule} for {self.method}"
        else:
            return self.rule



@dataclass
class RuleResult:
    """
    Rule result parsed out of SandboxedRunResult.
    name is the name of the rule, status is the status of the rule.
    If status == VIOLATED, then cex_dump is non-null, and will contain the XML representation
    of the CEX
    """
    path: RulePath
    cex_dump: Optional[str]
    status: StatusCodes

    @property
    def name(self) -> str:
        return self.path.pprint()