from typing import Literal, Optional, TypeVar
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

StatusCodes = Literal["VERIFIED", "VIOLATED", "TIMEOUT", "ERROR", "SANITY_FAILED", "SKIPPED"]

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

@dataclass(frozen=True)
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

    If status == ERROR, error_msg is non-none
    """
    path: RulePath
    cex_dump: Optional[str]
    status: StatusCodes

    error_messages: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.pprint()


class AnalyzedDiagnosis(BaseModel):
    """A single root cause shared by one or more failing rule instances.

    Used internally by handlers (and by the codegen-side report store
    that ``cex_remediation`` looks up against). NOT part of the
    ``CexHandler`` interface — handlers that don't mint keyed
    diagnoses don't need to construct these.

    BaseModel rather than a frozen dataclass so that the report store
    can round-trip via ``model_dump()`` / ``model_validate()``. Pydantic
    v2 handles stdlib-dataclass fields like ``RulePath`` transparently
    inside a BaseModel.
    """

    model_config = ConfigDict(frozen=True)

    report_key: str
    diagnosis: str
    attributed_rules: list[RulePath]
