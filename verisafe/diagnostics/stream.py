from typing import Literal, Generic, TypeVar, TypedDict, List, Dict, Optional, Annotated, Union
from pydantic import Discriminator
from verisafe.prover.ptypes import StatusCodes
from verisafe.rag.types import ManualRef


UserUpdateTy = Literal["prover_result", "cex_analysis", "prover_run"]

UserUpdateTV = TypeVar("UserUpdateTV", Literal["prover_result"], Literal["prover_run"], Literal["cex_analysis"])

class UserUpdateData(TypedDict, Generic[UserUpdateTV]):
    type: UserUpdateTV

class ProverRun(UserUpdateData[Literal["prover_run"]]):
    args: List[str]

class ProverResult(UserUpdateData[Literal["prover_result"]]):
    status: Dict[str, StatusCodes]

class CEXAnalysis(UserUpdateData[Literal["cex_analysis"]]):
    rule_name: str

AuditUpdateTy = Literal["rule_result", "manual_search", "foundry_test"]

AuditUpdateTV = TypeVar("AuditUpdateTV", Literal["rule_result"], Literal["manual_search"], Literal["foundry_test"])

class AuditResult(TypedDict, Generic[AuditUpdateTV]):
    type: AuditUpdateTV
    tool_id: str

class RuleAuditResult(AuditResult[Literal["rule_result"]]):
    rule: str
    status: StatusCodes
    analysis: Optional[str]

class ManualSearchResult(AuditResult[Literal["manual_search"]]):
    ref: ManualRef

class FoundryResult(AuditResult[Literal["foundry_test"]]):
    test_function: Optional[str]
    success: bool
    return_code: int
    stdout: str
    stderr: str
    error_message: Optional[str]

ProgressUpdate = Annotated[
    Union[CEXAnalysis, ProverResult, ProverRun], Discriminator("type")
]

AuditUpdate = Annotated[
    Union[RuleAuditResult | ManualSearchResult | FoundryResult], Discriminator("type")
]

AllUpdates = ProgressUpdate | AuditUpdate
