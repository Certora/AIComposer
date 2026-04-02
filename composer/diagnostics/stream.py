from typing import Literal, Generic, TypeVar, TypedDict, List, Dict, Optional, Annotated, Union
from pydantic import Discriminator
from composer.prover.ptypes import StatusCodes
from composer.rag.types import ManualRef


UserUpdateTy = Literal["prover_result", "cex_analysis", "prover_run", "rule_analysis", "summarization_notice", "prover_output", "cloud_polling"]

UserUpdateTV = TypeVar("UserUpdateTV", Literal["prover_result"], Literal["prover_run"], Literal["cex_analysis"], Literal["rule_analysis"], Literal["summarization_notice"], Literal["prover_output"], Literal["cloud_polling"])

class UserUpdateData(TypedDict, Generic[UserUpdateTV]):
    type: UserUpdateTV

class ProverRun(UserUpdateData[Literal["prover_run"]]):
    args: List[str]
    tool_call_id: str

class ProverResult(UserUpdateData[Literal["prover_result"]]):
    tool_call_id: str
    status: Dict[str, StatusCodes]

class RuleAnalysisResult(UserUpdateData[Literal["rule_analysis"]]):
    tool_call_id: str
    rule: str
    analysis: str

class CEXAnalysisStart(UserUpdateData[Literal["cex_analysis"]]):
    tool_call_id: str
    rule_name: str

class SummarizationNotice(UserUpdateData[Literal["summarization_notice"]]):
    summary: str

class ProverOutputEvent(UserUpdateData[Literal["prover_output"]]):
    tool_call_id: str
    line: str

class CloudPollingEvent(UserUpdateData[Literal["cloud_polling"]]):
    tool_call_id: str
    status: str
    message: str

AuditUpdateTy = Literal["rule_result", "manual_search", "summarization"]

AuditUpdateTV = TypeVar("AuditUpdateTV", Literal["rule_result"], Literal["manual_search"])

class AuditResult(TypedDict, Generic[AuditUpdateTV]):
    type: AuditUpdateTV
    tool_id: str

class RuleAuditResult(AuditResult[Literal["rule_result"]]):
    rule: str
    status: StatusCodes
    analysis: Optional[str]

class SummarizationPartial(TypedDict):
    type: Literal["summarization_raw"]
    summary: str

class Summarization(TypedDict):
    type: Literal["summarization"]
    summary: str
    checkpoint_id: str

class ManualSearchResult(AuditResult[Literal["manual_search"]]):
    ref: ManualRef

ProgressUpdate = Annotated[
    Union[CEXAnalysisStart, ProverResult, ProverRun, RuleAnalysisResult, SummarizationNotice, ProverOutputEvent, CloudPollingEvent], Discriminator("type")
]

AuditUpdate = Annotated[
    Union[RuleAuditResult | ManualSearchResult | Summarization], Discriminator("type")
]

PartialAuditUpdate = Annotated[
    Union[RuleAuditResult | ManualSearchResult | SummarizationPartial], Discriminator("type")
]

AllUpdates = ProgressUpdate | AuditUpdate

PartialUpdates = ProgressUpdate | PartialAuditUpdate