from typing import Protocol

from composer.prover.ptypes import StatusCodes
from composer.rag.types import ManualRef


class AuditSink(Protocol):
    async def on_rule_result(
        self,
        rule: str,
        status: StatusCodes,
        analysis: str | None,
        tool_id: str,
    ) -> None: ...
    async def on_manual_search(self, tool_id: str, ref: ManualRef) -> None: ...
    async def on_summarization(self, checkpoint_id: str, summary: str) -> None: ...
