from typing import cast

from composer.io.event_handler import EventHandler
from composer.io.protocol import CodeGenIOHandler
from composer.audit.sink import AuditSink

from composer.diagnostics.stream import (
    PartialUpdates, SummarizationNotice,
)
from composer.diagnostics.handlers import is_user_update, is_audit_update


class CodeGenEventHandler(EventHandler):
    def __init__(
        self,
        io: CodeGenIOHandler,
        audit: AuditSink | None
    ):
        self._io = io
        self._audit = audit

    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        d = cast(PartialUpdates, payload)
        if d["type"] == "summarization_raw":
            if self._audit is not None:
                self._audit.on_summarization(checkpoint_id=checkpoint_id, summary=d["summary"])
            notice: SummarizationNotice = {"type": "summarization_notice", "summary": d["summary"]}
            await self._io.progress_update(path, notice)
        elif is_audit_update(d) and self._audit is not None:
            match d["type"]:
                case "rule_result":
                    self._audit.on_rule_result(
                        rule=d["rule"],
                        status=d["status"],
                        analysis=d["analysis"],
                        tool_id=d["tool_id"]
                    )
                case "manual_search":
                    self._audit.on_manual_search(
                        tool_id=d["tool_id"],
                        ref=d["ref"]
                    )
        elif is_user_update(d):
            await self._io.progress_update(path, d)
