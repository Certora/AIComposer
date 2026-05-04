"""
Auto-prove pipeline TUI.

Subclass of ``MultiJobApp`` for the auto-prove pipeline, which
streams prover output to per-task ``RichLog`` widgets and handles
prover lifecycle events directly via the ``NullEventHandler`` mixin.
"""

import enum
from typing import cast, override

from textual.containers import VerticalScroll
from textual.widgets import RichLog, Collapsible

from rich.text import Text

from composer.ui.tool_display import ToolDisplayConfig
from composer.io.event_handler import EventHandler, NullEventHandler
from composer.io.multi_job import TaskInfo
from composer.ui.multi_job_app import (
    MultiJobApp, MultiJobTaskHandler, TaskHost,
)
from composer.spec.source.prover import ProverOutputEvent, CloudPollingEvent
from composer.spec.source.preaudit_setup import PreAuditEvents


# ---------------------------------------------------------------------------
# Event type — events emitted by _SpecCallbacks (verify_spec tool)
# ---------------------------------------------------------------------------

type AutoProveEvent = ProverOutputEvent | CloudPollingEvent


# ---------------------------------------------------------------------------
# Phase type and labels
# ---------------------------------------------------------------------------

class AutoProvePhase(enum.Enum):
    HARNESS = "harness"
    INVARIANTS = "invariants"
    SUMMARIES = "summaries"
    COMPONENT_ANALYSIS = "component_analysis"
    BUG_ANALYSIS = "bug_analysis"
    CVL_GEN = "cvl_gen"


AUTOPROVE_PHASE_LABELS: dict[AutoProvePhase, str] = {
    AutoProvePhase.HARNESS: "Harness Setup",
    AutoProvePhase.INVARIANTS: "Structural Invariants",
    AutoProvePhase.SUMMARIES: "Summaries",
    AutoProvePhase.COMPONENT_ANALYSIS: "Component Analysis",
    AutoProvePhase.BUG_ANALYSIS: "Property Extraction",
    AutoProvePhase.CVL_GEN: "CVL Generation",
}

AUTOPROVE_SECTION_ORDER: list[str] = [
    "Harness Setup",
    "Structural Invariants",
    "Summaries",
    "Component Analysis",
    "Property Extraction",
    "CVL Generation",
]

# ---------------------------------------------------------------------------
# AutoProveTaskHandler
# ---------------------------------------------------------------------------

class AutoProveTaskHandler(MultiJobTaskHandler[None], NullEventHandler):
    """Per-task handler that doubles as its own ``EventHandler``.

    Handles prover lifecycle events (``prover_output``, ``cloud_polling``)
    by streaming output to a ``RichLog`` widget.
    """

    def __init__(
        self,
        task_id: str,
        label: str,
        panel: VerticalScroll,
        host: TaskHost,
        tool_config: ToolDisplayConfig,
    ):
        super().__init__(task_id, label, panel, host, tool_config)
        self._prover_logs: dict[str, RichLog] = {}

    def format_hitl_prompt(self, ty: None) -> list[Text | str]:
        raise NotImplementedError("Auto-prove does not support HITL interactions")

    # ── Prover output streaming ──────────────────────────────

    async def _ensure_prover_log(self, tool_call_id: str, title: str = "Prover Output") -> RichLog:
        if tool_call_id in self._prover_logs:
            return self._prover_logs[tool_call_id]
        log = RichLog(highlight=True, markup=False)
        collapsible = Collapsible(log, title=title)
        log.styles.min_height = 15
        self._prover_logs[tool_call_id] = log
        await self._mount_to(self._panel, collapsible)
        return log

    # ── EventHandler (from NullEventHandler mixin) ───────────
    @override
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        evt = cast(AutoProveEvent, payload)
        match evt["type"]:
            case "prover_output":
                evt = cast(ProverOutputEvent, evt)
                log = await self._ensure_prover_log(evt["tool_call_id"])
                log.write(evt["line"])
            case "cloud_polling":
                evt = cast(CloudPollingEvent, evt)
                log = await self._ensure_prover_log(evt["tool_call_id"])
                log.write(Text(f"[{evt['status']}] {evt['message']}", style="dim"))

    @override
    async def handle_progress_event(self, payload: dict) -> None:
        evt = cast(PreAuditEvents, payload)
        match evt["type"]:
            case "pre_audit_complete":
                log = await self._ensure_prover_log("_preaudit_setup", "PreAudit Agent")
                p : Collapsible = log.parent #type: ignore
                p.collapsed = True
            case "pre_audit_start":
                log = await self._ensure_prover_log("_preaudit_setup", "PreAudit Agent")
                p : Collapsible = log.parent #type: ignore
                p.collapsed = False
            case "pre_audit_output":
                log = await self._ensure_prover_log("_preaudit_setup", "PreAudit Agent")
                log.write(evt["line"])


# ---------------------------------------------------------------------------
# AutoProveApp
# ---------------------------------------------------------------------------

class AutoProveApp(MultiJobApp[AutoProvePhase, AutoProveTaskHandler]):
    """Textual TUI for the auto-prove pipeline."""

    def __init__(self):
        super().__init__(
            phase_labels=AUTOPROVE_PHASE_LABELS,
            section_order=AUTOPROVE_SECTION_ORDER,
            header_text="Auto-Prove | ESC: summary | q: quit (when done)",
        )

    def create_task_handler(
        self, panel: VerticalScroll, info: TaskInfo[AutoProvePhase],
    ) -> AutoProveTaskHandler:
        return AutoProveTaskHandler(info.task_id, info.label, panel, self, ToolDisplayConfig())

    def create_event_handler(
        self, handler: AutoProveTaskHandler, info: TaskInfo[AutoProvePhase],
    ) -> EventHandler:
        return handler
