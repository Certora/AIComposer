"""
Auto-prove pipeline TUI.

Subclass of ``MultiJobApp`` for the auto-prove pipeline, which
streams prover output to per-task ``RichLog`` widgets and handles
prover lifecycle events directly via the ``NullEventHandler`` mixin.
"""

import enum
from typing import cast, override

from textual.containers import VerticalScroll
from textual.widgets import Static, RichLog, Collapsible

from rich.text import Text

from composer.ui.tool_display import ToolDisplayConfig, ToolDisplay, CommonTools, _suppress_ack
from composer.io.event_handler import EventHandler, NullEventHandler
from composer.ui.multi_job_app import (
    MultiJobApp, MultiJobTaskHandler, TaskInfo, TaskHost,
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


def _tool_config_for_phase(phase: AutoProvePhase) -> ToolDisplayConfig:
    match phase:
        case AutoProvePhase.HARNESS:
            return ToolDisplayConfig(tool_display={
                **CommonTools.source_displays(),
                "erc20_guidance": ToolDisplay("ERC20 guidance", None),
                "result": CommonTools.result,
                "memory": CommonTools.memory,
            })
        case AutoProvePhase.SUMMARIES:
            return ToolDisplayConfig(tool_display={
                **CommonTools.source_displays(),
                **CommonTools.cvl_research_displays(),
                "put_cvl": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
                "put_cvl_raw": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
                "get_cvl": ToolDisplay("Reading spec", None),
                "typechecker": ToolDisplay("Type checking", "Typecheck result"),
                "plan_write": ToolDisplay("Writing plan", _suppress_ack("Plan accepted")),
                "read_plan": ToolDisplay("Reading plan", None),
                "erc20_guidance": ToolDisplay("ERC20 guidance", None),
                "resolution_guidance": ToolDisplay("Resolution guidance", None),
                "result": CommonTools.result,
                "memory": CommonTools.memory,
            })
        case AutoProvePhase.INVARIANTS:
            return ToolDisplayConfig(tool_display={
                "invariant_feedback": ToolDisplay("Getting feedback", "Invariant feedback"),
                "result": CommonTools.result,
                "memory": CommonTools.memory,
                **CommonTools.source_displays(),
                **CommonTools.rough_draft_displays()
            })
        case AutoProvePhase.COMPONENT_ANALYSIS:
            return ToolDisplayConfig(tool_display={
                **CommonTools.source_displays(),
                "result": CommonTools.result,
                "memory": CommonTools.memory,
                "explore_code": CommonTools.code_explorer
            })
        case AutoProvePhase.BUG_ANALYSIS:
            return ToolDisplayConfig(tool_display={
                **CommonTools.source_displays(),
                **CommonTools.rough_draft_displays(),
                "result": CommonTools.result,
            })
        case AutoProvePhase.CVL_GEN:
            return ToolDisplayConfig(tool_display={
                **CommonTools.cvl_research_displays(),
                **CommonTools.source_displays(),
                "put_cvl": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
                "put_cvl_raw": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
                "get_cvl": ToolDisplay("Reading spec", None),
                "feedback_tool": ToolDisplay("Getting feedback", "Feedback"),
                "extended_reasoning": CommonTools.extended_reasoning,
                "record_skip": ToolDisplay(
                    lambda p: f"Skipping property #{p.get('property_index', '?')}",
                    _suppress_ack("Skip result", ("Recorded skip",)),
                ),
                "unskip_property": ToolDisplay(
                    lambda p: f"Un-skipping property #{p.get('property_index', '?')}",
                    _suppress_ack("Unskip result", ("Removed skip",)),
                ),
                "explore_code": ToolDisplay("Exploring code", "Code exploration"),
                "verify_spec": ToolDisplay("Running prover", None),
                "unresolved_call_guidance": ToolDisplay("Unresolved call guidance", "Guidance"),
                "result": CommonTools.result,
                "memory": CommonTools.memory,
            })


# ---------------------------------------------------------------------------
# AutoProveTaskHandler
# ---------------------------------------------------------------------------

from logging import getLogger
_logger = getLogger(__name__)

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
        _logger.error(str(payload))
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
        tc = _tool_config_for_phase(info.phase)
        return AutoProveTaskHandler(info.task_id, info.label, panel, self, tc)

    def create_event_handler(
        self, handler: AutoProveTaskHandler, info: TaskInfo[AutoProvePhase],
    ) -> EventHandler:
        return handler
