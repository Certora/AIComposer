"""
Foundry test-generation pipeline TUI.

Subclass of ``MultiJobApp`` for the foundry pipeline, parameterized by the
pipeline's own ``FoundryPhase``.

The per-task handler doubles as its own ``EventHandler`` (via the
``NullEventHandler`` mixin) to stream ``forge test`` run summaries —
emitted by ``ForgeTestTool`` on langgraph's custom stream — into a
per-task ``RichLog``.
"""

from typing import cast, override

from textual.containers import VerticalScroll
from textual.widgets import Collapsible, RichLog

from rich.text import Text

from composer.io.event_handler import EventHandler, NullEventHandler
from composer.io.multi_job import TaskInfo
from composer.ui.multi_job_app import (
    MultiJobApp, MultiJobTaskHandler, TaskHost,
)
from composer.ui.tool_display import ToolDisplayConfig

from composer.foundry.pipeline import FoundryPhase
from composer.foundry.runner import ForgeTestRunEvent


# ---------------------------------------------------------------------------
# Phase labels
# ---------------------------------------------------------------------------

FOUNDRY_PHASE_LABELS: dict[FoundryPhase, str] = {
    FoundryPhase.SYSTEM_ANALYSIS: "System Analysis",
    FoundryPhase.PROPERTY_EXTRACTION: "Property Extraction",
    FoundryPhase.TEST_GENERATION: "Test Generation",
}

FOUNDRY_SECTION_ORDER: list[str] = [
    "System Analysis",
    "Property Extraction",
    "Test Generation",
]


# ---------------------------------------------------------------------------
# FoundryTaskHandler
# ---------------------------------------------------------------------------


class FoundryTaskHandler(MultiJobTaskHandler[None], NullEventHandler):
    """Per-task handler that doubles as its own ``EventHandler``.

    Streams ``forge_test_run`` summaries into a collapsible ``RichLog``
    mounted under the task panel.
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
        self._forge_log: RichLog | None = None

    def format_hitl_prompt(self, ty: None) -> list[Text | str]:
        raise NotImplementedError(
            "The foundry pipeline does not use HITL interrupts"
        )

    async def _ensure_forge_log(self) -> RichLog:
        if self._forge_log is None:
            log = RichLog(highlight=True, markup=False)
            log.styles.min_height = 15
            self._forge_log = log
            await self._mount_to(
                self._panel, Collapsible(log, title="Forge Test Runs"),
            )
        return self._forge_log

    @override
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        evt = cast(ForgeTestRunEvent, payload)
        match evt["type"]:
            case "forge_test_run":
                log = await self._ensure_forge_log()
                log.write(evt["summary"])
                log.write(Text("─" * 40, style="dim"))


# ---------------------------------------------------------------------------
# FoundryApp
# ---------------------------------------------------------------------------


class FoundryApp(MultiJobApp[FoundryPhase, FoundryTaskHandler]):
    """Textual TUI for the foundry test-generation pipeline."""

    def __init__(self):
        super().__init__(
            phase_labels=FOUNDRY_PHASE_LABELS,
            section_order=FOUNDRY_SECTION_ORDER,
            header_text="Foundry Test Author | ESC: summary | q: quit (when done)",
        )

    def create_task_handler(
        self, panel: VerticalScroll, info: TaskInfo[FoundryPhase],
    ) -> FoundryTaskHandler:
        return FoundryTaskHandler(info.task_id, info.label, panel, self, ToolDisplayConfig())

    def create_event_handler(
        self, handler: FoundryTaskHandler, info: TaskInfo[FoundryPhase],
    ) -> EventHandler:
        return handler
