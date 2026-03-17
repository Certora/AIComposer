"""
NatSpec pipeline TUI.

Thin subclass of ``MultiJobApp`` that provides natspec-specific
task handlers, event routing, tool configs, and completion behavior.
"""

import asyncio
import pathlib
from typing import cast

from textual.containers import VerticalScroll
from textual.widgets import Static, Input, Collapsible, ContentSwitcher

from rich.syntax import Syntax
from rich.text import Text

from composer.ui.tool_display import ToolDisplayConfig, ToolDisplay, CommonTools, _suppress_ack
from composer.io.event_handler import EventHandler
from composer.ui.ide_bridge import IDEBridge
from composer.ui.multi_job_app import (
    MultiJobApp, MultiJobTaskHandler, TaskInfo,
)
from composer.spec.natspec_pipeline import Phase, PipelineResult
from composer.spec.pipeline_events import NatspecEvent
from composer.spec.ptypes import HumanQuestionSchema


# ---------------------------------------------------------------------------
# Phase labels and tool configs
# ---------------------------------------------------------------------------

PHASE_LABELS: dict[Phase, str] = {
    "component_analysis": "Component Analysis",
    "bug_analysis": "Property Extraction",
    "interface_gen": "Interface & Stub Generation",
    "stub_gen": "Interface & Stub Generation",
    "cvl_gen": "CVL Generation",
}

_SECTION_ORDER: list[str] = [
    "Component Analysis",
    "Property Extraction",
    "Interface & Stub Generation",
    "CVL Generation",
]


def tool_config_for_phase(phase: Phase) -> ToolDisplayConfig:
    """Return the appropriate ``ToolDisplayConfig`` for *phase*."""
    match phase:
        case "component_analysis":
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
                "memory": CommonTools.memory,
            })
        case "bug_analysis":
            return ToolDisplayConfig(tool_display={
                **CommonTools.rough_draft_displays(),
                "result": CommonTools.result,
            })
        case "interface_gen" | "stub_gen":
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
            })
        case "cvl_gen":
            return ToolDisplayConfig(tool_display={
                **CommonTools.cvl_research_displays(),
                **CommonTools.cvl_manipulation(),
                "extended_reasoning": CommonTools.extended_reasoning,
                "publish_spec": ToolDisplay("Publishing to master spec", _suppress_ack("Publish result")),
                "give_up": ToolDisplay("Giving up on property", _suppress_ack("Give up result")),
                "read_stub": ToolDisplay("Reading verification stub", None),
                "request_stub_field": ToolDisplay("Requesting stub field", "Stub field result"),
                "advisory_typecheck": ToolDisplay("Type-checking spec", "Type-check result"),
                **CommonTools.cvl_research_displays(),
                "result": CommonTools.result,
                **CommonTools.rough_draft_displays(),
                "memory": CommonTools.memory,
            })


# ---------------------------------------------------------------------------
# NatspecTaskHandler
# ---------------------------------------------------------------------------

class NatspecTaskHandler(MultiJobTaskHandler[HumanQuestionSchema]):
    """Per-task handler with natspec-specific state detection and HITL formatting."""

    async def on_node_state(self, path: list[str], node_name: str, values: dict) -> None:
        if "curr_spec" in values and isinstance(values["curr_spec"], str) and len(path) == 1:
            await self.render_content_link(
                "Working copy updated", values["curr_spec"], "working.spec",
            )

    def format_hitl_prompt(self, ty: HumanQuestionSchema) -> list[Text | str]:
        parts: list[Text | str] = [Text("Question: ", style="bold yellow"), ty.question]
        if ty.context:
            parts.append(f"\n  Context: {ty.context}")
        return parts
    
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        evt = cast(NatspecEvent, payload)
        match evt["type"]:
            case "master_spec_update":
                await self.render_content_link(
                    "Master spec updated", evt["spec"], "input.spec",
                )
            case "stub_update":
                await self.render_content_link(
                    "Stub updated", evt["stub"], "Impl.sol",
                )


# ---------------------------------------------------------------------------
# NatspecPipelineApp
# ---------------------------------------------------------------------------

class NatspecPipelineApp(MultiJobApp[Phase, NatspecTaskHandler]):
    """Textual TUI for the NatSpec multi-agent pipeline."""

    def __init__(self, ide: IDEBridge | None = None):
        super().__init__(
            phase_labels=PHASE_LABELS,
            section_order=_SECTION_ORDER,
            header_text="NatSpec Pipeline | ESC: summary | q: quit (when done)",
            ide=ide,
        )

    def create_task_handler(self, panel: VerticalScroll, info: TaskInfo[Phase]) -> NatspecTaskHandler:
        tc = tool_config_for_phase(info.phase)
        return NatspecTaskHandler(info.task_id, info.label, panel, self, tc)

    def create_event_handler(self, handler: NatspecTaskHandler, info: TaskInfo[Phase]) -> EventHandler:
        return handler

    # ── Pipeline completion ───────────────────────────────────

    async def on_pipeline_done(self, result: PipelineResult) -> None:
        """Show completion banner, preview results, and enable quit."""
        self._pipeline_done = True

        summary = self.query_one("#summary", VerticalScroll)
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

        n_fail = len(result.failures)

        banner_text = Text()
        banner_text.append("\n━━ Pipeline Complete ━━\n", style="bold green")
        banner_text.append(f"Contract: {result.contract_name} (solc {result.solc_version})\n")
        banner_text.append(f"Failures: {n_fail}\n" if n_fail else "All properties succeeded\n")
        if n_fail:
            for f in result.failures:
                banner_text.append(f"  \u2717 {f.prop.description}: {f.reason}\n", style="red")

        await summary.mount(Static(banner_text))

        files: dict[str, str] = {
            "input.spec": result.spec,
            "Impl.sol": result.stub,
            "Intf.sol": result.interface,
        }

        if self._ide is not None:
            preview_id: str | None = None
            try:
                preview_id = await self._ide.preview_results(files)
            except Exception:
                self.notify("Failed to preview results in VS Code", severity="warning")

            if preview_id is not None:
                await self._show_accept_reject_prompt(summary, preview_id)
            else:
                await summary.mount(
                    Static(Text("Preview unavailable.", style="dim"))
                )
        else:
            out_dir = pathlib.Path.cwd()
            for path, content in files.items():
                (out_dir / path).write_text(content)
                lexer = self._guess_lang(path) or "text"
                syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title=path, collapsed=False)
                await summary.mount(coll)
            await summary.mount(
                Static(Text(f"Wrote {len(files)} file(s) to {out_dir}", style="bold green"))
            )

        await summary.mount(Static(Text("Press q to quit.", style="dim")))

    async def _show_accept_reject_prompt(
        self,
        summary: VerticalScroll,
        preview_id: str,
    ) -> None:
        """Show ACCEPT/REJECT prompt and handle the IDE preview lifecycle."""
        assert self._ide is not None

        prompt_widget = Static(Text.assemble(
            ("Results previewed in VS Code.\n", "bold"),
            ("Type ACCEPT to write files or REJECT to discard.", "dim"),
        ))
        hint_widget = Static("Response must be ACCEPT or REJECT", classes="interaction-hint")
        input_widget = Input(placeholder="ACCEPT / REJECT", validate_on=["submitted"])

        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._active_inputs[input_widget] = queue

        await summary.mount(prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        response = await queue.get()
        del self._active_inputs[input_widget]

        await prompt_widget.remove()
        await input_widget.remove()
        await hint_widget.remove()

        decision = response.strip().upper()
        if decision == "ACCEPT":
            try:
                written = await self._ide.accept_results(preview_id)
                await summary.mount(
                    Static(Text(f"Results accepted — wrote {len(written)} file(s).", style="bold green"))
                )
            except Exception:
                self.notify("Failed to accept results in VS Code", severity="warning")
        else:
            try:
                await self._ide.reject_results(preview_id)
            except Exception:
                pass
            await summary.mount(Static(Text("Results rejected.", style="yellow")))


# Backwards compat alias
PipelineApp = NatspecPipelineApp
