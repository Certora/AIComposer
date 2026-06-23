"""
NatSpec pipeline TUI.

Thin subclass of ``MultiJobApp`` that provides natspec-specific
task handlers, event routing, tool configs, and completion behavior.
"""

import pathlib
import traceback
from typing import cast, override

from textual.containers import VerticalScroll
from textual.widgets import Static, Collapsible, ContentSwitcher

from rich.syntax import Syntax
from rich.text import Text

from composer.ui.tool_display import ToolDisplayConfig, ToolDisplay, CommonTools, suppress_ack
from composer.io.event_handler import EventHandler, NullEventHandler
from composer.ui.ide_bridge import IDEBridge
from composer.ui.multi_job_app import (
    MultiJobApp, MultiJobTaskHandler, TaskInfo,
)
from composer.spec.natspec.pipeline import Phase, PipelineResult, ContractFormulation
from composer.spec.natspec.pipeline_events import NatspecEvent
# ---------------------------------------------------------------------------
# Phase labels and tool configs
# ---------------------------------------------------------------------------

PHASE_LABELS: dict[Phase, str] = {
    Phase.COMPONENT_ANALYSIS: "Component Analysis",
    Phase.BUG_ANALYSIS: "Property Extraction",
    Phase.INTERFACE_GEN: "Interface & Stub Generation",
    Phase.STUB_GEN: "Interface & Stub Generation",
    Phase.CVL_GEN: "CVL Generation",
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
        case Phase.COMPONENT_ANALYSIS:
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
                "memory": CommonTools.memory,
            })
        case Phase.BUG_ANALYSIS:
            return ToolDisplayConfig(tool_display={
                **CommonTools.rough_draft_displays(),
                "result": CommonTools.result,
            })
        case Phase.INTERFACE_GEN | Phase.STUB_GEN:
            return ToolDisplayConfig(tool_display={
                "result": CommonTools.result,
            })
        case Phase.CVL_GEN:
            return ToolDisplayConfig(tool_display={
                **CommonTools.cvl_research_displays(),
                **CommonTools.cvl_manipulation(),
                "give_up": ToolDisplay("Giving up on property", suppress_ack("Give up result")),
                "record_skip": ToolDisplay(lambda d: f"Skipping Property `{d['property_title']}`: {d['reason']}", suppress_ack("Skip Request Result", ("Recorded skip", ))),
                "request_stub_field": ToolDisplay(
                    lambda d: f"Requesting stub field: {d["purpose"]}",
                    "Stub field result",
                ),
                "advisory_typecheck": ToolDisplay("Type-checking spec", "Type-check result"),
                **CommonTools.cvl_research_displays(),
                "result": CommonTools.result,
                **CommonTools.rough_draft_displays(),
                "memory": CommonTools.memory,
                "feedback_tool": ToolDisplay("Seeking CVL feedback", "Feedback")
            })


# ---------------------------------------------------------------------------
# NatspecTaskHandler
# ---------------------------------------------------------------------------

class NatspecTaskHandler(MultiJobTaskHandler[None], NullEventHandler):
    """Per-task handler with natspec-specific state detection and HITL formatting."""

    async def on_node_state(self, path: list[str], node_name: str, values: dict) -> None:
        if "curr_spec" in values and isinstance(values["curr_spec"], str) and len(path) == 1:
            await self.render_content_link(
                "Working copy updated", values["curr_spec"], "working.spec",
            )

    def format_hitl_prompt(self, ty: None) -> list[Text | str]:
        raise NotImplementedError("no hitl tools in this workflow")
    
    @override
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        evt = cast(NatspecEvent, payload)
        match evt["type"]:
            case "master_spec_update":
                await self.render_content_link(
                    "Master spec updated", evt["spec"], "input.spec",
                )
            case "stub_update":
                contract_id = evt.get("contract_id", "stub")
                await self.render_content_link(
                    f"Stub updated: {contract_id}", evt["stub"], f"{contract_id}.sol",
                )



# ---------------------------------------------------------------------------
# NatspecPipelineApp
# ---------------------------------------------------------------------------

class NatspecPipelineApp(MultiJobApp[Phase, NatspecTaskHandler]):
    """Textual TUI for the NatSpec multi-agent pipeline."""

    def __init__(
        self,
        ide: IDEBridge | None = None,
        *,
        output_root: pathlib.Path | None = None,
    ):
        super().__init__(
            phase_labels=PHASE_LABELS,
            section_order=_SECTION_ORDER,
            header_text="NatSpec Pipeline | ESC: summary | q: quit (when done)",
            ide=ide,
        )
        self._output_root = output_root

    def create_task_handler(self, panel: VerticalScroll, info: TaskInfo[Phase]) -> NatspecTaskHandler:
        tc = tool_config_for_phase(info.phase)
        return NatspecTaskHandler(info.task_id, info.label, panel, self, tc)

    def create_event_handler(self, handler: NatspecTaskHandler, info: TaskInfo[Phase]) -> EventHandler:
        return handler

    # ── Pipeline completion ───────────────────────────────────

    async def on_pipeline_done(self, result: PipelineResult) -> None:
        """Render a completion summary and dump every contract's interface,
        stub, and specs under ``<output_root>/natspec_output/`` (defaulting
        to ``cwd/natspec_output/`` when no ``--output-root`` was passed).
        """
        self._pipeline_done = True

        summary = self.query_one("#summary", VerticalScroll)
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

        await summary.mount(Static(self._render_completion_banner(result)))

        out_root = (self._output_root or pathlib.Path.cwd()).resolve() / "natspec_output"
        written = self._dump_to(out_root, result)
        for path, content in sorted(written.items()):
            lexer = self._guess_lang(path) or "text"
            syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
            coll = Collapsible(Static(syntax), title=path, collapsed=True)
            await summary.mount(coll)
        await summary.mount(Static(Text(
            f"Wrote {len(written)} file(s) under {out_root}",
            style="bold green",
        )))

        await summary.mount(Static(Text("Press q to quit.", style="dim")))

    async def mount_error(self, exc: Exception) -> None:
        """Display a fatal pipeline error in the summary pane and enable quit.

        Called from the CLI entry point's outer ``except Exception`` when the
        pipeline body raises. Switches the view to ``#summary`` (the same pane
        ``on_pipeline_done`` uses) and renders the exception + traceback, so
        per-task error panels stay visible alongside the top-level cause.
        """
        self._pipeline_done = True
        summary = self.query_one("#summary", VerticalScroll)
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

        tb = "".join(traceback.format_exception(exc))
        banner = Text()
        banner.append("\n━━ Pipeline Error ━━\n\n", style="bold red")
        banner.append(f"{type(exc).__name__}: {exc}\n\n", style="red")
        banner.append(tb, style="red dim")
        banner.append("\nPress q to quit.", style="dim")
        await summary.mount(Static(banner))

    def _render_completion_banner(self, result: PipelineResult) -> Text:
        """Rich summary banner — one line per contract."""
        banner = Text()
        banner.append("\n━━ Pipeline Complete ━━\n", style="bold green")
        banner.append(f"\nApp: {result.app.application_type}\n", style="bold")
        banner.append(f"Contracts: {len(result.contracts)}\n")

        for i, c in enumerate(result.contracts, 1):
            n_specs = len(c.spec_results.specs)
            n_fail = len(c.spec_results.failures)
            banner.append(f"\n  {i}. ")
            banner.append(c.name, style="bold")
            if c.name != c.solidity_identifier:
                banner.append(f"  ({c.solidity_identifier})", style="dim")
            banner.append("\n")
            banner.append(
                f"     specs: {n_specs}   failures: {n_fail}\n",
                style="red" if n_fail else "dim",
            )
        return banner

    def _dump_to(
        self, out_root: pathlib.Path, result: PipelineResult
    ) -> dict[str, str]:
        """Write each contract's interface, stub, and specs under ``out_root``.

        Returns the ``{relative_path: content}`` map of what was written.
        Relative paths are the agent-chosen ``interface.path`` / ``stub.path``
        for greenfield, or workspace-relative for from-source mode. Spec
        filenames come from the author's ``suggested_path`` (or a synthesized
        ``<identifier>_<idx>.spec`` fallback).
        """
        out_root.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for c in result.contracts:
            written[c.interface.path] = c.interface.content
            written[c.stub.path] = c.stub.content
            for spec in self._spec_files(c):
                written[spec[0]] = spec[1]

        for rel, content in written.items():
            tgt = out_root / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(content)
        return written

    @staticmethod
    def _spec_files(c: ContractFormulation) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for i, success in enumerate(c.spec_results.specs):
            basename = success.suggested_path or f"{c.solidity_identifier}_{i}.spec"
            if not basename.endswith(".spec"):
                basename = f"{basename}.spec"
            out.append((basename, success.spec))
        return out

# Backwards compat alias
PipelineApp = NatspecPipelineApp
