"""
NatSpec pipeline TUI.

Thin subclass of ``MultiJobApp`` that provides natspec-specific
task handlers, event routing, tool configs, and completion behavior.
"""

import json
import pathlib
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
from composer.spec.natspec.pipeline import Phase, PipelineResult
from composer.spec.natspec.pipeline_events import NatspecEvent
from composer.spec.natspec.codegen_export import (
    ImplementationPlan,
    build_implementation_plan,
    plan_preview_files,
    plan_to_json,
)
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
        system_doc_path: pathlib.Path | None = None,
        source_root: pathlib.Path | None = None,
        prover_conf: dict | None = None,
        output_root: pathlib.Path | None = None,
    ):
        super().__init__(
            phase_labels=PHASE_LABELS,
            section_order=_SECTION_ORDER,
            header_text="NatSpec Pipeline | ESC: summary | q: quit (when done)",
            ide=ide,
        )
        self._system_doc_path = system_doc_path
        self._source_root = source_root
        self._prover_conf = prover_conf
        self._output_root = output_root
        self._plan: ImplementationPlan | None = None

    def create_task_handler(self, panel: VerticalScroll, info: TaskInfo[Phase]) -> NatspecTaskHandler:
        tc = tool_config_for_phase(info.phase)
        return NatspecTaskHandler(info.task_id, info.label, panel, self, tc)

    def create_event_handler(self, handler: NatspecTaskHandler, info: TaskInfo[Phase]) -> EventHandler:
        return handler

    # ── Pipeline completion ───────────────────────────────────

    async def on_pipeline_done(self, result: PipelineResult) -> None:
        """Build the implementation plan, render a summary, and write the
        generated interfaces / stubs / specs (plus the plan JSON) under
        ``--output-root`` if set; otherwise emit a warning."""
        self._pipeline_done = True

        summary = self.query_one("#summary", VerticalScroll)
        switcher = self.query_one("#switcher", ContentSwitcher)
        switcher.current = "summary"

        plan = build_implementation_plan(
            result,
            system_doc_path=self._system_doc_path or pathlib.Path("<unknown>"),
            source_root=self._source_root,
            prover_conf=self._prover_conf,
        )
        self._plan = plan

        await summary.mount(Static(self._render_completion_banner(plan, result)))

        files = plan_preview_files(plan)

        plan_json_path = self._write_plan_json(plan)
        if plan_json_path is not None:
            await summary.mount(
                Static(Text(f"Plan: {plan_json_path}", style="bold cyan"))
            )

        if self._output_root is not None:
            written = self._write_files_under_output_root(files)
            for path, content in sorted(written.items()):
                lexer = self._guess_lang(path) or "text"
                syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title=path, collapsed=True)
                await summary.mount(coll)
            await summary.mount(Static(Text(
                f"Wrote {len(written)} file(s) under {self._output_root}",
                style="bold green",
            )))
        else:
            await summary.mount(Static(Text(
                "No --output-root set. No files written.\n"
                "Rerun with --output-root <dir> to persist.",
                style="bold yellow",
            )))

        await summary.mount(Static(Text("Press q to quit.", style="dim")))

    def _render_completion_banner(
        self,
        plan: ImplementationPlan,
        result: PipelineResult,
    ) -> Text:
        """Rich summary of the plan for the completion banner."""
        banner = Text()
        banner.append("\n━━ Pipeline Complete ━━\n", style="bold green")

        if plan.cycles:
            banner.append(
                f"\n! {len(plan.cycles)} dependency cycle(s) detected - "
                "those contracts are omitted from the plan order.\n",
                style="bold yellow",
            )
            for cyc in plan.cycles:
                banner.append(
                    f"  cycle: {' -> '.join(cyc)} -> {cyc[0]}\n",
                    style="yellow",
                )

        banner.append(f"\nApp: {plan.application_type}\n", style="bold")
        banner.append(f"Contracts in dep order: {len(plan.contracts)}\n")

        failures_by_name = {
            c.name: len(c.spec_results.failures) for c in result.contracts
        }

        for i, c in enumerate(plan.contracts, 1):
            n_fail = failures_by_name.get(c.name, 0)
            banner.append(f"\n  {i}. ")
            banner.append(c.name, style="bold")
            if c.tag:
                banner.append(f"  [{c.tag}]", style="dim")
            if c.depends_on:
                banner.append(f"  <- {', '.join(c.depends_on)}", style="dim")
            banner.append("\n")
            banner.append(
                f"     specs: {len(c.specs)}   "
                f"stub fields: {len(c.required_stub_fields)}   "
                f"failures: {n_fail}\n",
                style="red" if n_fail else "dim",
            )

        return banner

    def _write_plan_json(self, plan: ImplementationPlan) -> pathlib.Path | None:
        """Write ``implementation_plan.json`` under ``--output-root`` if set.

        Returns the absolute path written, or ``None`` if no output_root was
        configured on this app.
        """
        if self._output_root is None:
            return None
        out = self._output_root.resolve()
        out.mkdir(parents=True, exist_ok=True)
        path = out / "implementation_plan.json"
        path.write_text(json.dumps(plan_to_json(plan), indent=2, default=str))
        return path


    def _write_files_under_output_root(
        self, files: dict[str, str]
    ) -> dict[str, str]:
        """Persist every file in the preview map under ``--output-root``.

        Called only when no IDE bridge is available. Returns the files that
        were actually written (all of them; kept as a map for symmetry with
        the preview flow).
        """
        assert self._output_root is not None
        root = self._output_root.resolve()
        for path, content in files.items():
            tgt = root / path
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_text(content)
        return files

# Backwards compat alias
PipelineApp = NatspecPipelineApp
