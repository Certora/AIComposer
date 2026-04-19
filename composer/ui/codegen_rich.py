import difflib

from rich.spinner import Spinner

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static, Input, Collapsible, DataTable, Header, Footer, Markdown
from textual.widgets.data_table import RowKey, ColumnKey
from textual.validation import Function, Validator
from textual.timer import Timer

from rich.syntax import Syntax
from rich.text import Text

from composer.ui.ide_bridge import IDEBridge
from composer.ui.tool_display import ToolDisplayConfig, ToolDisplay
from composer.ui.rich_console import BaseRichConsoleApp
from composer.io.protocol import WorkflowPurpose
from composer.workflow.types import WorkflowResult, WorkflowSuccess
from composer.ui.message_renderer import _DOT

from graphcore.tools.vfs import VFSAccessor

from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import (
    HumanInteractionType, ProposalType, QuestionType,
    RequirementRelaxationType, ExtractionQuestionType,
)
from composer.core.state import ResultStateSchema, AIComposerState
from composer.prover.ptypes import StatusCodes
from composer.prover.cloud import _TERMINAL_STATUSES

_STATUS_STYLES: dict[StatusCodes, str] = {
    "VERIFIED": "green",
    "VIOLATED": "bold red",
    "TIMEOUT": "yellow",
    "ERROR": "red",
    "SANITY_FAILED": "magenta",
}

import logging
logger = logging.getLogger(__name__)

class _ContentScreen(Screen):
    """Full-screen fallback viewer: Header + title + scrollable body + Footer.

    Used when no IDE bridge is attached — callers push an instance with
    a title Text and a pre-built body widget (e.g. ``Static(Syntax(...))``
    for file snapshots, ``Markdown(...)`` for prover CEX analyses).
    Escape pops the screen.
    """

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=True),
    ]

    CSS = """
    _ContentScreen { background: $surface; }
    #content-body { height: 1fr; }
    """

    def __init__(self, title: Text, body: Widget):
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(
            Static(self._title),
            self._body,
            id="content-body",
        )
        yield Footer()

    def action_dismiss_screen(self) -> None:
        self.app.pop_screen()


class _ProverSpinner(Static):
    """Animated spinner for cloud polling status."""

    def __init__(self, message: str):
        super().__init__("")
        self._spinner = Spinner("dots", message)
        self.timer : Timer | None = None

    def on_mount(self) -> None:
        self.timer = self.set_interval(1 / 12, self._tick)

    def update_message(self, message: str) -> None:
        self._spinner.text = message

    def finish(self, msg: str) -> None:
        if self.timer is not None:
            self.timer.stop()
        self.update(msg)


    def _tick(self) -> None:
        self.update(self._spinner)


class CodeGenRichApp(BaseRichConsoleApp[HumanInteractionType, ProgressUpdate]):
    """Textual TUI for the code generation workflow."""

    DEFAULT_CSS = """
.prover-scroll {
   max-height: 15;
}
"""

    def __init__(self, show_checkpoints: bool = False, ide: IDEBridge | None = None):
        super().__init__(
            tool_config=ToolDisplayConfig(
                tool_display={
                    "requirement_relaxation_request": ToolDisplay(
                        lambda p: (
                            f"Requesting requirement relaxation #{p.get('req_number', '?')}: {p.get('req_text', '')}"
                            if p.get("req_text")
                            else "Requesting requirement relaxation"
                        ),
                        None,
                    ),

                    "propose_spec_change": ToolDisplay(
                        lambda p: (
                            f"Proposing spec change: {p['explanation']}"
                            if p.get("explanation") else "Proposing spec change"
                        ),
                        None,
                    ),
                    "human_in_the_loop": ToolDisplay(
                        lambda p: (
                            f"Asking for input: {p['question']}"
                            if p.get("question") else "Asking for input"
                        ),
                        None,
                    ),
                }
            ),
            show_checkpoints=show_checkpoints,
            ide=ide,
        )
        self._prover_table: DataTable | None = None
        self._analysis_col: ColumnKey | None = None
        self._rule_row_keys: dict[str, RowKey] = {}
        self._rule_analyses: dict[str, str] = {}
        self._tool_output_panes: dict[str, VerticalScroll] = {}
        self._tool_spinners: dict[str, _ProverSpinner] = {}
        self.workflow_threads: dict[WorkflowPurpose, str] = {}
        self.result: WorkflowResult | None = None

    @property
    def exit_code(self) -> int:
        return 0 if isinstance(self.result, WorkflowSuccess) else 1

    # ── CodeGenIOHandler protocol ───────────────────────────────

    async def log_workflow_thread(self, purpose: WorkflowPurpose, thread_id: str) -> None:
        self.workflow_threads[purpose] = thread_id
        if purpose == WorkflowPurpose.CODEGEN:
            await self._mounted.wait()
            self._session_id = thread_id
            self._update_status_bar()

    # ── Abstract method implementations ───────────────────────

    def build_interaction(self, ty: HumanInteractionType) -> tuple[Text, str, list[Validator]]:
        _PROPOSAL_VALIDATOR: list[Validator] = [Function(
            lambda x: x.startswith("ACCEPTED") or x.startswith("REJECTED") or x.startswith("REFINE"),
            "Response must begin with ACCEPTED/REJECTED/REFINE",
        )]
        _REQ_VALIDATOR: list[Validator] = [Function(
            lambda r: r.startswith("ACCEPTED") or r.startswith("REJECTED"),
            "Response must begin with ACCEPTED/REJECTED",
        )]

        match ty["type"]:
            case "proposal":
                return self._build_proposal(ty), "Response must start with ACCEPTED, REJECTED, or REFINE", _PROPOSAL_VALIDATOR
            case "question":
                return self._build_question(ty), "Begin response with FOLLOWUP to request clarification", []
            case "extraction_question":
                return self._build_extraction_question(ty), "Enter your response", []
            case "req_relaxation":
                return self._build_req_relaxation(ty), "Response must start with ACCEPTED or REJECTED", _REQ_VALIDATOR
            case _:
                return Text("Unknown interaction type"), "", []

    async def render_progress(self, target: VerticalScroll, path: list[str], upd: ProgressUpdate) -> None:
        match upd["type"]:
            case "prover_run":
                tool_call_id = upd["tool_call_id"]
                logger.info("Prover run info")
                anchor = self._renderer.get_tool_call_anchor(tool_call_id)
                logger.info(f"Anchor is non-null? {anchor is not None}")
                if anchor is not None:
                    inner = VerticalScroll()
                    coll = Collapsible(inner, title="Prover output", collapsed=False, classes="prover-scroll")
                    parent = anchor.parent
                    assert isinstance(parent, VerticalScroll)
                    await parent.mount(coll, after=anchor)
                    await self._auto_scroll()
                    self._tool_output_panes[tool_call_id] = inner
            case "prover_output":
                pane = self._tool_output_panes.get(upd["tool_call_id"])
                if pane is not None:
                    await self._mount_to(pane, Static(Text(upd["line"], style="dim")))
            case "cloud_polling":
                tool_call_id = upd["tool_call_id"]
                pane = self._tool_output_panes.get(tool_call_id)
                if pane is not None:
                    existing = self._tool_spinners.get(tool_call_id)
                    if existing is not None:
                        existing.update_message(f"Waiting for cloud: {upd['message']}")
                    else:
                        spinner = _ProverSpinner(f"Waiting for cloud: {upd['message']}")
                        existing = spinner
                        self._tool_spinners[tool_call_id] = spinner
                        await self._mount_to(pane, spinner)
                    if upd["status"] in _TERMINAL_STATUSES:
                        existing.finish("Complete")
            case "prover_result":
                tool_call_id = upd["tool_call_id"]
                # Collapse the stdout output pane
                self._tool_spinners.pop(tool_call_id, None)
                pane = self._tool_output_panes.pop(tool_call_id, None)
                if pane is not None:
                    parent_coll = pane.parent
                    if isinstance(parent_coll, Collapsible):
                        parent_coll.collapsed = True
                # Render results table
                table = DataTable()
                _, _, self._analysis_col = table.add_columns("Rule", "Status", "Analysis")
                self._rule_row_keys.clear()
                self._rule_analyses.clear()
                for rule, status in upd["status"].items():
                    style = _STATUS_STYLES.get(status, "white")
                    analysis_cell = Text("...", style="dim") if status == "VIOLATED" else Text("")
                    row_key = table.add_row(
                        Text(rule, style="bold"),
                        Text(status, style=style),
                        analysis_cell,
                    )
                    self._rule_row_keys[rule] = row_key
                self._prover_table = table
                await self._mount_to(target, table)
            case "cex_analysis":
                rule_name = upd["rule_name"]
                row_key = self._rule_row_keys.get(rule_name)
                if row_key is not None and self._prover_table is not None and self._analysis_col is not None:
                    self._prover_table.update_cell(
                        row_key, self._analysis_col,
                        Text("Analyzing...", style="dim italic"),
                        update_width=True,
                    )
            case "rule_analysis":
                rule_name = upd["rule"]
                self._rule_analyses[rule_name] = upd["analysis"]
                row_key = self._rule_row_keys.get(rule_name)
                if row_key is not None and self._prover_table is not None and self._analysis_col is not None:
                    self._prover_table.update_cell(
                        row_key, self._analysis_col,
                        Text("View Analysis", style="bold underline cyan"),
                        update_width=True,
                    )
            case "summarization_notice":
                await self._mount_to(
                    target,
                    Static(Text("Context compacted (summarization applied)", style="dim italic"))
                )

    # ── Overrides ─────────────────────────────────────────────

    def _show_content_fallback(
        self, snap_id: int, label: str, content: str, filename: str
    ) -> None:
        """No-IDE fallback: push a full-screen viewer for the snapshot."""
        lang = self._guess_lang(filename)
        # Syntax lexer expects a string name; fall back to "text" for unknown.
        syntax = Syntax(
            content, lang or "text", theme="monokai",
            line_numbers=True, word_wrap=False,
        )
        title = Text.assemble((label, "bold "), ("  ", ""), (filename, "dim"))
        self.push_screen(_ContentScreen(title, Static(syntax)))

    async def render_state_extras(self, target: VerticalScroll, node_name: str, node_data: dict) -> None:
        if "vfs" not in node_data:
            return
        self._reset_tool_collapsing()
        count = len(node_data["vfs"])
        names = list(node_data["vfs"].keys())
        contents = {
            k: val.decode("utf-8") if isinstance(val, bytes) else val
            for k, val in node_data["vfs"].items()
        }

        links = []
        for name in names:
            snap_id = self._store_snapshot(name, contents[name], name)
            links.append(self._make_content_link_markup(snap_id, name))
        markup = f"[cyan]{_DOT}[/cyan]Wrote {count} file{'s' if count != 1 else ''}: " + ", ".join(links)
        widget = Static(markup, classes="vfs-change")
        await self._mount_to(target, widget)

    # ── DataTable cell click (analysis view) ──────────────────

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        if event.coordinate.column != 2:
            return
        for rule_name, row_key in self._rule_row_keys.items():
            if row_key == event.cell_key.row_key:
                if rule_name in self._rule_analyses:
                    text = self._rule_analyses[rule_name]
                    if self._ide is not None:
                        self.run_worker(self._ide_show_analysis(rule_name, text), thread=False)
                    else:
                        title = Text.assemble(
                            ("Analysis: ", "bold "),
                            (rule_name, "bold cyan"),
                        )
                        self.push_screen(_ContentScreen(title, Markdown(text)))
                return

    async def _ide_show_analysis(self, rule_name: str, analysis: str) -> None:
        try:
            assert self._ide is not None
            await self._ide.show_webview(analysis, title=f"Analysis: {rule_name}")
        except Exception:
            self.notify("Failed to show analysis in VS Code", severity="warning")

    # ── Interaction builders ──────────────────────────────────

    def _build_proposal(self, ty: ProposalType) -> Text:
        parts: list[tuple[str, str] | str | Text] = [
            ("SPEC CHANGE PROPOSAL\n\n", "bold"),
            ("Explanation: ", "bold"),
            ty["explanation"],
            "\n\n",
        ]

        if self._ide is not None:
            self.run_worker(self._show_proposal_diff(ty), thread=False)
            parts.append(("Diff opened in VS Code.\n", "dim italic"))
        else:
            current_lines = ty["current_spec"].splitlines(keepends=True)
            proposed_lines = ty["proposed_spec"].splitlines(keepends=True)
            diff_lines = list(difflib.unified_diff(
                current_lines, proposed_lines,
                fromfile="current", tofile="proposed",
            ))
            if diff_lines:
                diff_text = Text()
                for line in diff_lines:
                    stripped = line.rstrip("\n")
                    if line.startswith("+++") or line.startswith("---"):
                        diff_text.append(stripped + "\n", style="bold white")
                    elif line.startswith("@@"):
                        diff_text.append(stripped + "\n", style="cyan")
                    elif line.startswith("+"):
                        diff_text.append(stripped + "\n", style="green")
                    elif line.startswith("-"):
                        diff_text.append(stripped + "\n", style="red")
                    else:
                        diff_text.append(stripped + "\n")
                parts.append(("Diff:\n", "bold"))
                parts.append(diff_text)

        return Text.assemble(*parts)

    async def _show_proposal_diff(self, ty: ProposalType) -> None:
        try:
            assert self._ide is not None
            await self._ide.show_diff(ty["current_spec"], ty["proposed_spec"], "Spec Change Proposal")
        except Exception:
            self.notify("Failed to open diff in VS Code", severity="warning")

    @staticmethod
    def _build_question(ty: QuestionType) -> Text:
        parts: list[tuple[str, str] | str] = [
            ("HUMAN ASSISTANCE REQUESTED\n\n", "bold"),
            ("Question: ", "bold"),
            ty["question"],
            "\n",
            ("Context: ", "bold"),
            ty["context"],
        ]
        if ty["code"]:
            parts.append("\n\nCode:\n")
            parts.append(ty["code"])
        return Text.assemble(*parts)

    @staticmethod
    def _build_extraction_question(ty: ExtractionQuestionType) -> Text:
        return Text.assemble(
            ("HUMAN ASSISTANCE REQUESTED\n\n", "bold"),
            ("Context: ", "bold"),
            ty["context"],
            "\n",
            ("Question: ", "bold"),
            ty["question"],
        )

    @staticmethod
    def _build_req_relaxation(ty: RequirementRelaxationType) -> Text:
        return Text.assemble(
            ("REQUIREMENTS SKIP REQUEST\n\n", "bold"),
            "The agent would like to skip satisfying one of the requirements\n\n",
            ("Context: ", "bold"),
            ty["context"], "\n",
            ("Req #", "bold"),
            str(ty["req_number"]), ": ", ty["req_text"], "\n",
            ("Judgment: ", "bold"),
            ty["judgment"], "\n",
            ("Explanation: ", "bold"),
            ty["explanation"],
        )

    # ── Output (CodeGenIOHandler protocol) ────────────────────

    async def output(
        self,
        res: ResultStateSchema,
        mat: VFSAccessor[AIComposerState],
        st: AIComposerState
    ):
        await self._mounted.wait()
        target = self.query_one("#event-log", VerticalScroll)

        await self._mount_to(
            target,
            Static(Text("━━ CODE GENERATION COMPLETED ━━", style="bold green"))
        )

        # Build files dict for both paths
        files: dict[str, str] = {}
        for path in res.source:
            file_contents = mat.get(st, path)
            assert file_contents is not None
            files[path] = file_contents.decode("utf-8")

        if self._ide is not None:
            # Show files collapsed in TUI for reference
            for path, content in files.items():
                lexer = "cvl" if path.endswith(".spec") else "solidity"
                syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title=path, collapsed=True)
                await self._mount_to(target, coll)

            if res.comments:
                await self._mount_to(
                    target,
                    Static(Text.assemble(("\nComments: ", "bold"), res.comments))
                )

            # Preview in VS Code
            preview_id: str | None = None
            try:
                preview_id = await self._ide.preview_results(files)
            except Exception:
                self.notify("Failed to preview results in VS Code", severity="warning")

            if preview_id is not None:
                # Show inline ACCEPT/REJECT prompt
                prompt_widget = Static(Text.assemble(
                    ("Results previewed in VS Code.\n", "bold"),
                    ("Type ACCEPT to write files or REJECT to discard.", "dim"),
                ))
                hint_widget = Static("Response must be ACCEPT or REJECT", classes="interaction-hint")
                input_widget = Input(placeholder="ACCEPT / REJECT", validate_on=["submitted"])
                input_widget.validators = [Function(
                    lambda x: x.strip().upper() in ("ACCEPT", "REJECT"),
                    "Response must be ACCEPT or REJECT",
                )]
                await self._mount_to(target, prompt_widget, input_widget, hint_widget)
                input_widget.focus()

                response = await self._input_queue.get()
                await prompt_widget.remove()
                await input_widget.remove()
                await hint_widget.remove()
                decision = response.strip().upper()

                if decision == "ACCEPT":
                    try:
                        written = await self._ide.accept_results(preview_id)
                        await self._mount_to(
                            target,
                            Static(Text(f"Results accepted — wrote {len(written)} file(s).", style="bold green"))
                        )
                    except Exception:
                        self.notify("Failed to accept results in VS Code", severity="warning")
                else:
                    try:
                        await self._ide.reject_results(preview_id)
                    except Exception:
                        pass
                    await self._mount_to(
                        target,
                        Static(Text("Results rejected.", style="yellow"))
                    )
            else:
                await self._mount_to(
                    target,
                    Static(Text("Preview unavailable — results shown above.", style="dim"))
                )

            self._graph_done = True
            await self._mount_to(
                target,
                Static(Text("Press q to quit.", style="dim"))
            )
        else:
            # No IDE — current behavior: show files expanded
            for path, content in files.items():
                lexer = "cvl" if path.endswith(".spec") else "solidity"
                syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title=path, collapsed=False)
                await self._mount_to(target, coll)

            if res.comments:
                await self._mount_to(
                    target,
                    Static(Text.assemble(("\nComments: ", "bold"), res.comments))
                )

            self._graph_done = True
            await self._mount_to(
                target,
                Static(Text("Press q to quit.", style="dim"))
            )
