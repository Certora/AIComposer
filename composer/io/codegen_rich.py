import difflib

from textual.containers import VerticalScroll
from textual.widgets import Static, Input, Collapsible, DataTable
from textual.widgets.data_table import RowKey, ColumnKey
from textual.validation import Function, Validator

from rich.syntax import Syntax
from rich.text import Text

from composer.io.ide_bridge import IDEBridge
from composer.io.tool_display import CodeGenToolDisplay
from composer.io.rich_console import BaseRichConsoleApp
from composer.io.message_renderer import _DOT

from langchain_core.messages import HumanMessage

from graphcore.tools.vfs import VFSAccessor

from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import (
    HumanInteractionType, ProposalType, QuestionType,
    RequirementRelaxationType, ExtractionQuestionType,
)
from composer.core.state import ResultStateSchema, AIComposerState
from composer.prover.ptypes import StatusCodes

_STATUS_STYLES: dict[StatusCodes, str] = {
    "VERIFIED": "green",
    "VIOLATED": "bold red",
    "TIMEOUT": "yellow",
    "ERROR": "red",
    "SANITY_FAILED": "magenta",
}


class CodeGenRichApp(BaseRichConsoleApp[HumanInteractionType, ProgressUpdate]):
    """Textual TUI for the code generation workflow."""

    def __init__(self, show_checkpoints: bool = False, ide: IDEBridge | None = None):
        super().__init__(
            tool_config=CodeGenToolDisplay(),
            show_checkpoints=show_checkpoints,
            ide=ide,
        )
        self._prover_table: DataTable | None = None
        self._analysis_col: ColumnKey | None = None
        self._rule_row_keys: dict[str, RowKey] = {}
        self._rule_analyses: dict[str, str] = {}
        self._vfs_snapshots: dict[int, tuple[list[str], dict[str, str]]] = {}
        self._next_snap_id: int = 0

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
                pass  # tool call already shows target contract + rule
            case "prover_result":
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

        if self._ide is not None:
            snap_id = self._next_snap_id
            self._next_snap_id += 1
            self._vfs_snapshots[snap_id] = (names, contents)
            links = []
            for idx, name in enumerate(names):
                escaped = name.replace("[", "\\[")
                links.append(
                    f"[@click=app.show_vfs_file({snap_id}, {idx})]"
                    f"[bold underline cyan]{escaped}[/bold underline cyan][/]"
                )
            markup = f"[cyan]{_DOT}[/cyan]Wrote {count} file{'s' if count != 1 else ''}: " + ", ".join(links)
            widget = Static(markup, classes="vfs-change")
        else:
            file_parts: list[tuple[str, str] | str] = [(_DOT, "cyan"), f"Wrote {count} file{'s' if count != 1 else ''}: "]
            for i, name in enumerate(names):
                if i > 0:
                    file_parts.append(", ")
                file_parts.append((name, "bold underline cyan"))
            widget = Static(Text.assemble(*file_parts), classes="vfs-change")
        await self._mount_to(target, widget)

    _HUMAN_TAG_DISPLAY: dict[str, tuple[str, bool]] = {
        "initial_prompt": ("Initial prompt", True),
        "resume": ("Resume context", True),
        "summarization": ("Summarization", True),
        "scolding": ("System correction", True),
        "prover_summary": ("Prover violation summary", False),
    }

    def classify_human_message(self, m: HumanMessage) -> tuple[str, bool]:
        tag = getattr(m, "display_tag", None)
        if tag is not None:
            return self._HUMAN_TAG_DISPLAY.get(tag, ("User input", True))
        # Fallback for untagged messages (e.g., older checkpoints)
        content = m.text()
        if "prover output was too large" in content:
            return ("Prover violation summary", False)
        return ("Initial prompt", True)

    # ── VFS file actions ──────────────────────────────────────

    def action_show_vfs_file(self, snap_id: int, file_idx: int) -> None:
        snap = self._vfs_snapshots.get(snap_id)
        if snap is None or self._ide is None:
            return
        names, contents = snap
        if file_idx < 0 or file_idx >= len(names):
            return
        path = names[file_idx]
        content = contents[path]
        lang = self._guess_lang(path)
        self.run_worker(self._ide_show_file(content, path, lang), thread=False)

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
                        self.notify(text[:200] + "...", title=f"Analysis: {rule_name}", timeout=10)
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
