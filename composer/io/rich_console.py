import asyncio
import difflib
from collections.abc import Coroutine
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Container
from textual.widgets import Static, Input, Collapsible, DataTable
from textual.widgets.data_table import RowKey, ColumnKey
from textual.binding import Binding
from textual.validation import Function, Validator

from rich.syntax import Syntax
from rich.text import Text

from composer.io.ide_bridge import IDEBridge

from graphcore.tools.vfs import VFSAccessor
from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from composer.diagnostics.handlers import normalize_content
from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import (
    HumanInteractionType, ProposalType, QuestionType,
    RequirementRelaxationType, ExtractionQuestionType,
)
from composer.core.state import ResultStateSchema, AIComposerState
from composer.prover.ptypes import StatusCodes

_KNOWN_NODES = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

_STATUS_STYLES: dict[StatusCodes, str] = {
    "VERIFIED": "green",
    "VIOLATED": "bold red",
    "TIMEOUT": "yellow",
    "ERROR": "red",
    "SANITY_FAILED": "magenta",
}

# ── Friendly tool display names ─────────────────────────────

_TOOL_DISPLAY: dict[str, str] = {
    "certora_prover": "Running prover",
    "put_file": "Writing files",
    "get_file": "Reading file",
    "list_files": "Listing files",
    "grep_files": "Searching files",
    "propose_spec_change": "Proposing spec change",
    "human_in_the_loop": "Asking for input",
    "code_result": "Finalizing result",
    "cvl_manual_search": "Searching CVL manual",
    "requirements_evaluation": "Evaluating requirements",
    "requirement_relaxation_request": "Requesting requirement relaxation",
    "memory": "Accessing memory",
}

_TOOL_RESULT_DISPLAY: dict[str, str] = {
    "certora_prover": "Prover results",
    "put_file": "File write result",
    "get_file": "File contents",
    "list_files": "File listing",
    "grep_files": "Search results",
    "cvl_manual_search": "Manual search results",
    "requirements_evaluation": "Requirements evaluation",
    "requirement_relaxation_request": "Relaxation result",
    "propose_spec_change": "Spec change result",
    "human_in_the_loop": "Human response",
    "code_result": "Final result",
    "memory": "Memory result",
}

_TOOL_COLLAPSE_GROUP: dict[str, str] = {
    "get_file": "read",
    "put_file": "write",
    "memory": "memory",
}

_SUPPRESS_TOOL_RESULT: set[str] = {
    "human_in_the_loop",
    "propose_spec_change",
    "requirement_relaxation_request",
    "certora_prover",
}


_DOT = "\u25cf "  # ● filled circle


def _dot(style: str, text: Text | str) -> Text:
    """Prepend a colored dot to a Text or string for visual structure."""
    if isinstance(text, str):
        text = Text(text)
    result = Text()
    result.append(_DOT, style=style)
    result.append_text(text)
    return result


def _friendly_tool_call(name: str, input: dict) -> str:
    base = _TOOL_DISPLAY.get(name, f"Tool: {name}")
    match name:
        case "certora_prover":
            target = input.get("target_contract", "")
            rule = input.get("rule")
            detail = f": {target}" + (f" — rule {rule}" if rule else "")
            return base + detail
        case "get_file":
            return f"{base}: {input.get('path', '?')}"
        case "grep_files":
            return f"{base} for: {input.get('search_string', '?')}"
        case "cvl_manual_search":
            q = input.get("question", "?")[:60]
            return f"{base}: {q}"
        case "put_file":
            files = input.get("files", {})
            return f"{base}: {', '.join(files.keys())}"
        case "memory":
            cmd = input.get("command", "?")
            path = input.get("path", "")
            return f"{base}: {cmd} {path}".strip()
        case "human_in_the_loop":
            q = input.get("question", "")
            return f"{base}: {q}" if q else base
        case "propose_spec_change":
            expl = input.get("explanation", "")
            return f"{base}: {expl}" if expl else base
        case "requirement_relaxation_request":
            num = input.get("req_number", "?")
            req = input.get("req_text", "")
            return f"{base} #{num}: {req}" if req else base
        case _:
            return base


def _collapse_detail(name: str, input: dict) -> str:
    """Extract the detail item for collapsing (e.g. file path)."""
    match name:
        case "get_file":
            return input.get("path", "?")
        case "put_file":
            files = input.get("files", {})
            return ", ".join(files.keys())
        case _:
            return ""


class RichConsoleApp(App):
    CSS = """
    #status-bar { dock: top; height: 1; background: $surface; padding: 0 1; }
    #event-log { height: 1fr; padding: 0 1; }
    #event-log > * { margin-bottom: 1; }
    .interaction-hint { color: $text-muted; padding: 0 1; }
    .nested-workflow { margin-left: 2; border-left: solid $secondary; padding-left: 1; }
    .vfs-change { color: cyan; }
    Collapsible { background: transparent; border: none; padding: 0; }
    CollapsibleTitle { padding: 0 1; }
    Collapsible Contents { padding: 0 0 0 3; }
    """

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(self, show_checkpoints: bool = False, ide: IDEBridge | None = None):
        super().__init__()
        self._input_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._mounted: asyncio.Event = asyncio.Event()
        self._nested_containers: dict[str, VerticalScroll] = {}
        self._graph_done = False
        self.result_code = 1
        self._session_id = "—"
        self._checkpoint_id = "—"
        self._prover_table: DataTable | None = None
        self._analysis_col: ColumnKey | None = None
        self._rule_row_keys: dict[str, RowKey] = {}
        self._rule_analyses: dict[str, str] = {}
        self._work_fn: Callable[[], Coroutine[None, None, None]] | None = None
        self._show_checkpoints = show_checkpoints
        self._ide: IDEBridge | None = ide
        self._vfs_snapshots: dict[int, tuple[list[str], dict[str, str]]] = {}
        self._next_snap_id: int = 0

        # Token stats accumulators
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_cache_read: int = 0

        # Consecutive tool call collapsing state
        self._last_tool_group: str | None = None
        self._last_tool_items: list[str] = []
        self._last_tool_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static("Session: — | Checkpoint: —", id="status-bar")
        yield VerticalScroll(id="event-log")

    def set_work(self, fn: Callable[[], Coroutine[None, None, None]]):
        self._work_fn = fn

    def on_mount(self):
        self._mounted.set()
        if self._work_fn is not None:
            self.run_worker(self._work_fn(), thread=False)

    # ── Key bindings ──────────────────────────────────────────

    def action_quit_app(self):
        if self._graph_done:
            self.exit()

    # ── Helpers ───────────────────────────────────────────────

    def _get_mount_target(self, path: list[str]) -> VerticalScroll:
        if len(path) > 1 and path[-1] in self._nested_containers:
            return self._nested_containers[path[-1]]
        return self.query_one("#event-log", VerticalScroll)

    async def _auto_scroll(self):
        log = self.query_one("#event-log", VerticalScroll)
        if log.max_scroll_y - log.scroll_y <= 3:
            log.scroll_end(animate=False)

    async def _mount_to(self, target: VerticalScroll, *widgets):
        await target.mount_all(widgets)
        await self._auto_scroll()

    def _reset_tool_collapsing(self):
        """Reset consecutive tool call collapsing state."""
        self._last_tool_group = None
        self._last_tool_items = []
        self._last_tool_widget = None

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

    async def _ide_show_file(self, content: str, path: str, lang: str | None) -> None:
        try:
            assert self._ide is not None
            await self._ide.show_file(content, path, lang=lang)
        except Exception as exc:
            self.notify(f"Failed to open {path} (lang={lang}): {exc}", severity="warning")

    @staticmethod
    def _guess_lang(path: str) -> str | None:
        if path.endswith(".sol"):
            return "solidity"
        elif path.endswith(".json"):
            return "json"
        return None

    def _render_collapsed_text(self, group: str, items: list[str]) -> str:
        """Build the display text for a collapsed group of tool calls."""
        match group:
            case "read":
                return f"Reading: {', '.join(items)}"
            case "write":
                return f"Wrote: {', '.join(items)}"
            case "memory":
                count = len(items)
                if count == 1:
                    return "Accessing memory"
                return f"Accessing memory (×{count})"
            case _:
                return f"Tools: {', '.join(items)}"

    def _render_ai_turn(self, msg: AIMessage) -> list[Static | Collapsible]:
        """Render an AI turn as a list of widgets (not a single collapsible)."""
        widgets: list[Static | Collapsible] = []

        for c in normalize_content(msg.content):
            match c["type"]:
                case "thinking":
                    full_text = c.get("thinking", "")
                    widgets.append(
                        Collapsible(Static(full_text), title="Thinking...", collapsed=True)
                    )
                case "text":
                    text = c["text"]
                    if text.strip():
                        widgets.append(Static(_dot("blue", text)))
                case "tool_use":
                    name = c["name"]
                    input_args = c.get("input", {})
                    group = _TOOL_COLLAPSE_GROUP.get(name)

                    if group is not None and group == self._last_tool_group:
                        # Same group — update existing widget
                        detail = _collapse_detail(name, input_args)
                        if detail:
                            self._last_tool_items.append(detail)
                        else:
                            self._last_tool_items.append("")
                        new_text = self._render_collapsed_text(group, self._last_tool_items)
                        if self._last_tool_widget is not None:
                            self._last_tool_widget.update(_dot("green", Text(new_text, style="dim")))
                        # Don't append a new widget
                    elif group is not None:
                        # New collapsible group
                        detail = _collapse_detail(name, input_args)
                        self._last_tool_group = group
                        self._last_tool_items = [detail] if detail else [""]
                        display_text = self._render_collapsed_text(group, self._last_tool_items)
                        w = Static(_dot("green", Text(display_text, style="dim")))
                        self._last_tool_widget = w
                        widgets.append(w)
                    else:
                        # Non-collapsible tool — reset and emit standalone
                        self._reset_tool_collapsing()
                        friendly = _friendly_tool_call(name, input_args)
                        widgets.append(Static(_dot("green", Text(friendly, style="dim"))))
                case other:
                    widgets.append(Static(f"Unknown block: {other}"))

        # Accumulate token stats
        if isinstance(msg.response_metadata, dict) and "usage" in msg.response_metadata:
            usage = msg.response_metadata["usage"]
            self._total_input += usage.get("input_tokens", 0)
            self._total_output += usage.get("output_tokens", 0)
            self._total_cache_read += usage.get("cache_read_input_tokens", 0)
            self._update_status_bar()

        return widgets

    # ── IOHandler protocol ────────────────────────────────────

    def _update_status_bar(self):
        bar = self.query_one("#status-bar", Static)
        parts = [
            f"Session: {self._session_id}",
            f"Checkpoint: {self._checkpoint_id}",
        ]
        if self._total_input > 0 or self._total_output > 0:
            parts.append(f"in:{self._total_input} out:{self._total_output} cache:{self._total_cache_read}")
        bar.update(" | ".join(parts))

    async def log_thread_id(self, tid: str, chosen: bool):
        await self._mounted.wait()
        self._session_id = tid
        self._update_status_bar()

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str):
        await self._mounted.wait()
        self._checkpoint_id = checkpoint_id
        self._update_status_bar()
        if self._show_checkpoints:
            target = self._get_mount_target(path)
            await self._mount_to(
                target,
                Static(Text(f"checkpoint: {checkpoint_id}", style="dim"))
            )

    async def log_start(self, *, path: list[str], tool_id: str | None):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        if len(path) == 1:
            banner = Static(
                Text(f"━━ Workflow start: {path[0]} ━━", style="bold"),
            )
            await self._mount_to(target, banner)
        else:
            # Nested workflow: create a collapsible with an inner container
            tool_info = f" (tool={tool_id})" if tool_id else ""
            label = " > ".join(path) + tool_info
            inner = VerticalScroll(classes="nested-workflow")
            coll = Collapsible(inner, title=f"Nested: {label}", collapsed=True)
            self._nested_containers[path[-1]] = inner
            await self._mount_to(target, coll)

    async def log_end(self, path: list[str]):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        if len(path) == 1:
            banner = Static(
                Text(f"━━ Workflow end: {path[0]} ━━", style="bold"),
            )
            await self._mount_to(target, banner)
        else:
            # Collapse the nested workflow
            tid = path[-1]
            if tid in self._nested_containers:
                container = self._nested_containers.pop(tid)
                parent_coll = container.parent
                if isinstance(parent_coll, Collapsible):
                    parent_coll.collapsed = True

    async def log_state_update(self, path: list[str], st: dict):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        for node_name, v in st.items():
            if node_name not in _KNOWN_NODES:
                continue

            if "messages" in v:
                for m in v["messages"]:
                    match m:
                        case AIMessage():
                            widgets = self._render_ai_turn(m)
                            if widgets:
                                await self._mount_to(target, *widgets)
                        case SystemMessage():
                            self._reset_tool_collapsing()
                            coll = Collapsible(Static(m.text()), title="System prompt", collapsed=True)
                            await self._mount_to(target, coll)
                        case HumanMessage():
                            self._reset_tool_collapsing()
                            content = m.text()
                            # Injected prover violation summary (not from the human)
                            if "prover output was too large" in content:
                                title = "Prover violation summary"
                                collapsed = False
                            else:
                                title = "Initial prompt"
                                collapsed = True
                            coll = Collapsible(Static(content), title=title, collapsed=collapsed)
                            await self._mount_to(target, coll)
                        case ToolMessage():
                            name = getattr(m, "name", None) or "Tool result"
                            # Suppress results for collapsed tools and redundant human responses
                            if name in _TOOL_COLLAPSE_GROUP or name in _SUPPRESS_TOOL_RESULT:
                                continue
                            self._reset_tool_collapsing()
                            friendly = _TOOL_RESULT_DISPLAY.get(name, name)
                            coll = Collapsible(Static(m.text()), title=friendly, collapsed=True)
                            await self._mount_to(target, coll)
                        case _:
                            self._reset_tool_collapsing()
                            await self._mount_to(target, Static(Text(f"[Message: {type(m).__name__}]", style="dim")))

            if "vfs" in v:
                self._reset_tool_collapsing()
                count = len(v["vfs"])
                names = list(v["vfs"].keys())
                contents = {
                    k: val.decode("utf-8") if isinstance(val, bytes) else val
                    for k, val in v["vfs"].items()
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

    async def progress_update(self, path: list[str], upd: ProgressUpdate):
        await self._mounted.wait()
        target = self._get_mount_target(path)

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

    async def human_interaction(
        self,
        ty: HumanInteractionType,
        debug_thunk: Callable[[], None]
    ) -> str:
        await self._mounted.wait()
        target = self.query_one("#event-log", VerticalScroll)

        # Mount directly from worker — post_message races with state update mounts
        prompt_content, hint_text, validators = self._build_interaction(ty)

        prompt_widget = Static(prompt_content)
        hint_widget = Static(hint_text, classes="interaction-hint")
        input_widget = Input(placeholder="Type here...", validate_on=["submitted"])
        input_widget.validators = validators

        await self._mount_to(target, prompt_widget, input_widget, hint_widget)
        input_widget.focus()

        response = await self._input_queue.get()

        # Replace interaction widgets with compact summary
        await prompt_widget.remove()
        await input_widget.remove()
        await hint_widget.remove()
        await self._mount_to(
            target,
            Static(_dot("green", Text.assemble(("You: ", "bold green"), response)))
        )

        return response

    def _build_interaction(self, ty: HumanInteractionType) -> tuple[Text, str, list[Validator]]:
        """Return (prompt_renderable, hint_text, validators) for the interaction type."""
        _PROPOSAL_VALIDATOR : list[Validator] = [Function(
            lambda x: x.startswith("ACCEPTED") or x.startswith("REJECTED") or x.startswith("REFINE"),
            "Response must begin with ACCEPTED/REJECTED/REFINE",
        )]
        _REQ_VALIDATOR : list[Validator] = [Function(
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

    def on_input_submitted(self, event: Input.Submitted):
        value = event.value.strip()
        if not value:
            return

        if event.validation_result and not event.validation_result.is_valid:
            for desc in event.validation_result.failure_descriptions:
                self.notify(desc, severity="error")
            return

        # Disable input to prevent double-submit
        event.input.disabled = True
        self._input_queue.put_nowait(value)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        if event.coordinate.column != 2:
            return
        # Find which rule this row corresponds to
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
