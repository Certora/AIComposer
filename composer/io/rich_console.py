import asyncio
from collections.abc import Coroutine
from typing import Callable

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll, Container
from textual.widgets import Static, Input, Collapsible, DataTable
from textual.widgets.data_table import RowKey, ColumnKey
from textual.message import Message
from textual import on
from textual.binding import Binding
from textual.validation import Function, Validator

from rich.syntax import Syntax
from rich.text import Text

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

class RichConsoleApp(App):
    CSS = """
    #status-bar { dock: top; height: 1; background: $surface; padding: 0 1; }
    #event-log { height: 1fr; padding: 0 1; }
    #interaction-panel { dock: bottom; height: auto; max-height: 50%; border-top: double $accent; padding: 1; display: none; }
    #interaction-prompt { height: auto; max-height: 15; margin-bottom: 1; }
    #user-input { width: 1fr; }
    #input-hint { color: $text-muted; padding: 0 1; }
    .nested-workflow { margin-left: 2; border-left: solid $secondary; padding-left: 1; }
    .vfs-change { color: cyan; }
    .token-stats { color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(self):
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

    def compose(self) -> ComposeResult:
        yield Static("Session: — | Checkpoint: —", id="status-bar")
        yield VerticalScroll(id="event-log")
        with Container(id="interaction-panel"):
            yield Static(id="interaction-prompt")
            yield Input(placeholder="Type here...", id="user-input", validate_on=["submitted"])
            yield Static("", id="input-hint")

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

    def _render_ai_turn(self, msg: AIMessage) -> Collapsible:
        parts: list[str] = []
        children: list[Static] = []

        for c in normalize_content(msg.content):
            match c["type"]:
                case "thinking":
                    parts.append("Thinking...")
                    children.append(Static(Text("Thinking...", style="dim")))
                case "text":
                    preview = c["text"][:80].replace("\n", " ")
                    parts.append(f"Text: {preview}...")
                    children.append(Static(c["text"]))
                case "tool_use":
                    name = c["name"]
                    parts.append(f"Call tool: {name}")
                    children.append(Static(Text(f"Call tool: {name}", style="bold cyan")))
                case other:
                    parts.append(f"Unknown: {other}")
                    children.append(Static(f"Unknown block: {other}"))

        if isinstance(msg.response_metadata, dict) and "usage" in msg.response_metadata:
            usage = msg.response_metadata["usage"]
            stats = (
                f"cache_read: {usage.get('cache_read_input_tokens', 0)} | "
                f"input: {usage.get('input_tokens', 0)} | "
                f"cache_write: {usage.get('cache_creation', {}).get('ephemeral_5m_input_tokens', 0)}"
            )
            children.append(Static(Text(stats, style="dim"), classes="token-stats"))

        title = "AI: " + " | ".join(parts) if parts else "AI turn"
        coll = Collapsible(*children, title=title, collapsed=True)
        return coll

    # ── IOHandler protocol ────────────────────────────────────

    def _update_status_bar(self):
        bar = self.query_one("#status-bar", Static)
        bar.update(f"Session: {self._session_id} | Checkpoint: {self._checkpoint_id}")

    async def log_thread_id(self, tid: str, chosen: bool):
        await self._mounted.wait()
        self._session_id = tid
        self._update_status_bar()

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str):
        await self._mounted.wait()
        self._checkpoint_id = checkpoint_id
        self._update_status_bar()
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
            coll = Collapsible(inner, title=f"Nested: {label}", collapsed=False)
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
                            widget = self._render_ai_turn(m)
                            await self._mount_to(target, widget)
                        case SystemMessage():
                            content = m.content if isinstance(m.content, str) else str(m.content)
                            coll = Collapsible(Static(content), title="System prompt", collapsed=True)
                            await self._mount_to(target, coll)
                        case HumanMessage():
                            content = m.content if isinstance(m.content, str) else str(m.content)
                            coll = Collapsible(Static(content), title="Initial prompt", collapsed=True)
                            await self._mount_to(target, coll)
                        case ToolMessage():
                            content = m.content if isinstance(m.content, str) else str(m.content)
                            name = getattr(m, "name", None) or "Tool result"
                            coll = Collapsible(Static(content), title=name, collapsed=True)
                            await self._mount_to(target, coll)
                        case _:
                            await self._mount_to(target, Static(Text(f"[Message: {type(m).__name__}]", style="dim")))

            if "vfs" in v:
                files = ", ".join(v["vfs"].keys())
                await self._mount_to(
                    target,
                    Static(Text(f"VFS write: {files}", style="cyan"), classes="vfs-change")
                )

    async def progress_update(self, path: list[str], upd: ProgressUpdate):
        await self._mounted.wait()
        target = self._get_mount_target(path)

        match upd["type"]:
            case "prover_run":
                args_str = " ".join(upd["args"])
                await self._mount_to(
                    target,
                    Static(Text(f"▶ Prover run: {args_str}", style="bold yellow"))
                )
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

    class _InteractionRequest(Message):
        def __init__(self, ty: HumanInteractionType):
            super().__init__()
            self.ty = ty

    async def human_interaction(
        self,
        ty: HumanInteractionType,
        debug_thunk: Callable[[], None]
    ) -> str:
        await self._mounted.wait()
        # Push the request into the app's message loop — the handler
        # runs in the app's own context where focus / layout work.
        self.post_message(self._InteractionRequest(ty))
        response = await self._input_queue.get()

        # Log the interaction to the event log
        target = self.query_one("#event-log", VerticalScroll)
        await self._mount_to(
            target,
            Static(Text(f"[Human response: {response[:60]}...]", style="dim"))
        )

        return response

    @on(_InteractionRequest)
    def _on_interaction_request(self, event: _InteractionRequest) -> None:
        panel = self.query_one("#interaction-panel", Container)
        prompt_widget = self.query_one("#interaction-prompt", Static)
        hint_widget = self.query_one("#input-hint", Static)
        input_widget = self.query_one("#user-input", Input)

        match event.ty["type"]:
            case "proposal":
                validators = self._setup_proposal(event.ty, prompt_widget, hint_widget)
            case "question":
                validators = self._setup_question(event.ty, prompt_widget, hint_widget)
            case "extraction_question":
                validators = self._setup_extraction_question(event.ty, prompt_widget, hint_widget)
            case "req_relaxation":
                validators = self._setup_req_relaxation(event.ty, prompt_widget, hint_widget)
            case _:
                validators = []

        input_widget.validators = validators
        input_widget.value = ""
        panel.styles.display = "block"
        input_widget.focus()

    def _setup_proposal(self, ty: ProposalType, prompt: Static, hint: Static) -> list[Validator]:
        prompt.update(
            Text.assemble(
                ("SPEC CHANGE PROPOSAL\n\n", "bold"),
                ("Explanation: ", "bold"),
                ty["explanation"],
                "\n\n",
                ("Note: Full diff view available in VS Code extension.", "dim italic"),
            )
        )
        hint.update("Response must start with ACCEPTED, REJECTED, or REFINE")
        return [Function(
            lambda x: x.startswith("ACCEPTED") or x.startswith("REJECTED") or x.startswith("REFINE"),
            "Response must begin with ACCEPTED/REJECTED/REFINE",
        )]

    def _setup_question(self, ty: QuestionType, prompt: Static, hint: Static) -> list[Validator]:
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
        prompt.update(Text.assemble(*parts))
        hint.update("Begin response with FOLLOWUP to request clarification")
        return []

    def _setup_extraction_question(self, ty: ExtractionQuestionType, prompt: Static, hint: Static) -> list[Validator]:
        prompt.update(
            Text.assemble(
                ("HUMAN ASSISTANCE REQUESTED\n\n", "bold"),
                ("Context: ", "bold"),
                ty["context"],
                "\n",
                ("Question: ", "bold"),
                ty["question"],
            )
        )
        hint.update("Enter your response")
        return []

    def _setup_req_relaxation(self, ty: RequirementRelaxationType, prompt: Static, hint: Static) -> list[Validator]:
        prompt.update(
            Text.assemble(
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
        )
        hint.update("Response must start with ACCEPTED or REJECTED")
        return [Function(
            lambda r: r.startswith("ACCEPTED") or r.startswith("REJECTED"),
            "Response must begin with ACCEPTED/REJECTED",
        )]

    def on_input_submitted(self, event: Input.Submitted):
        value = event.value.strip()
        if not value:
            return

        if event.validation_result and not event.validation_result.is_valid:
            for desc in event.validation_result.failure_descriptions:
                self.notify(desc, severity="error")
            return

        self.query_one("#user-input", Input).value = ""
        self.query_one("#interaction-panel", Container).styles.display = "none"
        self._input_queue.put_nowait(value)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        if event.coordinate.column != 2:
            return
        # Find which rule this row corresponds to
        for rule_name, row_key in self._rule_row_keys.items():
            if row_key == event.cell_key.row_key:
                if rule_name in self._rule_analyses:
                    self.notify("VS Code integration pending")
                return

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

        for path in res.source:
            file_contents = mat.get(st, path)
            assert file_contents is not None
            content = file_contents.decode("utf-8")

            # Guess lexer from extension
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
