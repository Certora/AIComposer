from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.tool import ToolCall
from langchain_core.messages import AIMessage

from prompt_toolkit import PromptSession

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from composer.assistant.tools import (
    ProposeCodegen,
    ProposeNatSpec,
    ProposeResume,
    AskUser,
)
from composer.assistant.types import (
    CodegenWorkflowArgs,
    NatSpecWorkflowArgs,
    OrchestratorState,
)
from composer.io.codegen_rich import CodeGenRichApp
from composer.io.natspec_rich import NatSpecRichApp
from composer.io.events import AllEvents, StateUpdate, Start, End
from composer.io.ide_bridge import IDEBridge
from composer.input.files import upload_input
from composer.input.types import ResumeFSData
import composer.spec.natspec as natspec
from composer.workflow.executor import execute_ai_composer_workflow
from composer.workflow.factories import create_llm
from composer.audit.db import DEFAULT_CONNECTION as AUDIT_DEFAULT
from composer.rag.db import DEFAULT_CONNECTION as RAG_DEFAULT


class _UploadArgs:
    """Minimal object satisfying CommandLineArgs for upload_input."""

    def __init__(self, spec_file: str, interface_file: str, system_doc: str):
        self.spec_file = spec_file
        self.interface_file = interface_file
        self.system_doc = system_doc


class ThreadIdCapture(CodeGenRichApp):
    """Intercepts log_thread_id to capture the chosen thread_id."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.captured_thread_id: str | None = None

    async def log_thread_id(self, tid: str, chosen: bool):
        self.captured_thread_id = tid
        await super().log_thread_id(tid, chosen)


class OrchestratorHandler:
    """Rich console handler for the orchestrator agent.

    Provides two callbacks consumed by ``graph_runner.run_graph``:

    * ``on_event``   – ``SinkProtocol`` — renders events progressively.
    * ``on_interrupt`` – ``HumanHandler`` — dispatches questions vs proposals.
    """

    def __init__(
        self,
        console: Console,
        workspace: Path,
        ide: IDEBridge | None,
        llm: BaseChatModel,
        model_args: dict[str, Any],
    ):
        self._console = console
        self._workspace = workspace
        self._ide = ide
        self._llm = llm
        self._model_args = model_args

        self._session = PromptSession()

    async def _prompt(self, message: str = "> ", multiline: bool = False) -> str:
        """Prompt with readline-style editing. Alt+Enter submits in multiline mode."""
        return await self._session.prompt_async(message, multiline=multiline)

    # ------------------------------------------------------------------
    # SinkProtocol
    # ------------------------------------------------------------------

    def on_event(self, event: AllEvents) -> None:
        """Render orchestrator graph events to the console."""
        match event:
            case Start(description=desc):
                self._console.print(Rule(f"[bold]{desc}[/bold]"))
            case End():
                pass
            case StateUpdate(payload=payload):
                self._render_state_update(payload)
            case _:
                pass

    def _render_state_update(self, payload: dict) -> None:
        for node_name, node_data in payload.items():
            if node_name == "__interrupt__":
                continue
            messages = node_data.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    text = msg.text()
                    if text:
                        self._console.print(text)
                    for tc in msg.tool_calls:
                        self._render_tool_call(tc)

    def _render_tool_call(self, tc: ToolCall) -> None:
        name = tc.get("name", "")
        args = tc.get("args", {})
        match name:
            case "get_file":
                self._console.print(
                    f"  [dim]Reading {args.get('path', '?')}[/dim]"
                )
            case "list_files":
                self._console.print("  [dim]Listing project files[/dim]")
            case "grep_files":
                self._console.print(
                    f"  [dim]Searching for '{args.get('search_string', '?')}'[/dim]"
                )
            case "memory":
                cmd = args.get("command", "")
                path = args.get("path", "")
                self._console.print(
                    f"  [dim]Memory {cmd} {path}[/dim]"
                )
            case n if n.startswith("propose_") or n == "ask_user":
                pass  # handled in on_interrupt
            case "done":
                pass
            case _:
                self._console.print(f"  [dim]Tool: {name}[/dim]")

    # ------------------------------------------------------------------
    # HumanHandler
    # ------------------------------------------------------------------

    async def on_interrupt(self, payload: Any, state: OrchestratorState) -> str:
        """Dispatch interrupt based on payload type."""
        if isinstance(payload, AskUser):
            return await self._prompt_user(payload)
        elif isinstance(payload, (ProposeCodegen, ProposeNatSpec, ProposeResume)):
            return await self._handle_proposal(payload)
        else:
            return await self._prompt_user_fallback(payload)

    async def _prompt_user(self, payload: AskUser) -> str:
        self._console.print()
        self._console.print(Panel(
            f"[bold]Question:[/bold] {payload.question}\n\n"
            f"[dim]Context:[/dim] {payload.context}",
            title="Agent Question",
        ))
        return await self._prompt(multiline=True)

    async def _prompt_user_fallback(self, payload: Any) -> str:
        self._console.print(Panel(str(payload), title="Agent Message"))
        return await self._prompt(multiline=True)

    async def _handle_proposal(
        self, payload: ProposeCodegen | ProposeNatSpec | ProposeResume
    ) -> str:
        self._console.print()
        self._show_proposal(payload)

        choice = (await self._prompt("Accept (y), Reject (n), or provide corrections: ")).strip()
        if not choice or choice.lower() in ("y", "yes", "accept"):
            return await self._dispatch_workflow(payload)
        elif choice.lower() in ("n", "no", "reject"):
            return "Rejected. Please propose a different workflow or ask what I'd like to do."
        else:
            return f"User correction: {choice}"

    def _show_proposal(
        self, payload: ProposeCodegen | ProposeNatSpec | ProposeResume
    ) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        match payload:
            case ProposeCodegen():
                title = "Code Generation"
                table.add_row("Spec file", payload.spec_file)
                table.add_row("Interface", payload.interface_file)
                table.add_row("System doc", payload.system_doc)
            case ProposeNatSpec():
                title = "NatSpec Generation"
                table.add_row("Input file", payload.input_file)
            case ProposeResume():
                title = "Resume Workflow"
                table.add_row("Working dir", payload.working_dir)
                if payload.commentary:
                    table.add_row("Commentary", payload.commentary)

        table.add_row("Reason", payload.explanation)

        panel = Panel(
            table,
            title=f"Proposed: {title}",
        )
        self._console.print(panel)

    async def _dispatch_workflow(
        self, payload: ProposeCodegen | ProposeNatSpec | ProposeResume
    ) -> str:
        match payload:
            case ProposeCodegen():
                return await self._launch_codegen(payload)
            case ProposeNatSpec():
                return await self._launch_natspec(payload)
            case ProposeResume():
                return await self._launch_resume(payload)

    # ------------------------------------------------------------------
    # Sub-workflow launchers
    # ------------------------------------------------------------------

    def _make_codegen_args(self) -> CodegenWorkflowArgs:
        return CodegenWorkflowArgs(
            audit_db=AUDIT_DEFAULT,
            rag_db=RAG_DEFAULT,
            model=self._model_args.get("model", "claude-opus-4-6"),
            tokens=self._model_args.get("tokens", 10_000),
            thinking_tokens=self._model_args.get("thinking_tokens", 2048),
            memory_tool=self._model_args.get("memory_tool", True),
            recursion_limit=200
        )

    async def _launch_codegen(self, payload: ProposeCodegen) -> str:
        spec_path = str(self._workspace / payload.spec_file)
        intf_path = str(self._workspace / payload.interface_file)
        doc_path = str(self._workspace / payload.system_doc)

        self._console.print("[bold]Uploading input files...[/bold]")
        upload_args = _UploadArgs(
            spec_file=spec_path,
            interface_file=intf_path,
            system_doc=doc_path,
        )
        input_data = upload_input(upload_args)  # type: ignore[arg-type]

        workflow_args = self._make_codegen_args()
        llm = create_llm(workflow_args)

        app = ThreadIdCapture(ide=self._ide)
        captured_error: Exception | None = None

        async def work():
            nonlocal captured_error
            try:
                result = await execute_ai_composer_workflow(
                    handler=app,
                    llm=llm,
                    input=input_data,
                    workflow_options=workflow_args,
                )
                app.result_code = result
            except Exception as exc:
                app.result_code = 1
                captured_error = exc

        app.set_work(work)
        await app.run_async()

        tid = app.captured_thread_id or "unknown"
        code = getattr(app, "result_code", 1)
        if captured_error is not None:
            tb = "".join(traceback.format_exception(captured_error))
            return (
                f"Code generation crashed with {type(captured_error).__name__}: {captured_error}\n"
                f"Traceback:\n{tb}\n"
                f"Thread ID: {tid}."
            )
        if code == 0:
            return (
                f"Code generation completed successfully. "
                f"Thread ID: {tid}. "
                f"Save this to /memories/last_run.json for future resume."
            )
        else:
            return f"Code generation finished with exit code {code}. Thread ID: {tid}."

    async def _launch_resume(self, payload: ProposeResume) -> str:
        working_dir = str(self._workspace / payload.working_dir)

        input_data = ResumeFSData(
            thread_id=payload.thread_id,
            file_path=working_dir,
            comments=payload.commentary or None,
            new_system=None,
        )

        workflow_args = self._make_codegen_args()
        llm = create_llm(workflow_args)

        app = ThreadIdCapture(ide=self._ide)
        captured_error: Exception | None = None

        async def work():
            nonlocal captured_error
            try:
                result = await execute_ai_composer_workflow(
                    handler=app,
                    llm=llm,
                    input=input_data,
                    workflow_options=workflow_args,
                )
                app.result_code = result
            except Exception as exc:
                app.result_code = 1
                captured_error = exc

        app.set_work(work)
        await app.run_async()

        tid = app.captured_thread_id or payload.thread_id
        code = getattr(app, "result_code", 1)
        if captured_error is not None:
            tb = "".join(traceback.format_exception(captured_error))
            return (
                f"Resume crashed with {type(captured_error).__name__}: {captured_error}\n"
                f"Traceback:\n{tb}\n"
                f"Thread ID: {tid}."
            )
        if code == 0:
            return (
                f"Resume completed successfully. "
                f"Thread ID: {tid}. "
                f"Save this to /memories/last_run.json for future resume."
            )
        else:
            return f"Resume finished with exit code {code}. Thread ID: {tid}."

    async def _launch_natspec(self, payload: ProposeNatSpec) -> str:
        input_path = str(self._workspace / payload.input_file)

        args = NatSpecWorkflowArgs(
            input_file=input_path,
            rag_db=RAG_DEFAULT,
            model=self._model_args.get("model", "claude-opus-4-6"),
            tokens=self._model_args.get("tokens", 10_000),
            thinking_tokens=self._model_args.get("thinking_tokens", 2048),
            memory_tool=self._model_args.get("memory_tool", True),
            recursion_limit=150
        )

        app = NatSpecRichApp(ide=self._ide)
        captured_error: Exception | None = None

        async def work():
            nonlocal captured_error
            try:
                result = await natspec.execute(args, handler=app)  # type: ignore[arg-type]
                app.result_code = result
            except Exception as exc:
                app.result_code = 1
                captured_error = exc

        app.set_work(work)
        await app.run_async()

        if captured_error is not None:
            tb = "".join(traceback.format_exception(captured_error))
            return f"NatSpec generation crashed with {type(captured_error).__name__}: {captured_error}\nTraceback:\n{tb}"
        code = getattr(app, "result_code", 1)
        if code == 0:
            return "NatSpec generation completed successfully. Spec and interface files have been generated."
        else:
            return f"NatSpec generation finished with exit code {code}."
