from __future__ import annotations

from typing import Any

import questionary
from questionary import Choice, Style

from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall

from prompt_toolkit import PromptSession

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from composer.assistant.launch_args import (
    LaunchCodegenArgs,
    LaunchNatSpecArgs,
    LaunchResumeArgs,
)
from composer.assistant.types import (
    ConversationTurn,
    ConfirmLaunch,
    OrchestratorState,
)
from composer.io.events import AllEvents, StateUpdate, Start, End


_CONFIRM_STYLE = Style([
    ("highlighted", "fg:#61afef bold"),
    ("pointer", "fg:#61afef bold"),
    ("question", ""),
])


class OrchestratorHandler:
    """Rich console handler for the orchestrator agent.

    Provides two callbacks consumed by ``graph_runner.run_graph``:

    * ``on_event``   – ``SinkProtocol`` — renders events progressively.
    * ``on_interrupt`` – ``HumanHandler`` — dispatches conversation vs confirmation.
    """

    def __init__(self, console: Console):
        self._console = console
        self._session = PromptSession()

    async def _prompt(self, message: str = "> ", multiline: bool = False) -> str:
        return await self._session.prompt_async(message, multiline=multiline)

    # ------------------------------------------------------------------
    # SinkProtocol
    # ------------------------------------------------------------------

    def on_event(self, event: AllEvents) -> None:
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
                    text = msg.text
                    if text:
                        self._console.print(text)
                    for tc in msg.tool_calls:
                        self._render_tool_call(tc)

    def _render_tool_call(self, tc: ToolCall) -> None:
        name = tc.get("name", "")
        args = tc.get("args", {})
        match name:
            case "get_file":
                self._console.print(f"  [dim]Reading {args.get('path', '?')}[/dim]")
            case "list_files":
                self._console.print("  [dim]Listing project files[/dim]")
            case "grep_files":
                self._console.print(f"  [dim]Searching for '{args.get('search_string', '?')}'[/dim]")
            case "memory":
                cmd = args.get("command", "")
                path = args.get("path", "")
                self._console.print(f"  [dim]Memory {cmd} {path}[/dim]")
            case "launch_codegen" | "launch_resume" | "launch_natspec":
                self._console.print(f"  [dim]Launching {name.removeprefix('launch_')}...[/dim]")
            case "result":
                pass
            case _:
                self._console.print(f"  [dim]Tool: {name}[/dim]")

    # ------------------------------------------------------------------
    # HumanHandler
    # ------------------------------------------------------------------

    async def on_interrupt(self, payload: Any, state: OrchestratorState) -> str:
        match payload:
            case ConversationTurn():
                # AI message already rendered by on_event
                return await self._prompt()
            case LaunchCodegenArgs() | LaunchNatSpecArgs() | LaunchResumeArgs():
                return await self._confirm_launch(payload)
            case _:
                self._console.print(Panel(str(payload), title="Agent Message"))
                return await self._prompt(multiline=True)

    async def _confirm_launch(self, payload: ConfirmLaunch) -> str:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        match payload:
            case LaunchNatSpecArgs(input_file=f, contract_name=c, solc_version=v,
                                   cache_namespace=cache, memory_namespace=mem):
                title = "NatSpec Pipeline"
                table.add_row("Input file", f)
                table.add_row("Contract", c)
                table.add_row("Solc version", v)
                if cache:
                    table.add_row("Cache NS", cache)
                if mem:
                    table.add_row("Memory NS", mem)
            case LaunchCodegenArgs(spec_file=s, interface_file=i, system_doc=d):
                title = "Code Generation"
                table.add_row("Spec file", s)
                table.add_row("Interface", i)
                table.add_row("System doc", d)
            case LaunchResumeArgs(thread_id=t, working_dir=w, commentary=c):
                title = "Resume Workflow"
                table.add_row("Thread ID", t)
                table.add_row("Working dir", w)
                if c:
                    table.add_row("Commentary", c)

        self._console.print(Panel(table, title=f"Launch: {title}"))

        choice = await questionary.select(
            "",
            choices=[
                Choice("Yes, proceed", value="yes"),
                Choice("No", value="no"),
                Choice("No, with feedback", value="feedback"),
            ],
            qmark="",
            pointer="›",
            style=_CONFIRM_STYLE,
            instruction="",
        ).ask_async()

        match choice:
            case "yes":
                return "yes"
            case "no":
                return "no"
            case "feedback":
                return await self._prompt("Feedback: ")
            case _:
                return "no"
