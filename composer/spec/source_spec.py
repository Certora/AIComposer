"""
Source-based spec generation.

Generate CVL specs for existing smart contracts using PreAudit for
compilation analysis and verification.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import tempfile
import sqlite3
import sys
import subprocess

import composer.certora as _

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Iterable, NotRequired, TypeVar, Literal, get_args, get_origin, Any, override, Coroutine, Awaitable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import InjectedToolCallId, tool
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.widgets import Button, Static, Header, Footer, Collapsible
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from langgraph.prebuilt import InjectedState
from langgraph.runtime import get_runtime
from langgraph.types import Command
from pydantic import BaseModel, Field, Discriminator

from composer.input.types import ModelOptions, RAGDBOptions, LangraphOptions
from composer.input.parsing import add_protocol_args
from composer.prover.analysis import analyze_cex
from composer.prover.ptypes import RuleResult
from composer.prover.results import read_and_format_run_result
from composer.rag.db import PostgreSQLRAGDatabase
from graphcore.graph import MessagesState, Builder
from graphcore.tools.vfs import VFSState, VFSToolConfig, fs_tools, VFSInput
from graphcore.tools.schemas import WithInjectedState, WithImplementation, WithInjectedId, WithImplementation

import uuid
from typing import cast

from langchain_core.runnables import RunnableConfig

from composer.spec.preaudit_setup import run_preaudit_setup, SetupFailure, SetupSuccess
from composer.spec.cvl_tools import put_cvl_raw, put_cvl
from composer.tools.search import cvl_manual_search
from composer.human.handlers import handle_human_interrupt
from graphcore.tools.results import result_tool_generator, ValidationResult
from langgraph.store.postgres import PostgresStore
from composer.workflow.services import create_llm, get_checkpointer, get_memory, get_store
from composer.rag.db import PostgreSQLRAGDatabase
from composer.rag.models import get_model
from composer.templates.loader import load_jinja_template
from graphcore.graph import build_workflow, FlowInput
from graphcore.tools.vfs import vfs_tools
from graphcore.tools.memory import memory_tool, SqliteMemoryBackend
from langgraph._internal._typing import StateLike
from langgraph.graph.state import CompiledStateGraph
from graphcore.summary import SummaryConfig

T = TypeVar('T')


MAX_TOOL_CONTENT_LENGTH = 500


def _normalize_content(s: str | list[str | dict]) -> list[dict]:
    """Normalize message content to a list of typed blocks."""
    l: list[str | dict]
    if isinstance(s, str):
        l = [s]
    else:
        l = s
    to_ret = []
    for r in l:
        if isinstance(r, str):
            to_ret.append({"type": "text", "text": r})
        else:
            to_ret.append(r)
    return to_ret


def _format_text_content(content: str) -> Text | Markdown:
    """Format string content with markdown if appropriate."""
    import re
    html_tag_pattern = r'<[^>]+>'
    if re.search(html_tag_pattern, content):
        return Text(content)

    markdown_markers = ['#', '*', '`', '```', '- ', '* ', '1. ', '## ', '### ']
    if any(marker in content for marker in markdown_markers):
        return Markdown(content)
    else:
        return Text(content)


class MessageWidget(Static):
    """Widget for displaying a single message."""

    def __init__(self, message: BaseMessage, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        match self.message:
            case AIMessage():
                yield from self._render_ai_message(self.message)
            case ToolMessage():
                yield from self._render_tool_message(self.message)
            case HumanMessage():
                yield from self._render_human_message(self.message)
            case SystemMessage():
                yield from self._render_system_message(self.message)
            case _:
                content = _normalize_content(self.message.content)
                text = "\n".join(c.get("text", str(c)) for c in content)
                yield Static(Panel(text, title=type(self.message).__name__))

    def _render_ai_message(self, msg: AIMessage) -> Iterable[Static | Collapsible]:
        content_parts: list[str] = []

        for block in _normalize_content(msg.content):
            block_type = block.get("type", "unknown")

            if block_type == "thinking":
                thinking_text = block.get("thinking", "")
                if thinking_text:
                    yield Collapsible(
                        Static(_format_text_content(thinking_text)),
                        title="Thinking",
                        collapsed=True
                    )
            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            elif block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_input = block.get("input", {})
                args_str = json.dumps(tool_input, indent=2) if tool_input else ""
                yield Static(Panel(
                    f"[bold]{tool_name}[/bold]\n{args_str}",
                    title="Tool Call",
                    border_style="yellow"
                ))

        if content_parts:
            yield Static(Panel(
                _format_text_content("\n".join(content_parts)),
                title="Assistant",
                border_style="magenta"
            ))

    def _render_tool_message(self, msg: ToolMessage) -> Iterable[Static]:
        content = str(msg.content)
        tool_name = getattr(msg, 'name', 'Tool Result')

        if len(content) > MAX_TOOL_CONTENT_LENGTH:
            truncated = content[:MAX_TOOL_CONTENT_LENGTH] + f"\n... [truncated {len(content) - MAX_TOOL_CONTENT_LENGTH} chars]"
            yield Static(Panel(truncated, title=tool_name, border_style="cyan"))
        else:
            yield Static(Panel(content, title=tool_name, border_style="cyan"))

    def _render_human_message(self, msg: HumanMessage) -> Iterable[Static]:
        content_parts: list[str] = []

        for block in _normalize_content(msg.content):
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            elif block_type == "image" or block_type == "image_url":
                content_parts.append("[Image content]")
            elif block_type == "document":
                content_parts.append("[Document content]")
            else:
                content_parts.append(f"[{block_type}]")

        if content_parts:
            combined = "\n".join(content_parts)
            if len(combined) > MAX_TOOL_CONTENT_LENGTH:
                truncated = combined[:MAX_TOOL_CONTENT_LENGTH] + f"\n... [truncated {len(combined) - MAX_TOOL_CONTENT_LENGTH} chars]"
                yield Static(Panel(truncated, title="Human", border_style="green"))
            else:
                yield Static(Panel(_format_text_content(combined), title="Human", border_style="green"))

    def _render_system_message(self, msg: SystemMessage) -> Iterable[Static]:
        content_parts: list[str] = []

        for block in _normalize_content(msg.content):
            block_type = block.get("type", "unknown")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            else:
                content_parts.append(f"[{block_type}]")

        if content_parts:
            combined = "\n".join(content_parts)
            if len(combined) > MAX_TOOL_CONTENT_LENGTH:
                truncated = combined[:MAX_TOOL_CONTENT_LENGTH] + f"\n... [truncated {len(combined) - MAX_TOOL_CONTENT_LENGTH} chars]"
                yield Static(Panel(truncated, title="System", border_style="blue"))
            else:
                yield Static(Panel(_format_text_content(combined), title="System", border_style="blue"))


class UpdateWidget(Static):
    """Widget for displaying a state update."""

    def __init__(self, node_name: str, state_update: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_name = node_name
        self.state_update = state_update

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]Node: {self.node_name}[/bold]", classes="update-header")

        if "messages" in self.state_update:
            for msg in self.state_update["messages"]:
                if isinstance(msg, BaseMessage):
                    yield MessageWidget(msg)

        other_keys = {k: v for k, v in self.state_update.items() if k != "messages"}
        if other_keys:
            summary = ", ".join(f"{k}={type(v).__name__}" for k, v in other_keys.items())
            yield Static(f"[dim]State updates: {summary}[/dim]")


class GraphRunnerApp(App):
    """TUI for running a LangGraph to completion with pause/resume."""

    CSS = """
    #status-bar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    #status-bar Static {
        width: auto;
    }

    #message-area {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #controls {
        dock: bottom;
        height: 3;
        align: center middle;
    }

    .update-header {
        background: $boost;
        padding: 0 1;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("p", "toggle_pause", "Pause/Resume"),
        ("q", "quit", "Quit"),
    ]

    is_paused: reactive[bool] = reactive(False)
    is_complete: reactive[bool] = reactive(False)

    def __init__(
        self,
        graph: CompiledStateGraph,
        input: Any,
        thread_id: str,
        initial_checkpoint_id: str | None,
        recursion_limit: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.graph = graph
        self.graph_input = input
        self.thread_id = thread_id
        self.checkpoint_id = initial_checkpoint_id
        self.recursion_limit = recursion_limit
        self._stream_task: asyncio.Task | None = None
        self._error: Exception | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="status-bar"):
            yield Static(f"Thread: {self.thread_id}", id="thread-display")
            yield Static(" | ", classes="separator")
            yield Static("Checkpoint: -", id="checkpoint-display")
            yield Static(" | ", classes="separator")
            yield Static("Node: -", id="node-display")
        yield ScrollableContainer(id="message-area")
        with Horizontal(id="controls"):
            yield Button("Pause", id="pause-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self._start_streaming()

    def _start_streaming(self) -> None:
        self._stream_task = asyncio.create_task(self._run_stream())

    async def _run_stream(self) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": self.recursion_limit,
        }

        if self.checkpoint_id is not None:
            config["configurable"]["checkpoint_id"] = self.checkpoint_id

        stream_input = None if self.checkpoint_id is not None else self.graph_input

        try:
            async for (tag, payload) in self.graph.astream(
                input=stream_input,
                config=config,
                stream_mode=["checkpoints", "updates"]
            ):
                if self.is_paused:
                    break

                assert isinstance(payload, dict)

                if tag == "checkpoints":
                    new_checkpoint = payload["config"]["configurable"]["checkpoint_id"]
                    self.checkpoint_id = new_checkpoint
                    self.query_one("#checkpoint-display", Static).update(f"Checkpoint: {new_checkpoint[:12]}...")
                else:
                    for node_name, update in payload.items():
                        if node_name == "__interrupt__":
                            continue
                        self.query_one("#node-display", Static).update(f"Node: {node_name}")

                        if isinstance(update, dict):
                            message_area = self.query_one("#message-area", ScrollableContainer)
                            widget = UpdateWidget(node_name, update)
                            await message_area.mount(widget)
                            message_area.scroll_end(animate=False)

            if not self.is_paused:
                self.is_complete = True
                self.query_one("#pause-btn", Button).label = "Complete"
                self.query_one("#pause-btn", Button).disabled = True

        except Exception as e:
            self._error = e
            self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pause-btn":
            self.action_toggle_pause()

    def action_toggle_pause(self) -> None:
        if self.is_complete:
            return

        self.is_paused = not self.is_paused
        btn = self.query_one("#pause-btn", Button)

        if self.is_paused:
            btn.label = "Resume"
            btn.variant = "success"
        else:
            btn.label = "Pause"
            btn.variant = "primary"
            self._start_streaming()

    async def action_quit(self) -> None:
        if self.is_complete:
            self.exit()


def run_to_completion[I: StateLike, S: StateLike](
    graph: CompiledStateGraph[S, None, I, Any],
    input: I,
    thread_prefix: str,
    *,
    thread_id: str | None = None,
    checkpoint_id: str | None = None,
    recursion_limit: int = 100,
) -> S:
    """
    Run a compiled state graph to completion using a TUI with pause/resume support.

    Args:
        graph: The compiled state graph to execute
        input: The input to the graph (ignored if checkpoint_id is set)
        thread_prefix: Prefix for auto-generated thread IDs.
        thread_id: Optional thread ID for checkpointing. Auto-generated if None.
        checkpoint_id: Optional checkpoint ID to resume from.
        recursion_limit: Maximum recursion depth (default 100).

    Returns:
        The final state after graph completion.
    """
    if thread_id is None:
        thread_id = f"{thread_prefix}{uuid.uuid1().hex}"
        print(f"Chose thread id {thread_id}")

    app = GraphRunnerApp(
        graph=graph,
        input=input,
        thread_id=thread_id,
        initial_checkpoint_id=checkpoint_id,
        recursion_limit=recursion_limit,
    )

    app.run()

    if app._error is not None:
        raise app._error

    return cast(S, graph.get_state({"configurable": {"thread_id": thread_id}}).values)

R = TypeVar('R')
_S = TypeVar('_S', bound=MessagesState)
_C = TypeVar('_C', bound=StateLike | None)
_I = TypeVar('_I', bound=FlowInput | None)

def bind_standard(
    builder: Builder[Any, _C, _I],
    state_type: type[_S],
    doc: str | None = None,
    validator: Callable[[_S], str | None] | None = None
) -> Builder[_S, _C, _I]:
    """
    Bind a state type to the builder and generate a result tool based on the state's `result` annotation.

    Extracts the result type from the state's `result: NotRequired[T]` annotation and generates
    a result tool using `result_tool_generator`. The tool is then attached to the builder.

    Args:
        builder: The builder to modify
        state_type: The state type to bind, must have a `result: NotRequired[T]` annotation
        doc: Description for the result field. Required if the result type is not a BaseModel.

    Returns:
        Builder with state bound and result tool attached, preserving context and input types

    Raises:
        ValueError: If state_type has no 'result' annotation, or if doc is missing for non-BaseModel result types
    """
    annotations = getattr(state_type, '__annotations__', {})
    if 'result' not in annotations:
        raise ValueError(f"State type {state_type.__name__} must have a 'result' annotation")

    result_annotation = annotations['result']

    # Extract inner type from NotRequired[T]
    origin = get_origin(result_annotation)
    if origin is NotRequired:
        result_type = get_args(result_annotation)[0]
    else:
        result_type = result_annotation

    # Check if result_type is a BaseModel
    is_basemodel = isinstance(result_type, type) and issubclass(result_type, BaseModel)

    if not is_basemodel and doc is None:
        raise ValueError(f"doc parameter is required when result type {result_type} is not a BaseModel")

    tool_doc = "Used to indicate successful completion with result."

    valid : tuple[type[_S], Callable[[_S, Any, str], ValidationResult]] | None = None
    if validator:
        valid = (state_type, lambda s, r, id: validator(s))

    # Generate the result tool
    if is_basemodel:
        result_tool = result_tool_generator("result", result_type, tool_doc, valid)
    else:
        assert doc is not None
        result_tool = result_tool_generator("result", (result_type, doc), tool_doc, valid)

    # Bind state and add tool
    return builder.with_state(state_type).with_tools([result_tool]).with_output_key("result").with_default_summarizer(
        max_messages=50
    )


@dataclass
class SourceSpecContext:
    """Context for source-based spec generation."""
    project_root: Path
    main_contract: str
    main_contract_path: str
    compilation_config: dict
    summaries_import: str | None
    rag_db: PostgreSQLRAGDatabase
    unbound_llm: BaseChatModel

class SourceSpecInput(FlowInput):
    vfs: dict
    curr_spec: None

class SourceSpecState(MessagesState, VFSState):
    """State for source-based spec generation workflow."""
    curr_spec: str | None
    result: NotRequired[dict]


class SourceSpecArgs(ModelOptions, RAGDBOptions, LangraphOptions):
    """Arguments for source-based spec generation."""
    project_root: str
    main_contract: str
    system_doc: str


def apply_async_parallel(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T]
) -> list[R]:
    """
    Apply an async function to items in parallel and return results.

    Works whether or not there's an active event loop.
    """
    async def _gather_results():
        tasks = [func(item) for item in items]
        return await asyncio.gather(*tasks)

    in_loop = False
    try:
        # Check if there's a running event loop
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        pass
    if not in_loop:
        return asyncio.run(_gather_results())
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, _gather_results())
            return future.result()


class VerifySpecSchema(BaseModel):
    """
    Run the Certora prover to verify the current spec against the source code.

    Returns verification results:
    - VERIFIED: Rule holds for all inputs
    - VIOLATED: Counterexample found (with CEX analysis)
    - TIMEOUT: Verification did not complete in time

    Use these results to refine your spec.
    """
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[SourceSpecState, InjectedState]

    rules: list[str] | None = Field(
        default=None,
        description="Specific rules to verify. If None, verifies all rules."
    )
    timeout: int = Field(
        default=300,
        description="Per-rule timeout in seconds"
    )


async def _analyze(
    llm: BaseChatModel, state, res: RuleResult, tool_call_id: str
) -> tuple[RuleResult, str | None]:
    cex_analysis = None
    if res.status == "VIOLATED":
        cex_analysis = await analyze_cex(llm, state, res, tool_call_id=tool_call_id)
    return (res, cex_analysis)


@tool(args_schema=VerifySpecSchema)
def verify_spec(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated["SourceSpecState", InjectedState],
    rules: list[str] | None = None,
    timeout: int = 300
) -> str | Command:
    """Run Certora prover to verify the spec against source code."""
    context = get_runtime(SourceSpecContext).context

    curr_spec = state.get("curr_spec")
    if curr_spec is None:
        return "No spec has been written yet. Use put_cvl_raw or put_cvl to create a spec first."

    # Create temp directory and copy source

    certora_dir = Path(context.project_root) / "certora"

    (certora_dir / "generated.spec").write_text(curr_spec)

    config = {
        **context.compilation_config,
        "verify": f"{context.main_contract}:certora/generated.spec",
        "parametric_contracts": context.main_contract,
        "optimistic_loop": True,
        "rule_sanity": "basic",
    }

    if rules:
        config["rule"] = rules

    # Write config file
    config_path = certora_dir / "verify.conf"
    config_path.write_text(json.dumps(config, indent=2))

    # Run certoraRunWrapper
    wrapper_script = Path(__file__).parent.parent / "prover" / "certoraRunWrapper.py"
    with tempfile.NamedTemporaryFile("rb", suffix=".pkl") as output_file:

        proc_result = subprocess.run(
            [sys.executable, str(wrapper_script), str(output_file.name), str(config_path)],
            cwd=context.project_root,
            capture_output=True,
            text=True,
            timeout=timeout * 2,
        )

        # Read the pickled output
        import pickle
        run_result = pickle.load(output_file)


    # Check for errors
    if proc_result.returncode != 0:
        return f"Verification failed:\nstdout:\n{proc_result.stdout}\nstderr:\n{proc_result.stderr}"


    # Check if it's an exception
    if isinstance(run_result, Exception):
        return f"Certora prover raised exception: {str(run_result)}\nstdout:\n{proc_result.stdout}"

    if run_result is None or not run_result.is_local_link or run_result.link is None:
        return f"Prover did not produce local results.\nstdout:\n{proc_result.stdout}"

    emv_path = Path(run_result.link)

    # Parse results using existing infrastructure
    results = read_and_format_run_result(emv_path)

    if isinstance(results, str):
        # Error occurred during parsing
        return f"Failed to parse prover results: {results}"

    # Run CEX analysis for violated rules using apply_async_parallel
    results_with_analysis = apply_async_parallel(
        lambda res: _analyze(context.unbound_llm, state, res, tool_call_id),
        list(results.values())
    )

    # Format results for LLM
    lines = ["## Verification Results\n"]
    verified, violated, timeout_count = 0, 0, 0

    for rule_result, cex_analysis in results_with_analysis:
        status = rule_result.status
        name = rule_result.name

        if status == "VERIFIED":
            verified += 1
            lines.append(f"✓ **{name}**: VERIFIED")
        elif status == "VIOLATED":
            violated += 1
            lines.append(f"✗ **{name}**: VIOLATED")
            if cex_analysis:
                lines.append(f"  Analysis: {cex_analysis}")
        elif status == "TIMEOUT":
            timeout_count += 1
            lines.append(f"⏱ **{name}**: TIMEOUT")
        else:
            lines.append(f"? **{name}**: {status}")

    lines.append(f"\n**Summary**: {verified} verified, {violated} violated, {timeout_count} timeout")

    return "\n".join(lines)

class GetCVLSchema(BaseModel):
    """
    View the (pretty-printed) version of the CVL file.
    """
    state: Annotated[SourceSpecState, InjectedState]

@tool(args_schema=GetCVLSchema)
def get_cvl(
    state: Annotated[SourceSpecState, InjectedState]
) -> str:
    if state["curr_spec"] is None:
        return "No spec file on VFS"
    return state["curr_spec"]

@dataclass
class ContractSpec:
    relative_path: str
    contract_name: str

class ComponentInteraction(BaseModel):
    """
    Describes an interaction between some component and another
    """
    other_component: str = Field(description="The name of the other component with which this component interacts")
    interaction_description: str = Field(description="Why the interaction occurs, and a brief description of what the interaction looks like")

class ExternalDependency(BaseModel):
    """
    A single external dependency for a component
    """
    name: str = Field(description="A succint name for the external dependency (e.g., 'Price Oracle', 'Off-chain oracle', 'ERC20 asset token', etc.)")
    requirements: list[str] = Field(description="A list of assumptions/requirements that this external dependency must satisfy (e.g., 'Honest validator', 'implements a standard erc20 interface', etc.)")

class ApplicationComponent(BaseModel):
    """
    A single component within the application
    """
    name: str = Field(description="The brief, concise name of the component (e.g., Price Tracking/Token Management/etc.)")
    description: str = Field(description="A longer description of *what* the component does, not *how* it does it.")
    requirements: list[str] = Field(description="A list of short, succint natural language requirements describing the component's *intended* behavior")
    external_entry_points: list[str] = Field(description="The signatures/names of any external methods that comprise this component")
    state_variables: list[str] = Field(description="State variables involved in the component")
    interactions: list[ComponentInteraction] = Field(description="A list of interactions with other components")

    dependencies: list[ExternalDependency] = Field(description="A list of external dependencies for this component")

class ApplicationSummary(BaseModel):
    """
    A summary of your analysis of the application
    """
    application_type : str = Field(description="A short, concise description of the type of application (AMM/Liquidity Provider/etc.)")
    components: list[ApplicationComponent] = Field(description="The list of components in the application")

def get_system_doc(sys_path: Path) -> dict | str | None:
    """Load a system document from a file path, returning base64-encoded PDF or text."""
    if not sys_path.is_file():
        print("System file not found")
        return None
    if sys_path.suffix == ".pdf":
        file_data = base64.standard_b64encode(sys_path.read_bytes()).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": file_data
            }
        }
    else:
        return sys_path.read_text()


def _hash_file(path: Path) -> str:
    """Return SHA256 hash of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_key_source_analysis(args: "SourceSpecArgs", spec: "ContractSpec") -> str:
    """Generate a cache key for source analysis based on inputs."""
    components = [
        args.project_root,
        _hash_file(Path(args.system_doc)),
        spec.relative_path,
        spec.contract_name,
    ]
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _cache_key_bug_analysis(
    args: "SourceSpecArgs",
    spec: "ContractSpec",
    component: "ApplicationComponent",
    summ: str
) -> str:
    """Generate a cache key for bug analysis based on inputs."""
    components = [
        args.project_root,
        spec.relative_path,
        spec.contract_name,
        component.model_dump_json(),
        summ,
    ]
    combined = "|".join(str(c) for c in components)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]

# Common forbidden read pattern for source analysis
FS_FORBIDDEN_READ = "(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"

def run_source_analysis(
    args: SourceSpecArgs,
    thread_id: str,
    spec: ContractSpec,
    store: PostgresStore,
    builder: Builder[None, None, FlowInput]
) -> ApplicationSummary | None:
    # Check cache first
    cache_key = _cache_key_source_analysis(args, spec)
    cached = store.get(("source_analysis",), cache_key)
    if cached is not None:
        print(f"Using cached source analysis (key={cache_key})")
        return ApplicationSummary.model_validate(cached.value)
    
    system_doc = get_system_doc(Path(args.system_doc))
    if system_doc is None:
        return None

    memory = memory_tool(get_memory(f"source-summary-{thread_id}"))

    class AnalysisState(MessagesState):
        result: NotRequired[ApplicationSummary]

    graph = bind_standard(
        builder=builder,
        state_type=AnalysisState
    ).with_sys_prompt(
        "You are an expert software analyst, who is very methodical in their work."
    ).with_tools(
        [memory]
    ).with_initial_prompt_template(
        "source_summarization.j2",
        main_contract_name=spec.contract_name,
        relative_path=spec.relative_path
    ).build()[0].compile(
        checkpointer=get_checkpointer()
    )
    task_thread_id = "summary-extraction-" + thread_id

    input: FlowInput = FlowInput(
        input=[
            "The system document is as follows",
            system_doc
        ]
    )

    res = graph.invoke(
        input=input,
        config={"configurable": {
            "thread_id": task_thread_id
        }, "recursion_limit": 50}
    )
    result: ApplicationSummary = res["result"]

    # Cache the result
    store.put(("source_analysis",), cache_key, result.model_dump())
    print(f"Cached source analysis (key={cache_key})")

    return result

def format_container(d: dict) -> str:
    c = d.get("containingContract", None)
    if c is None:
        return "at the top level"
    else:
        return f"in contract {c}"

def format_type(s: dict) -> str | None:
    kind = s.get("typeCategory", None)
    if not kind:
        return None
    where_def = format_container(s)
    ty_name = s.get("typeName", None)
    if not ty_name:
        return None
    qual_name = s.get("qualifiedName", None)
    match kind:
        case "UserDefinedStruct":
            return f"A struct {ty_name} {where_def}: use `{qual_name}`"
        case "UserDefinedEnum":
            return f"An enum {ty_name} {where_def}: use `{qual_name}`"
        case "UserDefinedValueType":
            base = s.get("baseType", None)
            if not base:
                return None
            return f"An alias for {base} called {ty_name} {where_def}: use `{qual_name}`"
        case _:
            return None

def format_types(udts: list[dict]) -> str:
    to_format: list[str] = []
    for ty in udts:
        r = format_type(ty)
        if not r:
            continue
        to_format.append(r)
    return "\n".join(to_format)

class PropertyFormulation(BaseModel):
    """
    A property or invariant that must hold for the component
    """
    methods: list[str] | Literal["invariant"] = Field(description="A list of external methods involved in the property, or 'invariant' if the property is an invariant on the contract state")
    sort: Literal["attack_vector", "safety_property", "invariant"] = Field(description="The type of property you are describing.")
    description: str = Field(description="The description of the property")

    def to_template_args(self) -> dict:
        thing : str
        what_formal : str
        prop = self
        match prop.sort:
            case "attack_vector":
                thing = "potential attack vector/exploit"
                what_formal = f"that a {thing} is not possible"
            case "invariant":
                thing = "invariant"
                what_formal = "that an invariant holds"
            case "safety_property":
                thing = "safety property"
                what_formal = "that a safety property holds"
        return {
            "thing": thing,
            "what_formal": what_formal,
            "thing_tag": self.sort,
            "thing_descr": self.description
        }

def run_bug_analysis(
    args: SourceSpecArgs,
    component: "ComponentInst",
    builder: Builder[None, None, FlowInput],
    store: PostgresStore,
) -> list[PropertyFormulation] | None:
    # Check cache first
    cache_key = _cache_key_bug_analysis(args, component, component.component, component.summ.application_type)
    cached = store.get(("bug_analysis_2",), cache_key)
    if cached is not None:
        print(f"Using cached bug analysis (key={cache_key})")
        return [PropertyFormulation.model_validate(p) for p in cached.value["items"]]

    class ST(MessagesState):
        result: NotRequired[list[PropertyFormulation]]

    d = bind_standard(
        builder, ST, "The security properties you have extracted about the component"
    ).with_initial_prompt_template(
        "property_analysis_prompt.j2",
        context=component
    ).with_sys_prompt(
        "You are an expert security and software analyst, with extensive knowledge of the types of issues and vulnerabilities found in DeFi protocols"
    ).build()[0].compile()

    r = d.invoke(input=FlowInput(input=[]))

    result: list[PropertyFormulation] = r["result"]

    # Cache the result
    store.put(("bug_analysis_2",), cache_key, {"items": [p.model_dump() for p in result]})
    print(f"Cached bug analysis (key={cache_key})")

    return result

@dataclass
class ComponentInst(ContractSpec):
    summ: ApplicationSummary
    ind: int

    @property
    def component(self) -> ApplicationComponent:
        return self.summ.components[self.ind]
    
    @property
    def application_type(self) -> str:
        return self.summ.application_type

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")

type FeedbackTool = Callable[[str], PropertyFeedback]

def property_feedback_judge(
    builder: Builder[None, None, FlowInput],
    inst: ComponentInst, prop: PropertyFormulation
) -> FeedbackTool:
    
    class ST(MessagesState):
        memory: NotRequired[str]
        result: NotRequired[PropertyFeedback]
        did_read: NotRequired[bool]

    class GetMemory(WithInjectedState[ST], WithImplementation[Command | str], WithInjectedId):
        """
        Retrieve the rough draft of the feedback
        """
        @override
        def run(self) -> str | Command:
            mem = self.state.get("memory", None)
            if mem is None:
                return "Rough draft not yet written"
            return Command(update={
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content=mem)],
                "did_read": True
            })

    class SetMemory(WithInjectedId, WithImplementation[Command]):
        """
        Write your rough draft for review
        """
        rough_draft : str = Field(description="The new rough draft of your feedback")

        @override
        def run(self) -> Command:
            return Command(update={
                "memory": self.rough_draft,
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content="Success")]
            })

    def did_rough_draft_read(s: ST) -> str | None:
        h = s.get("did_read", None) is None
        if h is None:
            return "Completion REJECTED: never read rough draft for review"
        return None

    db = sqlite3.connect(":memory:", check_same_thread=False)
    memory = memory_tool(SqliteMemoryBackend("dummy", db))
    workflow = bind_standard(
        builder, ST, validator=did_rough_draft_read
    ).with_initial_prompt_template(
        "property_judge_prompt.j2",
        context=inst,
        **prop.to_template_args()
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_tools([SetMemory.as_tool("write_rough_draft"), GetMemory.as_tool("read_rough_draft"), memory]).build()[0].compile()

    def the_tool(
        cvl: str
    ) -> PropertyFeedback:
        print("HIYA")
        print(cvl)
        res = workflow.invoke(FlowInput(input=[
            "The proposed CVL file is",
            cvl
        ]))
        r = res["result"]
        print(f"Returning feedback \n{r.feedback}\nFor:{prop}")
        return r

    return the_tool

def generate_property_cvl(
    feat: ComponentInst,
    prop: PropertyFormulation,
    builder: Builder[None, None, FlowInput],
    store: PostgresStore
) -> tuple[str, str]:
    
    class ST(MessagesState):
        curr_spec: NotRequired[str]
        result: NotRequired[str]

    feedback = property_feedback_judge(
        builder, feat, prop
    )

    class FeedbackSchema(WithInjectedState[ST], WithImplementation[str]):
        """
        Receive feedback on your CVL
        """
        @override
        def run(self) -> str:
            st = self.state
            spec = st.get("curr_spec", None)
            if spec is None:
                return "No spec put yet"
            t = feedback(spec)
            return f"""
Good? {str(t.good)}
Feedback {t.feedback}
"""

    d = bind_standard(
        builder, ST, "A description of your generated CVL"
    ).with_tools(
        [put_cvl, put_cvl_raw, FeedbackSchema.as_tool("feedback_tool")]
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt_template(
        "property_generation_prompt.j2",
        context=feat,
        **prop.to_template_args()
    ).build()[0].compile()

    r = d.invoke(input=FlowInput(input=[]), config={"recursion_limit": 100})

    return (r["result"], r["curr_spec"])

type PropertyFormalization = tuple[PropertyFormulation, str, str]

def _analyze_component(
    args: SourceSpecArgs,
    feat: ComponentInst,
    b: Builder[None, None, FlowInput],
    cvl_builder: Builder[None, None, FlowInput],
    store: PostgresStore
) -> None | list[tuple[PropertyFormulation, str, str]]:
    res = run_bug_analysis(args, feat, b, store)
    if res is None:
        print("Didn't work")
        return None
    work : list[tuple[PropertyFormulation, str, str]] = []
    for prop in res:
        print(prop)
        r, cvl = generate_property_cvl(
            feat, prop, cvl_builder, store
        )
        print(cvl)
        print(r)
        work.append((prop, r, cvl))
    return work

@dataclass
class Configuration(SetupSuccess):
    vfs_diff: dict[str, str]

class ExternalActor(BaseModel):
    name: str = Field(description="A short, descriptive name of the external contract being interacted with")
    description: str = Field(description="A short, precise description of what the external actor does and its interaction with the main contract")

class Summarizable(ExternalActor):
    l: Literal["SUMMARIZABLE"]
    suggested_summaries: str = Field(description="A natural language description of the suggested summaries to use for this contract")
    path: str | None = Field(description="The relative path to the source of the contract to be summarized; null if the implementation isn't available")

class SourceAvailable(ExternalActor):
    path: str = Field(description="The relative path to the source of the contract being described")

class NotFoundHavoc(ExternalActor):
    l: Literal["NOTFOUND_HAVOC"]


class Singleton(SourceAvailable):
    l: Literal["SINGLETON"]

class HarnessDef(BaseModel):
    path: str = Field(description="Path to the harness definition")
    harness_name: str = Field(description="The name of the contract defined in the harness file")
    suggested_role: str = Field(description="The suggested role of this harness; e.g., 'the first token' of the pool, etc.")


class WithHarnesses(BaseModel):
    harnesses: list[HarnessDef] = Field(description="The harnesses created to model this contract.")

class Dynamic(SourceAvailable, WithHarnesses):
    l: Literal["DYNAMIC"]

class Multiple(SourceAvailable, WithHarnesses):
    l: Literal["MULTIPLE"]

type ContractClassification = Annotated[
    Summarizable |
    NotFoundHavoc |
    Dynamic |
    Singleton |
    Multiple,
    Discriminator("l")
]

class ERC20TokenGuidance(WithImplementation[Command], WithInjectedId):
    """
    Invoke this tool to receive guidance on how ERC20 is usually modelled using the prover.

    You MUST NOT invoke this tool in parallel with other tools.
    """
    @override
    def run(self) -> Command:
        return Command(update={
            "messages": [ToolMessage(
                tool_call_id=self.tool_call_id,
                content="Advice is as follows..."
            ), HumanMessage(
                content=[load_jinja_template(
                    "erc20_advice.j2"
                ), "Carefully consider if explicit ERC20 contract instances are necessary for this protocol, or if the 'standard summarization' is sufficient."]
            )]
        })

class ContractSetup(BaseModel):
    """
    The result of your analysis
    """
    external_contracts: list[ContractClassification] = Field(description="The external actors classified by your analysis")
    primary_entity: str = Field(description="A description of the primary entity managed by this contract")
    non_trivial_state: str = Field(description="A semi-formal description of a `non-trivial state`. Reference the external " \
    "contracts you identified during the harnessing step as necessary.")

def _preaudit_agent(
    args: SourceSpecArgs,
    contract_spec: ContractSpec,
    b: Builder[None, None, None],
) -> Configuration:
    class ST(VFSState, MessagesState):
        result: NotRequired[ContractSetup]

    fs_tools, mat = vfs_tools(conf=VFSToolConfig(
        immutable=False,
        forbidden_read=FS_FORBIDDEN_READ,
        fs_layer=args.project_root,
        forbidden_write=r"^(?!certora/)",
        put_doc_extra="You may only write files in the certora/ subdirectory"
    ), ty=ST)

    def validate_mocks(
        s: ST,
        r: ContractSetup,
        tid: str
    ) -> ValidationResult:
        errors = []
        for m in r.external_contracts:
            if m.l == "DYNAMIC" or m.l == "MULTIPLE":
                for h in m.harnesses:
                    path = h.path
                    if path not in s["vfs"]:
                        errors.append(f"Harness {path} for {m.path} not found on the VFS")
        if errors:
            return "Update rejected:\n" + "\n".join(errors)
        return None

    result = result_tool_generator(
        "result",
        ContractSetup,
        "Tool to communicate the result of your analysis",
        validator=(ST, validate_mocks)
    )

    memory_back = sqlite3.connect(":memory:", check_same_thread=False)
    mem = memory_tool(SqliteMemoryBackend("ns", memory_back))

    graph = b.with_input(
        VFSInput
    ).with_output_key(
        "result"
    ).with_state(
        ST
    ).with_default_summarizer(
        max_messages=50
    ).with_sys_prompt(
        "You are an expert Solidity developer who is very good at following instructions who works at Certora, Inc."
    ).with_tools(
        [*fs_tools, result, mem, ERC20TokenGuidance.as_tool("erc20_guidance")]
    ).with_initial_prompt_template(
        "harness_prompt.j2",
        contract_spec=contract_spec
    ).build_async()[0].compile(checkpointer=get_checkpointer())

    st = run_to_completion(
        graph,
        input=VFSInput(vfs={}, input=[]),
        recursion_limit=100,
        thread_prefix="harness-setup-"
    )

    res : ContractSetup = st["result"] # pyright: ignore[reportTypedDictNotRequiredAccess]
    for r in res.external_contracts:
        print("=" * 80)
        print("Name: " + r.name)
        print("> " + r.description)
        print("Sort: " + r.l)
        print("-" * 80)
        if r.l != "NOTFOUND_HAVOC":
            print(f"path = {r.path}")
        match r.l:
            case "MULTIPLE" | "DYNAMIC":
                for h in r.harnesses:
                    print(f"Harness of {r.path}")
                    print(st["vfs"].keys())
                    print(h.path)
                    print(st["vfs"][h.path])
            case "SUMMARIZABLE":
                print(r.suggested_summaries)



    sys.exit(10)

def execute(args: SourceSpecArgs) -> int:
    """Execute source-based spec generation workflow."""

    thread_id = args.thread_id if args.thread_id else f"source_spec_{uuid.uuid4().hex}"
    print(f"Thread ID: {thread_id}")

    project_root = Path(args.project_root)

    main_contract_path, main_contract_name = args.main_contract.split(":", 1)

    full_contract_path = Path(main_contract_path).resolve()

    if not full_contract_path.is_relative_to(project_root.resolve()):
        print(f"Invalid path: {full_contract_path} doesn't appear in project root {project_root}")
        return 1
    
    relativized_main = full_contract_path.relative_to(project_root.resolve())

    spec = ContractSpec(str(relativized_main), main_contract_name)


    store = get_store()

    llm = create_llm(args)

    basic_builder = Builder().with_llm(llm).with_loader(load_jinja_template)

    _preaudit_agent(
        args, spec, basic_builder
    )

    b : Builder[None, None, FlowInput] = Builder().with_llm(
        llm
    ).with_input(
        FlowInput
    ).with_loader(
        load_jinja_template
    ).with_tools(
        fs_tools(args.project_root, forbidden_read=FS_FORBIDDEN_READ)
    )
    
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    analysis = run_source_analysis(args, thread_id, spec, store, b)

    cvl_builder = b.with_tools(
        [cvl_manual_search(rag_db)]
    )

    if analysis is None:
        print("Oh well")
        return 1


    work : list[tuple[ComponentInst, None | list[PropertyFormalization]]] = []
    for feature_idx in range(0, len(analysis.components)):
        def thunk() -> tuple[ComponentInst, None | list[PropertyFormalization]]:
            feat = ComponentInst(
                contract_name=spec.contract_name,
                relative_path=spec.relative_path,
                summ=analysis,
                ind=feature_idx
            )
            l = _analyze_component(
                args, feat, b, cvl_builder, store
            )
            return (feat, l)
        work.append(thunk())



    sys.exit(1)

    # Step 1: Run PreAudit setup
    print("Running PreAudit compilation analysis...")
    setup_result = run_preaudit_setup(
        project_root=Path(args.project_root),
        main_contract=main_contract_name,
        relative_path=str(relativized_main)
    )
    match setup_result:
        case SetupFailure(error=e):
            print(f"Auto setup failed: {e}")
            return 1
        case _:
            pass

    # Step 2: Create LLM and thread
    llm = create_llm(args)

    # Step 3: Build context
    rag_db = PostgreSQLRAGDatabase(
        conn_string=args.rag_db,
        model=get_model(),
        skip_test=True
    )

    summaries_import = f'import "{setup_result.summaries_path}";'

    context = SourceSpecContext(
        project_root=Path(args.project_root),
        main_contract=main_contract_name,
        main_contract_path=main_contract_path,
        compilation_config=setup_result.config,
        summaries_import=summaries_import,
        rag_db=rag_db,
        unbound_llm=llm
    )

    # Step 4: Build workflow
    manual = cvl_manual_search(SourceSpecContext)
    checkpointer = get_checkpointer()

    memory = memory_tool(get_memory(f"memory-{thread_id}"))
    generation_complete = result_tool_generator(
        outkey="result",
        result_schema=(dict, "Final result"),
        doc="Used to indicate successful result of your analysis."
    )

    v_tools, _ = vfs_tools(
        conf=VFSToolConfig(immutable=True, fs_layer=args.project_root, forbidden_read="(^lib/.*)|(^\\.certora_internal.*)|(^\\.git.*)|(^test/.*)|(^emv-.*)"),
        ty=SourceSpecState
    )

    graph = build_workflow(
        context_schema=SourceSpecContext,
        initial_prompt=load_jinja_template(
            "source_spec_prompt.j2",
            main_contract=args.main_contract,
            summaries_import=summaries_import
        ),
        sys_prompt=load_jinja_template("cvl_system_prompt.j2"),
        input_type=SourceSpecInput,
        output_key="result",
        state_class=SourceSpecState,
    
        unbound_llm=llm,
        summary_config=SummaryConfig(max_messages=50),
        tools_list=[
            # VFS tools for source code access
            *v_tools,

            # CVL spec tools
            put_cvl,
            put_cvl_raw,
            get_cvl,

            # Verification
            verify_spec,

            # Support tools
            manual,
            memory,
            generation_complete
        ]
    )[0].compile(checkpointer=checkpointer, store=get_store())

    # Step 5: Run workflow
    def fresh_config() -> RunnableConfig:
        return {
            "recursion_limit": args.recursion_limit,
            "configurable": {
                "thread_id": thread_id
            }
        }

    runnable_conf = fresh_config()

    sys_doc = get_system_doc(Path(args.system_doc))
    assert sys_doc is not None

    if args.checkpoint_id is not None:
        runnable_conf["configurable"]["checkpoint_id"] = args.checkpoint_id #type: ignore

    formatted_types = format_types(setup_result.user_types)

    graph_input = SourceSpecInput(input=[
        "The major components of the software is as follows",
        format_summary_xml(analysis),
        "The system document is as follows",
        sys_doc,
        "User defined types can be referenced in your specification, according to the following guidance",
        formatted_types
        ], curr_spec=None, vfs={}) if args.checkpoint_id is None else None

    while True:
        t = graph_input
        graph_input = None
        for (tag, payload) in graph.stream(
            input=t,
            config=runnable_conf,
            context=context,
            stream_mode=["updates", "checkpoints"]
        ):
            assert isinstance(payload, dict)
            if tag == "checkpoints":
                print("current checkpoint: " + payload["config"]["configurable"]["checkpoint_id"])
                continue
            if "__interrupt__" in payload:
                if "configurable" in runnable_conf and "checkpoint_id" in runnable_conf["configurable"]:
                    del runnable_conf["configurable"]["checkpoint_id"]
                interrupt_data = cast(dict, payload["__interrupt__"][0].value)
                def debug_thunk() -> None:
                    pass  # TODO: implement debug console if needed
                human_response = handle_human_interrupt(interrupt_data, debug_thunk)
                graph_input = Command(resume=human_response)
                break
            else:
                print(payload)
        if graph_input is None:
            break

    final_state = cast(SourceSpecState, graph.get_state(fresh_config()).values)
    if "result" not in final_state:
        return 1

    print("Spec file generation complete")
    print(final_state["result"])

    return 0

def auto_prover() -> int:
    parser = argparse.ArgumentParser()
    add_protocol_args(parser, ModelOptions)
    add_protocol_args(parser, RAGDBOptions)
    add_protocol_args(parser, LangraphOptions)
    parser.add_argument("project_root")
    parser.add_argument("main_contract")
    parser.add_argument("system_doc")

    res = cast(SourceSpecArgs, parser.parse_args())

    return execute(res)
