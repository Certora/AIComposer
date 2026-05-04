"""
Console-mode handler for the auto-prove source-spec pipeline.

Create one ``AutoProveConsoleHandler``, then pass ``handler.make_handler`` as
the ``handler_factory`` argument to ``run_autoprove_pipeline``.  The same
handler instance is reused across all phases so that path descriptions
accumulate correctly across the whole pipeline run.

Log format:

- Phase boundaries:   ``─────`` header printed by ``on_start``
- Start/end events:   ``[Foo / Bar] start``  /  ``[Foo / Bar] end``
- State updates:      ``[Foo / Bar] at node: <node>``
                      ``[Foo / Bar] at node: <node>; tool calls: [a, b]``

The path label is built lazily from the ``description`` values received in
``log_start`` calls.  Each thread ID maps to its description; the label for a
path is all descriptions joined with `` / ``.
"""

from typing import Callable, override, cast, Any, AsyncIterator
import sys
import asyncio
from contextlib import asynccontextmanager

from composer.spec.source.prover import ProverEvents
from composer.ui.autoprove_app import AutoProvePhase
from composer.io.event_handler import NullEventHandler
from composer.io.multi_job import TaskHandle, TaskInfo
from composer.io.conversation import (
    ConversationClient, ProgressPayload, AIYapping, ToolBatch, ToolComplete, ThinkingStart, StateUpdate
)
from composer.io.stream import managed_streamer, AsyncDataQueue, ManagedQueue, EndConversation, Checkpoint
from rich.console import Console, RenderableType
from rich.status import Status
from rich.markdown import Markdown

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

class _ConversationClient():
    def __init__(
        self, init_msg: RenderableType
    ):
        self.init_msg = init_msg
        self.ev_queue : ManagedQueue[ProgressPayload] = AsyncDataQueue(asyncio.Event(), [])
        self._thinking_item : Status | None = None
        self._console = Console()
        self.drain_task : asyncio.Task[None]

    def _reset_thinking(self):
        if self._thinking_item is not None:
            self._thinking_item.stop()
            self._thinking_item = None

    async def _update(
        self, r: ProgressPayload
    ):
        match r:
            case ThinkingStart():
                if self._thinking_item is None:
                    self._thinking_item = self._console.status("Thinking...")
                    self._thinking_item.start()
            case ToolComplete():
                pass
            case AIYapping():
                self._reset_thinking()
                self._console.print(r.yap_content, markup=False, style="italic dim")
            case ToolBatch():
                print(f"AI called: {", ".join([ t['name'] for t in r.calls ])}")
            case StateUpdate():
                self._reset_thinking()
                self._console.print(r.state_display, markup=False)

    def progress_update(
        self, progress: ProgressPayload
    ):
        self.ev_queue.push(progress)

    async def human_turn(
        self, ai_response: str | None
    ) -> str:
        self._reset_thinking()
        ev = asyncio.Event()
        self.ev_queue.push(Checkpoint(ev))
        await ev.wait()
        if ai_response is not None:
            self._console.print(Markdown(ai_response))
        multiline = False

        @Condition
        def is_multiline():
            return multiline

        kb = KeyBindings()

        @kb.add("c-e")  # Ctrl+E to toggle
        def _toggle(event):
            nonlocal multiline
            multiline = not multiline

        session = PromptSession()
        text = await session.prompt_async(
            ">>> ",
            multiline=is_multiline,
            key_bindings=kb,
            bottom_toolbar=lambda: HTML(
                "<b>Ctrl+E</b> multiline: <b>{}</b>{}".format(
                    "ON" if multiline else "OFF",
                    "  |  <b>Alt+Enter</b> to submit" if multiline else "",
                )
            ),
        )
        return text

    async def __aenter__(self):
        self.drain_task = managed_streamer(
            self.ev_queue, self._update
        )
        print("--- Entering refinment conversation (all other output suppressed) ---")
        self._console.print(self.init_msg)

    async def __aexit__(self, exc_type, exc, tb):
        self.ev_queue.push(EndConversation())
        try:
            await self.drain_task
        except:
            print("Conversation cleanup failed")


class AutoProveConsoleHandler(NullEventHandler):
    """``IOHandler[Never]`` + ``HandlerFactory`` for the auto-prove pipeline.

    One instance spans the whole pipeline run.  ``make_handler`` is passed as
    the ``handler_factory`` argument; it returns ``handler=self`` each time so
    path descriptions accumulated by one phase are visible to all later phases.
    """

    def __init__(self) -> None:
        self._descriptions: dict[str, str] = {}
        self._conversation_lock = asyncio.Semaphore()
        self._suppress_output = False

    def _output(self, to_print: Any):
        if self._suppress_output:
            return
        print(to_print)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self, path: list[str]) -> str:
        return " / ".join(self._descriptions.get(tid, tid) for tid in path)

    # ------------------------------------------------------------------
    # IOHandler protocol
    # ------------------------------------------------------------------

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str) -> None:
        pass  # checkpoint noise suppressed

    async def log_start(
        self, *, path: list[str], description: str, tool_id: str | None
    ) -> None:
        self._descriptions[path[-1]] = description
        label = self._label(path)
        suffix = f"  (via tool: {tool_id})" if tool_id else ""
        self._output(f"[{label}] start{suffix}")

    async def log_end(self, path: list[str]) -> None:
        self._output(f"[{self._label(path)}] end")

    async def log_state_update(self, path: list[str], st: dict) -> None:
        label = self._label(path)
        for node_name, update in st.items():
            if not isinstance(update, dict):
                continue
            tool_names: list[str] = []
            for msg in update.get("messages", []):
                tc = getattr(msg, "tool_calls", None)
                if tc:
                    tool_names.extend(c["name"] for c in tc)
            if tool_names:
                names = ", ".join(tool_names)
                self._output(f"[{label}] at node: {node_name}; tool calls: [{names}]")
            else:
                self._output(f"[{label}] at node: {node_name}")

    async def human_interaction(
        self, ty: None, debug_thunk: Callable[[], None]
    ) -> str:
        raise RuntimeError(
            "Unexpected HITL interrupt in auto-prove console handler"
        )

    @override
    def handle_event(self, payload: dict, path: list[str], checkpoint_id: str):
        d = cast(ProverEvents, payload)
        match d["type"]:
            case "prover_output":
                pass
            case "cloud_polling":
                pass
            case "prover_run":
                self._output(f"[{self._label(path)}]: prover start")
            case "prover_result":
                self._output(f"[{self._label(path)}]; prover complete")
            case "rule_analysis":
                self._output(f"[{self._label(path)}]: rule analysis complete -> {d['rule']}")
            case "cex_analysis":
                self._output(f"[{self._label(path)}]: rule analysis start -> {d['rule_name']}")
        return super().handle_event(payload, path, checkpoint_id)
    
    @asynccontextmanager
    async def _start_conversation(self, initial: RenderableType) -> AsyncIterator[ConversationClient]:
        async with self._conversation_lock:
            prev = self._suppress_output
            self._suppress_output = True
            to_yield = _ConversationClient(initial)
            async with to_yield:
                yield to_yield
            self._suppress_output = prev

    # ------------------------------------------------------------------
    # HandlerFactory
    # ------------------------------------------------------------------

    async def make_handler(self, info: TaskInfo[AutoProvePhase]) -> TaskHandle[None]:
        """Return a ``TaskHandle`` that routes all events back to *self*.

        Pass this bound method as ``handler_factory`` to
        ``run_autoprove_pipeline``.
        """
        async def _on_error(exc: Exception, tb: str) -> None:
            print(
                f"\n[ERROR] {info.label}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            print(tb, file=sys.stderr)

        return TaskHandle(
            handler=self,
            event_handler=self,
            on_start=lambda: print(
                f"\n{'─' * 60}\nPhase: {info.label}\n{'─' * 60}"
            ),
            on_done=lambda: print(f"[{info.label}] ✓ done"),
            on_error=_on_error,
            conversation_provider=self._start_conversation
        )
