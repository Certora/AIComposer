"""Rich-console-driven HITL ``ConversationClient`` implementation.

Shared between any handler that needs an interactive refinement
conversation — :class:`AutoProveConsoleHandler` (no full-screen UI to
suspend) and :class:`AutoProveLiveHandler` (pauses its ``rich.live.Live``
region around the conversation).

The client is self-contained: owns its own ``Console``, its own
progress-event drainer, and its own ``prompt_toolkit`` session.

An optional ``status_provider`` callable lets the host surface
background activity in the prompt's bottom toolbar — re-evaluated on
every prompt-toolkit redraw, with a 500 ms refresh interval so the
status ticks even when the user isn't typing. Typical use: the
autoprove live handler passes a closure over its agent table so the
prompt shows something like ``"Background: 3 agents running"`` while
the user composes refinement input.
"""

from __future__ import annotations

import asyncio
import html
from typing import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.status import Status

from composer.io.conversation import (
    AIYapping,
    ProgressPayload,
    StateUpdate,
    ThinkingStart,
    ToolBatch,
    ToolComplete,
)
from composer.io.stream import (
    AsyncDataQueue,
    Checkpoint,
    EndConversation,
    ManagedQueue,
    managed_streamer,
)


type StatusProvider = Callable[[], str | None]
"""Returns a short one-line description of background activity to
show in the prompt's bottom toolbar, or ``None`` to omit it."""


class RichConsoleConversationClient:
    """Implements ``composer.io.conversation.ConversationClient`` over a
    ``rich.Console`` + ``prompt_toolkit`` prompt."""

    def __init__(
        self,
        init_msg: RenderableType,
        *,
        status_provider: StatusProvider | None = None,
    ) -> None:
        self.init_msg = init_msg
        self._status_provider = status_provider
        self.ev_queue: ManagedQueue[ProgressPayload] = AsyncDataQueue(
            asyncio.Event(), []
        )
        self._thinking_item: Status | None = None
        self._console = Console()
        self.drain_task: asyncio.Task[None]

    def _reset_thinking(self) -> None:
        if self._thinking_item is not None:
            self._thinking_item.stop()
            self._thinking_item = None

    async def _update(self, r: ProgressPayload) -> None:
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
                print(f"AI called: {", ".join([t['name'] for t in r.calls])}")
            case StateUpdate():
                self._reset_thinking()
                self._console.print(r.state_display, markup=False)

    def progress_update(self, progress: ProgressPayload) -> None:
        self.ev_queue.push(progress)

    async def human_turn(self, ai_response: str | None) -> str:
        self._reset_thinking()
        ev = asyncio.Event()
        self.ev_queue.push(Checkpoint(ev))
        await ev.wait()
        if ai_response is not None:
            self._console.print(Markdown(ai_response))
        multiline = False

        @Condition
        def is_multiline() -> bool:
            return multiline

        kb = KeyBindings()

        @kb.add("c-e")  # Ctrl+E to toggle
        def _toggle(event) -> None:
            nonlocal multiline
            multiline = not multiline

        def _toolbar() -> HTML:
            parts: list[str] = []
            if self._status_provider is not None:
                bg = self._status_provider()
                if bg:
                    # status is host-supplied free text — escape so any
                    # stray ``<`` doesn't trip prompt-toolkit's mini-HTML.
                    parts.append(
                        f"<style fg='ansibrightblack'>{html.escape(bg)}</style>"
                    )
            parts.append(
                "<b>Ctrl+E</b> multiline: <b>{}</b>{}".format(
                    "ON" if multiline else "OFF",
                    "  |  <b>Alt+Enter</b> to submit" if multiline else "",
                )
            )
            return HTML("\n".join(parts))

        session: PromptSession[str] = PromptSession()
        text = await session.prompt_async(
            ">>> ",
            multiline=is_multiline,
            key_bindings=kb,
            bottom_toolbar=_toolbar,
            # Re-evaluate the toolbar twice a second so a periodic
            # ``status_provider`` (e.g. agent counters) appears to tick
            # in real time without depending on user keystrokes.
            refresh_interval=0.5,
        )
        return text

    async def __aenter__(self) -> "RichConsoleConversationClient":
        self.drain_task = managed_streamer(self.ev_queue, self._update)
        print("--- Entering refinement conversation (all other output suppressed) ---")
        self._console.print(self.init_msg)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.ev_queue.push(EndConversation())
        try:
            await self.drain_task
        except Exception:
            print("Conversation cleanup failed")
