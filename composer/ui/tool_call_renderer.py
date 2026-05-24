"""
Shared tool-call rendering logic.

``ToolCallRenderer`` holds the state needed to render a sequence of
tool calls with consecutive-group collapsing driven by
``ToolDisplayConfig``.  Two subclasses use it:

- ``MessageRenderer`` — full LangGraph stream rendering (AI turns,
  tool results, human/system messages).
- ``_ConversationRenderer`` — lightweight refinement-conversation
  rendering (only AI yapping, tool calls, thinking spinner).

Tool *result* rendering is intentionally not in this base — results are
a ``MessageRenderer`` concern only.
"""

import time

from textual.timer import Timer
from textual.widgets import Static

from rich.text import Text

from composer.ui.tool_display import ToolDisplayConfig


_DOT = "\u25cf "  # ● filled circle


def _dot(style: str, text: Text | str) -> Text:
    if isinstance(text, str):
        text = Text(text)
    result = Text()
    result.append(_DOT, style=style)
    result.append_text(text)
    return result


def format_elapsed(seconds: float) -> str:
    """Compact wall-time formatter: 1.2s / 2m13s / 1h05m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class _LiveToolCallStatic(Static):
    """Tool-call line that appends a live-ticking elapsed counter."""

    _TICK_INTERVAL = 1.0
    _SHOW_AFTER = 1.0  # don't tick until the call has been running this long

    def __init__(self, base_text: Text, dot_style: str = "green") -> None:
        super().__init__(_dot(dot_style, base_text))
        self._base_text = base_text
        self._dot_style = dot_style
        self._start = time.perf_counter()
        self._timer: Timer | None = None
        self._stopped = False

    def on_mount(self) -> None:
        self._timer = self.set_interval(self._TICK_INTERVAL, self._tick)

    def _update_label(self, suffix: str | None) -> None:
        text = Text()
        text.append_text(self._base_text)
        if suffix:
            text.append(suffix, style="dim yellow")
        self.update(_dot(self._dot_style, text))

    def _tick(self) -> None:
        if self._stopped:
            return
        elapsed = time.perf_counter() - self._start
        if elapsed < self._SHOW_AFTER:
            return
        self._update_label(f" (running {format_elapsed(elapsed)})")

    def stop(self, elapsed: float | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        final = elapsed if elapsed is not None else time.perf_counter() - self._start
        self._update_label(f" ({format_elapsed(final)})")


class ToolCallRenderer:
    """Produces widgets for tool calls, collapsing consecutive grouped calls.

    Maintains per-stream state:

    - ``_last_tool_group`` / ``_last_tool_items`` / ``_last_tool_widget``
      track the currently-open group so repeated calls to the same group
      fold into one in-place-updated ``Static``.
    - ``_tool_call_anchors`` maps ``tool_call_id`` to the widget that
      represents it, so callers can attach live output panes later.

    Subclasses must call ``reset_tool_collapsing()`` at dialogue-turn
    boundaries (e.g. on human input) to prevent grouping across turns.
    """

    def __init__(self, tool_config: ToolDisplayConfig):
        self.tool_config = tool_config

        self._last_tool_group: str | None = None
        self._last_tool_items: list[str] = []
        self._last_tool_widget: Static | None = None

        self._tool_call_anchors: dict[str, Static] = {}
        self._tool_call_starts: dict[str, float] = {}

    def render_tool_call(
        self, name: str, input_args: dict, tool_call_id: str | None
    ) -> Static | None:
        """Render a tool call.

        Returns the widget that should be mounted, or ``None`` when the
        call folded into an already-mounted group widget (updated
        in-place).
        """
        if tool_call_id is not None:
            self._tool_call_starts[tool_call_id] = time.perf_counter()
        tc = self.tool_config
        grouped = tc.get_group(name)

        if grouped is not None:
            raw = grouped.extract_group_items(input_args)
            new_items = [raw] if isinstance(raw, str) else list(raw)

            if grouped.group_id == self._last_tool_group:
                self._last_tool_items.extend(new_items)
                new_text = grouped.render_group(self._last_tool_items)
                if self._last_tool_widget is not None:
                    self._last_tool_widget.update(_dot("green", Text(new_text, style="dim")))
                if tool_call_id is not None and self._last_tool_widget is not None:
                    self._tool_call_anchors[tool_call_id] = self._last_tool_widget
                return None

            self._last_tool_group = grouped.group_id
            self._last_tool_items = new_items
            display_text = grouped.render_group(self._last_tool_items)
            w = Static(_dot("green", Text(display_text, style="dim")))
            self._last_tool_widget = w
            if tool_call_id is not None:
                self._tool_call_anchors[tool_call_id] = w
            return w

        self.reset_tool_collapsing()
        friendly = tc.format_tool_call(name, input_args)
        w = _LiveToolCallStatic(Text(friendly, style="dim"))
        if tool_call_id is not None:
            self._tool_call_anchors[tool_call_id] = w
        return w

    def reset_tool_collapsing(self) -> None:
        """Reset consecutive tool-call collapsing state.

        Call at dialogue turn boundaries, or when mounting a non-tool
        widget that should visually break the group (e.g. a human
        message).
        """
        self._last_tool_group = None
        self._last_tool_items = []
        self._last_tool_widget = None

    def get_tool_call_anchor(self, tool_call_id: str) -> Static | None:
        return self._tool_call_anchors.get(tool_call_id)

    def pop_tool_call_elapsed(self, tool_call_id: str | None) -> float | None:
        """Return seconds since `render_tool_call` for *tool_call_id*, removing the start.

        Also stops the live-ticking widget for that call, freezing its label
        at the final elapsed time.
        """
        if tool_call_id is None:
            return None
        t0 = self._tool_call_starts.pop(tool_call_id, None)
        if t0 is None:
            return None
        elapsed = time.perf_counter() - t0
        widget = self._tool_call_anchors.get(tool_call_id)
        if isinstance(widget, _LiveToolCallStatic):
            widget.stop(elapsed)
        return elapsed
