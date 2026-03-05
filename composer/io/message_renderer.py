"""
Extracted message rendering logic shared by BaseRichConsoleApp and PipelineTaskHandler.

``MessageRenderer`` holds per-stream rendering state (tool collapsing, nested
containers) and exposes widget-producing methods.

``TokenStats`` accumulates token usage from AI messages and updates a display widget.
"""

from textual.containers import VerticalScroll
from textual.widgets import Static, Collapsible

from rich.text import Text

from langchain_core.messages import AIMessage, ToolMessage

from composer.diagnostics.handlers import normalize_content
from composer.io.tool_display import ToolDisplayConfig

from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE

KNOWN_NODES: set[str] = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

_DOT = "\u25cf "  # ● filled circle


def dot(style: str, text: Text | str) -> Text:
    """Prepend a colored dot to a Text or string for visual structure."""
    if isinstance(text, str):
        text = Text(text)
    result = Text()
    result.append(_DOT, style=style)
    result.append_text(text)
    return result


class TokenStats:
    """Accumulates token usage across AI messages and updates a display widget."""

    def __init__(self, display: Static):
        self._display = display
        self.input: int = 0
        self.output: int = 0
        self.cache_read: int = 0
        self.cache_write: int = 0

    def update(self, msg: AIMessage) -> None:
        """Extract usage from the message and refresh the display widget."""
        if not isinstance(msg.response_metadata, dict):
            return
        usage = msg.response_metadata.get("usage")
        if usage is None:
            return
        self.input += usage.get("input_tokens", 0)
        self.output += usage.get("output_tokens", 0)
        self.cache_read += usage.get("cache_read_input_tokens", 0)
        self.cache_write += usage.get("cache_creation_input_tokens", 0)
        self._display.update(
            f"in:{self.input:,} out:{self.output:,} "
            f"cache_read:{self.cache_read:,} cache_write:{self.cache_write:,}"
        )


class MessageRenderer:
    """Per-stream rendering state and widget-producing methods.

    Used by both ``BaseRichConsoleApp`` (single-stream) and
    ``PipelineTaskHandler`` (per-task stream).
    """

    def __init__(self, tool_config: ToolDisplayConfig):
        self.tool_config = tool_config
        self.nested_containers: dict[str, VerticalScroll] = {}

        # Consecutive tool call collapsing state
        self._last_tool_group: str | None = None
        self._last_tool_items: list[str] = []
        self._last_tool_widget: Static | None = None

    def render_ai_turn(self, msg: AIMessage) -> list[Static | Collapsible]:
        """Render an AI turn as a list of widgets."""
        widgets: list[Static | Collapsible] = []
        tc = self.tool_config

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
                        widgets.append(Static(dot("blue", text)))
                case "tool_use":
                    name = c["name"]
                    input_args = c.get("input", {})
                    grouped = tc.get_group(name)

                    if grouped is not None:
                        raw = grouped.extract_group_items(input_args)
                        new_items = [raw] if isinstance(raw, str) else list(raw)

                        if grouped.group_id == self._last_tool_group:
                            # Same group — update existing widget
                            self._last_tool_items.extend(new_items)
                            new_text = grouped.render_group(self._last_tool_items)
                            if self._last_tool_widget is not None:
                                self._last_tool_widget.update(dot("green", Text(new_text, style="dim")))
                        else:
                            # New group
                            self._last_tool_group = grouped.group_id
                            self._last_tool_items = new_items
                            display_text = grouped.render_group(self._last_tool_items)
                            w = Static(dot("green", Text(display_text, style="dim")))
                            self._last_tool_widget = w
                            widgets.append(w)
                    else:
                        # Non-grouped tool — reset and emit standalone
                        self.reset_tool_collapsing()
                        friendly = tc.format_tool_call(name, input_args)
                        widgets.append(Static(dot("green", Text(friendly, style="dim"))))
                case other:
                    widgets.append(Static(f"Unknown block: {other}"))

        return widgets

    def render_tool_result(self, msg: ToolMessage) -> Collapsible | None:
        """Render a tool result as a collapsible, or ``None`` to suppress."""
        name = getattr(msg, "name", None) or "Tool result"
        result_info = self.tool_config.format_result(name, msg)
        if result_info is None:
            return None
        self.reset_tool_collapsing()
        label, body = result_info
        return Collapsible(Static(body, markup=False), title=label, collapsed=True)

    def reset_tool_collapsing(self):
        """Reset consecutive tool call collapsing state."""
        self._last_tool_group = None
        self._last_tool_items = []
        self._last_tool_widget = None

    def get_mount_target(self, root: VerticalScroll, path: list[str]) -> VerticalScroll:
        """Resolve the mount target for a given path.

        If the path references a nested container, returns it; otherwise
        falls back to ``root``.
        """
        if len(path) > 1 and path[-1] in self.nested_containers:
            return self.nested_containers[path[-1]]
        return root
