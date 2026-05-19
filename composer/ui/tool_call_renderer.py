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

    def render_tool_call(
        self, name: str, input_args: dict, tool_call_id: str | None
    ) -> Static | None:
        """Render a tool call.

        Returns the widget that should be mounted, or ``None`` when the
        call folded into an already-mounted group widget (updated
        in-place).
        """
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
        w = Static(_dot("green", Text(friendly, style="dim")))
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
