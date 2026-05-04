"""
Shared tool-call rendering primitives.

The single state-bearing piece is :class:`ToolGroupState` — the open
consecutive-tool-call group. Callers own one or more instances of it
(one per render stream) and pass it into :func:`render_tool_call`,
which mutates it in place to fold consecutive grouped calls into a
single in-place-updated widget.

Tool-call *anchors* (the ``tool_call_id -> Static`` widget map used by
``CodeGenRichApp`` to attach prover output panes to a specific tool
call) are a flat dict owned by the caller. Tool call ids are LLM-
minted UUIDs and treated as globally unique, so a single flat map is
sufficient even with multiple concurrent threads.

Tool *result* rendering is intentionally not here — results are a
``MessageRenderer`` concern only.
"""

from dataclasses import dataclass, field

from textual.widgets import Static

from rich.text import Text

from composer.ui.tool_display import ToolDisplayConfig


_DOT = "● "  # ● filled circle


def _dot(style: str, text: Text | str) -> Text:
    if isinstance(text, str):
        text = Text(text)
    result = Text()
    result.append(_DOT, style=style)
    result.append_text(text)
    return result


@dataclass
class ToolGroupState:
    """The currently-open consecutive-tool-call group within one render stream.

    Reset at dialogue-turn boundaries (e.g. on human input) or whenever
    a non-tool widget should visually break the group.
    """
    group_id: str | None = None
    items: list[str] = field(default_factory=list)
    widget: Static | None = None

    def reset(self) -> None:
        self.group_id, self.items, self.widget = None, [], None


def render_tool_call(
    tool_config: ToolDisplayConfig,
    group: ToolGroupState,
    *,
    name: str,
    input_args: dict,
    tool_call_id: str | None,
    anchors: dict[str, Static],
) -> Static | None:
    """Render a tool call into a widget.

    Mutates ``group`` for consecutive-call collapsing and writes
    ``tool_call_id -> widget`` into ``anchors``. Returns ``None`` when
    the call folded into an already-mounted group widget (the existing
    widget was updated in place).
    """
    grouped = tool_config.get_group(name)

    if grouped is not None:
        raw = grouped.extract_group_items(input_args)
        new_items = [raw] if isinstance(raw, str) else list(raw)

        if grouped.group_id == group.group_id:
            group.items.extend(new_items)
            new_text = grouped.render_group(group.items)
            if group.widget is not None:
                group.widget.update(_dot("green", Text(new_text, style="dim")))
            if tool_call_id is not None and group.widget is not None:
                anchors[tool_call_id] = group.widget
            return None

        group.group_id = grouped.group_id
        group.items = new_items
        display_text = grouped.render_group(group.items)
        w = Static(_dot("green", Text(display_text, style="dim")))
        group.widget = w
        if tool_call_id is not None:
            anchors[tool_call_id] = w
        return w

    group.reset()
    friendly = tool_config.format_tool_call(name, input_args)
    w = Static(_dot("green", Text(friendly, style="dim")))
    if tool_call_id is not None:
        anchors[tool_call_id] = w
    return w
