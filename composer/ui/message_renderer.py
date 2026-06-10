"""
Extracted message rendering logic shared by BaseRichConsoleApp and MultiJobTaskHandler.

``MessageRenderer`` holds per-stream rendering state (tool collapsing, nested
containers) and exposes both widget-producing and widget-mounting methods.

``TokenStats`` accumulates token usage from AI messages and updates a display widget.
"""

from dataclasses import dataclass
from typing import Callable, Protocol

from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Collapsible

from rich.text import Text

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from composer.ui.content import normalize_content
from composer.ui.tool_display import ToolDisplayConfig
from composer.ui.tool_call_renderer import ToolCallRenderer

from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE
from graphcore.utils import NormalizedTokenUsage, get_normalized_token_usage

KNOWN_NODES: set[str] = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

import logging
logger = logging.getLogger(__name__)

_DOT = "\u25cf "  # ● filled circle

def dot(style: str, text: Text | str) -> Text:
    """Prepend a colored dot to a Text or string for visual structure."""
    if isinstance(text, str):
        text = Text(text)
    result = Text()
    result.append(_DOT, style=style)
    result.append_text(text)
    return result


@dataclass(frozen=True)
class _PriceTier:
    """Per-million-token prices in USD for one model at one
    context tier.

    ``input`` is the price for *fresh* input tokens (the bucket left
    after subtracting ``cache_read`` and ``cache_write`` from the
    total). ``output`` is the price for output tokens, which on the
    OpenAI side already includes reasoning tokens (billed at the
    output rate by both providers). ``cache_read`` / ``cache_write``
    are the cache-bucket rates.

    Anthropic-specific note: the ``cache_write`` rate here is the
    5-minute ephemeral rate. Anthropic also has a 1-hour ephemeral
    rate that's higher (~2× base input); our normalized usage rolls
    both into a single ``cache_write_tokens`` bucket, so workloads
    that heavily use 1h caching will be slightly under-billed by
    this banner. Approximation is fine for a banner."""
    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class _ModelPricing:
    """Pricing entry for one model family. ``long`` is the
    long-context tier (used when input token count exceeds the
    threshold) and applies only to OpenAI models that publish a
    separate >272K-input rate; Anthropic models keep ``long = None``
    and bill everything at ``short`` rates."""
    short: _PriceTier
    long: _PriceTier | None = None


# OpenAI's published >272K input-token threshold for long-context
# pricing. Once an individual call's input crosses this, the long
# tier applies *for the full session* per OpenAI's terms; we
# approximate that by switching on a per-message basis (a session
# that drifts above 272K will mostly stay there).
_OPENAI_LONG_CONTEXT_THRESHOLD = 272_000


# Pricing tables transcribed from Anthropic + OpenAI rate cards.
# Sources should be re-checked when new model families ship.
_PRICING: list[tuple[str, _ModelPricing]] = [
    # ---- Anthropic ----
    # claude-opus-4.5 / 4.6 / 4.7 share a rate card; older 4 / 4.1
    # are pricier. Matching by prefix-of-prefix so "claude-opus-4-7"
    # and "claude-opus-4-7-20260301" both hit the right entry.
    ("claude-opus-4-7", _ModelPricing(short=_PriceTier(5.00, 25.00, 0.50, 6.25))),
    ("claude-opus-4-6", _ModelPricing(short=_PriceTier(5.00, 25.00, 0.50, 6.25))),
    ("claude-opus-4-5", _ModelPricing(short=_PriceTier(5.00, 25.00, 0.50, 6.25))),
    ("claude-opus-4-1", _ModelPricing(short=_PriceTier(15.00, 75.00, 1.50, 18.75))),
    ("claude-opus-4",   _ModelPricing(short=_PriceTier(15.00, 75.00, 1.50, 18.75))),

    ("claude-sonnet-4-6", _ModelPricing(short=_PriceTier(3.00, 15.00, 0.30, 3.75))),
    ("claude-sonnet-4-5", _ModelPricing(short=_PriceTier(3.00, 15.00, 0.30, 3.75))),
    ("claude-sonnet-4",   _ModelPricing(short=_PriceTier(3.00, 15.00, 0.30, 3.75))),

    ("claude-haiku-4-5", _ModelPricing(short=_PriceTier(1.00, 5.00, 0.10, 1.25))),

    # ---- OpenAI ----
    # gpt-5.5 / 5.4 publish short (≤272K input) and long (>272K) tiers.
    # Pro variants don't publish a cached-in discount (cache_read =
    # base input). Mini/nano don't publish a long tier; we use short
    # for everything on those.
    ("gpt-5.5-pro", _ModelPricing(
        short=_PriceTier(30.00, 180.00, 30.00, 30.00),
        long=_PriceTier(60.00, 270.00, 60.00, 60.00),
    )),
    ("gpt-5.5", _ModelPricing(
        short=_PriceTier(5.00, 30.00, 0.50, 5.00),
        long=_PriceTier(10.00, 45.00, 1.00, 10.00),
    )),
    ("gpt-5.4-pro", _ModelPricing(
        short=_PriceTier(30.00, 180.00, 30.00, 30.00),
        long=_PriceTier(60.00, 270.00, 60.00, 60.00),
    )),
    ("gpt-5.4-mini", _ModelPricing(short=_PriceTier(0.75, 4.50, 0.075, 0.75))),
    ("gpt-5.4-nano", _ModelPricing(short=_PriceTier(0.20, 1.25, 0.02, 0.20))),
    ("gpt-5.4", _ModelPricing(
        short=_PriceTier(2.50, 15.00, 0.25, 2.50),
        long=_PriceTier(5.00, 22.50, 0.50, 5.00),
    )),
]


def _price_per_mtok(model: str | None, input_tokens: int) -> _PriceTier | None:
    """Look up per-MTok pricing by model name and call size. Returns
    ``None`` for models with no table entry (cost contribution becomes
    zero — better than guessing).

    Matched by prefix on the lowercased model name so dated revisions
    (``claude-opus-4-7-20260301``, ``gpt-5.5-2026-...``) collapse into
    the same family entry. Table is searched in order, so list more
    specific prefixes (``gpt-5.5-pro``) before less specific
    (``gpt-5.5``). For OpenAI models with a long tier, ``input_tokens``
    chooses short vs. long; Anthropic always uses the short tier."""
    if model is None:
        return None
    m = model.lower()
    for prefix, pricing in _PRICING:
        if m.startswith(prefix):
            if pricing.long is not None and input_tokens > _OPENAI_LONG_CONTEXT_THRESHOLD:
                return pricing.long
            return pricing.short
    return None


class TokenStats:
    """Accumulates token usage across AI messages and updates a display widget.

    Cost is computed per-message against the model that produced
    each one (so mixed-model workflows are billed correctly) and
    accumulated. Counts use the normalized usage shape from
    ``graphcore.utils.get_normalized_token_usage`` — token totals
    already include their sub-bucket breakdowns (cache_read +
    cache_write counted within ``total_input_tokens``; thinking
    counted within ``total_output_tokens``), so the cost calc
    subtracts the cached portions from input before pricing
    instead of summing them on top."""

    def __init__(self, display: Static):
        self._display = display
        self.input: int = 0   # grand-total input (includes cache buckets)
        self.output: int = 0  # grand-total output (includes thinking on OpenAI)
        self.cache_read: int = 0
        self.cache_write: int = 0
        self.thinking: int = 0
        self.cost: float = 0.0
        self._last_model: str | None = None

    @staticmethod
    def _cost_of(usage: NormalizedTokenUsage) -> float:
        price = _price_per_mtok(usage["model_name"], usage["total_input_tokens"])
        if price is None:
            return 0.0
        # Fresh (non-cached) input is the total minus the cached
        # buckets. langchain rolls them into ``total_input_tokens``.
        fresh_input = max(
            0,
            usage["total_input_tokens"]
            - usage["cache_read_tokens"]
            - usage["cache_write_tokens"],
        )
        return (
            fresh_input * price.input
            + usage["total_output_tokens"] * price.output
            + usage["cache_read_tokens"] * price.cache_read
            + usage["cache_write_tokens"] * price.cache_write
        ) / 1_000_000

    def update(self, msg: AIMessage) -> None:
        """Extract usage from the message, accrue cost, refresh the display."""
        usage = get_normalized_token_usage(msg)
        self.input += usage["total_input_tokens"]
        self.output += usage["total_output_tokens"]
        self.cache_read += usage["cache_read_tokens"]
        self.cache_write += usage["cache_write_tokens"]
        self.thinking += usage["thinking_tokens"]
        self.cost += self._cost_of(usage)
        if usage["model_name"] is not None:
            self._last_model = usage["model_name"]

        model_tag = f"[{self._last_model}] " if self._last_model else ""
        think_tag = f" think:{self.thinking:,}" if self.thinking else ""
        self._display.update(
            f"{model_tag}in:{self.input:,} out:{self.output:,}{think_tag} "
            f"cache_read:{self.cache_read:,} cache_write:{self.cache_write:,} "
            f"| ${self.cost:.2f}"
        )


class MountFn(Protocol):
    """Callback for mounting widgets into a scrollable container."""
    async def __call__(self, target: VerticalScroll, *widgets: Widget) -> None: ...


_HUMAN_TAG_DISPLAY: dict[str, tuple[str, bool]] = {
    "initial_prompt": ("Initial prompt", True),
    "resume": ("Resume context", True),
    "summarization": ("Summarization", True),
    "scolding": ("System correction", True),
    "prover_summary": ("Prover violation summary", False),
}


class MessageRenderer(ToolCallRenderer):
    """Per-stream rendering state, widget production, and mounting.

    Used by both ``BaseRichConsoleApp`` (single-stream) and
    ``MultiJobTaskHandler`` (per-task stream).
    """

    def __init__(
        self,
        tool_config: ToolDisplayConfig,
        mount_to: MountFn,
        on_tokens: Callable[[AIMessage], None],
    ):
        super().__init__(tool_config)
        self._mount_to = mount_to
        self._on_tokens = on_tokens
        self.nested_containers: dict[str, VerticalScroll] = {}

    def render_ai_turn(self, msg: AIMessage) -> list[Static | Collapsible]:
        """Render an AI turn as a list of widgets.

        Content blocks (thinking / reasoning / text) come from
        ``msg.content``; tool calls come from ``msg.tool_calls`` —
        langchain's provider-normalized field. Anthropic AIMessages
        also have ``tool_use`` blocks inline in ``content`` (mirrored
        into ``tool_calls``); we skip those to avoid double-rendering."""
        widgets: list[Static | Collapsible] = []

        for c in normalize_content(msg.content):
            match c["type"]:
                case "thinking":
                    full_text = c.get("thinking", "")
                    widgets.append(
                        Collapsible(Static(full_text, markup=False), title="Thinking...", collapsed=True)
                    )
                case "reasoning":
                    # OpenAI Responses-API reasoning item. langchain's
                    # ``responses/v1`` output format renders these as
                    # ``{"type": "reasoning", "summary": [{"type":
                    # "summary_text", "text": ...}, ...], "id":
                    # "rs_..."}``. The ``encrypted_content`` field is
                    # also present (when ``include=
                    # ["reasoning.encrypted_content"]`` is set) but
                    # we don't surface it to the user — it just needs
                    # to round-trip in the messages list for the next
                    # turn.
                    summary = c.get("summary") or []
                    full_text = "\n\n".join(
                        s.get("text", "") for s in summary
                        if isinstance(s, dict) and s.get("type") == "summary_text"
                    )
                    if full_text:
                        widgets.append(
                            Collapsible(Static(full_text, markup=False), title="Reasoning...", collapsed=True)
                        )
                case "text":
                    text = c["text"]
                    if (stripped := text.strip()):
                        widgets.append(Static(dot("blue", stripped)))
                case "tool_use" | "function_call":
                    # Skip — captured by ``msg.tool_calls`` below.
                    pass
                case other:
                    widgets.append(Static(f"Unknown block: {other}"))

        for tc in msg.tool_calls:
            w = self.render_tool_call(
                name=tc["name"],
                input_args=tc.get("args", {}),
                tool_call_id=tc.get("id"),
            )
            if w is not None:
                widgets.append(w)

        # OpenAI Chat-Completions-style reasoning content lives on
        # additional_kwargs, not in the content array.
        reasoning_extra = msg.additional_kwargs.get("reasoning_content") if isinstance(msg.additional_kwargs, dict) else None
        if isinstance(reasoning_extra, str) and reasoning_extra.strip():
            widgets.insert(0, Collapsible(Static(reasoning_extra, markup=False), title="Reasoning...", collapsed=True))

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

    def get_mount_target(self, root: VerticalScroll, path: list[str]) -> VerticalScroll:
        """Resolve the mount target for a given path.

        If the path references a nested container, returns it; otherwise
        falls back to ``root``.
        """
        if len(path) > 1 and path[-1] in self.nested_containers:
            return self.nested_containers[path[-1]]
        return root

    # ── Shared rendering methods ─────────────────────────────

    def classify_human(self, m: HumanMessage) -> tuple[str, bool]:
        """Classify a human message for display. Returns (title, collapsed)."""
        tag = getattr(m, "display_tag", None)
        if tag is not None:
            return _HUMAN_TAG_DISPLAY.get(tag, ("User input", True))
        return ("User input", True)
    
    def get_flow_target(self, root: VerticalScroll, path: list[str]) -> VerticalScroll:
        # Walk from most specific to least specific: the current flow's container
        # may not exist yet (render_start creates it), so fall back to the parent's.
        if len(path) > 1 and path[-1] in self.nested_containers:
            return self.nested_containers[path[-1]]
        if len(path) > 1 and path[-2] in self.nested_containers:
            return self.nested_containers[path[-2]]
        return root

    async def render_start(self, root: VerticalScroll, *, path: list[str], description: str) -> None:
        """Render a workflow start banner or nested collapsible."""
        target = self.get_flow_target(root, path)
        if len(path) == 1:
            logger.debug("Starting top level workflow: %s", description)
            banner = Static(Text(f"━━ {description} ━━", style="bold"))
            await self._mount_to(target, banner)
        else:
            inner = VerticalScroll(classes="nested-workflow")
            coll = Collapsible(inner, title=description, collapsed=True)
            self.nested_containers[path[-1]] = inner
            await self._mount_to(target, coll)

    async def render_end(self, root: VerticalScroll, *, path: list[str]) -> None:
        """Render a workflow end banner or collapse a nested workflow."""
        if len(path) == 1:
            target = self.get_mount_target(root, path)
            banner = Static(Text("━━ end ━━", style="bold dim"))
            await self._mount_to(target, banner)
        else:
            tid = path[-1]
            if tid in self.nested_containers:
                container = self.nested_containers.pop(tid)
                parent_coll = container.parent
                if isinstance(parent_coll, Collapsible):
                    parent_coll.collapsed = True

    async def render_messages(self, target: VerticalScroll, messages: list) -> None:
        """Render a list of LangChain messages, mounting widgets to *target*."""
        for m in messages:
            match m:
                case AIMessage():
                    widgets = self.render_ai_turn(m)
                    if widgets:
                        await self._mount_to(target, *widgets)
                    self._on_tokens(m)
                case SystemMessage():
                    self.reset_tool_collapsing()
                    coll = Collapsible(Static(m.text, markup=False), title="System prompt", collapsed=True)
                    await self._mount_to(target, coll)
                case HumanMessage():
                    self.reset_tool_collapsing()
                    title, collapsed = self.classify_human(m)
                    content = m.text
                    coll = Collapsible(Static(content, markup=False), title=title, collapsed=collapsed)
                    await self._mount_to(target, coll)
                case ToolMessage():
                    coll = self.render_tool_result(m)
                    if coll is None:
                        continue
                    await self._mount_to(target, coll)
                case _:
                    self.reset_tool_collapsing()
                    await self._mount_to(
                        target,
                        Static(Text(f"[Message: {type(m).__name__}]", style="dim")),
                    )
