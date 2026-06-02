"""Lightweight helpers for rendering ``BaseMessage.content``.

Lives outside ``composer.diagnostics.handlers`` so renderers can import it
without dragging in the diagnostics + core + graphcore chain (which
transitively loads langgraph.prebuilt → langchain_core.language_models.base
→ transformers/torch).
"""


def normalize_content(s: str | list[str | dict]) -> list[dict]:
    """Coerce a ``BaseMessage.content`` value into a uniform list of blocks.

    Anthropic-shape content can be either a bare string or a list of blocks
    where each block is either a string (treated as a text block) or a dict
    (any block type — text, thinking, tool_use, etc.). This normalizes both
    forms to ``list[dict]`` so renderers can ``match`` on ``block["type"]``.
    """
    l: list[str | dict]
    if isinstance(s, str):
        l = [s]
    else:
        l = s
    to_ret: list[dict] = []
    for r in l:
        if isinstance(r, str):
            to_ret.append({"type": "text", "text": r})
        else:
            to_ret.append(r)
    return to_ret
