"""Provider-kind derivation from a model name.

Single source of truth for "are we talking to Anthropic or to an
OpenAI-compatible endpoint." Derive once at every workflow entry point
via :func:`provider_for` and pass the result through; do not scatter
``isinstance(llm, ChatAnthropic)`` checks.

Extend the matcher in :func:`provider_for` when a new model family
appears.
"""

from typing import Literal

type ProviderKind = Literal["anthropic", "openai"]


_ANTHROPIC_PREFIXES = ("claude-",)
_OPENAI_PREFIXES = ("gpt-", "chatgpt-", "o1-", "o2-", "o3-", "o4-", "o5-")


def provider_for(model: str) -> ProviderKind:
    """Map a model identifier to its provider family.

    ``claude-*`` is Anthropic; ``gpt-*`` / ``o[1-5]-*`` / ``chatgpt-*``
    is OpenAI (which also covers OpenAI-compatible endpoints whose
    surface mirrors OpenAI's). Anything else raises ``ValueError`` —
    extend the prefix tables here when a new family appears.
    """
    lowered = model.lower()
    if lowered.startswith(_ANTHROPIC_PREFIXES):
        return "anthropic"
    if lowered.startswith(_OPENAI_PREFIXES):
        return "openai"
    raise ValueError(
        f"Unrecognized model {model!r}: expected a name starting with one of "
        f"{_ANTHROPIC_PREFIXES + _OPENAI_PREFIXES}. Extend "
        f"composer.workflow.provider.provider_for when adding a new family."
    )
