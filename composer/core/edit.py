"""In-memory buffer editing primitive shared by spec-authoring workflows.

The CVL author and the foundry test author each keep a single text buffer in
graph state and, today, rewrite it wholesale on every change — an enormous
waste of output tokens when the agent only needs to touch one line of a failing
rule. This module is the one genuinely shared piece of a surgical-edit tool: a
pure single-occurrence string replacement with precise, LLM-actionable failure
messages.

It is deliberately *only* the string operation. Each workflow wraps it in its
own tool — its own tool-definition idiom, its own state buffer field, and its
own (optional) post-edit validator — because those differ between consumers and
a factory spanning them would be more machinery than it removes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EditOk:
    """A successful edit: ``text`` is the new buffer contents."""

    text: str


@dataclass(frozen=True)
class EditErr:
    """A failed edit. ``message`` is phrased to be returned to the LLM."""

    message: str


def replace_unique(buffer: str, old: str, new: str) -> EditOk | EditErr:
    """Replace the single occurrence of ``old`` in ``buffer`` with ``new``.

    An edit must identify exactly one site. Returns :class:`EditErr` when
    ``old`` is absent or matches more than once (telling the caller to add
    surrounding context) rather than risk silently editing the wrong place.
    On success returns :class:`EditOk` with the rewritten buffer.
    """
    count = buffer.count(old)
    if count == 0:
        return EditErr(
            "`old_string` was not found in the current buffer. It must match an "
            "exact span of the contents, including whitespace and indentation. "
            "Read the buffer back and copy the target span verbatim."
        )
    if count > 1:
        return EditErr(
            f"`old_string` matched {count} locations; it must match exactly one. "
            "Include more surrounding context so the target span is unique."
        )
    return EditOk(buffer.replace(old, new, 1))
