"""
Extension point for domain-specific custom events.

When an agent tool calls ``get_stream_writer()(payload)``, the
payload arrives at ``EventHandler.handle_event()`` via the
``CustomUpdate`` event path.  Implementations cast the untyped
dict to a domain-specific discriminated union for type-safe
dispatch.

``IOHandler`` handles structural events (start/end, state,
checkpoints).  ``EventHandler`` handles everything else — it is
the seam where workflows inject their own event vocabulary without
modifying the core dispatch logic.
"""

from typing import Protocol


class EventHandler(Protocol):
    """Receives custom event payloads emitted by agent tools via ``get_stream_writer()``.

    ``path`` is the list of thread IDs from outermost to innermost,
    identifying which (possibly nested) graph execution emitted the
    event.
    """
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None: ...

    async def handle_progress_event(self, payload: dict) -> None: ...


class NullEventHandler:
    """No-op handler — ignores all custom events."""
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        pass

    async def handle_progress_event(self, payload: dict) -> None:
        pass
