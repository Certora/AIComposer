"""
Custom stream writer event types for the NatSpec pipeline.

Emitted by tools via ``get_stream_writer()`` and routed through
``EventHandler.handle_event()`` to the TUI.
"""

from typing import Annotated, Literal, TypedDict

from pydantic import Discriminator


class MasterSpecUpdate(TypedDict):
    type: Literal["master_spec_update"]
    spec: str


class StubUpdate(TypedDict):
    type: Literal["stub_update"]
    stub: str


NatspecEvent = Annotated[
    MasterSpecUpdate | StubUpdate,
    Discriminator("type"),
]
