"""Typed event model for workflow execution buffers.

All workflow execution writes events to a ``JobEventBuffer``.  Display
layers (``GraphRunnerApp``, ``JobDisplay``) consume these events and
project them onto widgets.  This module defines the event vocabulary
and a small DOM-ID helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ---------------------------------------------------------------------------
# DOM ID helper — centralises the LangGraph-ID → Textual-widget-ID mapping
# ---------------------------------------------------------------------------

TOOL_CALL = "tool-call"
PROVER_OUTPUT = "prover-output"
SILENT_STATUS = "silent"
JOB_DETAIL = "job-detail"


def dom_id(kind: str, key: str) -> str:
    """Build a Textual widget ID from a semantic *kind* and a key string."""
    return f"{kind}-{key}"


# ---------------------------------------------------------------------------
# Buffer events
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CheckpointEvt:
    """A LangGraph checkpoint was persisted."""
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class NodeUpdateEvt:
    """A graph node emitted a state update."""
    node_name: str
    state_update: dict


@dataclass(frozen=True, slots=True)
class ProverOutputEvt:
    """A line of prover stdout (from ``get_stream_writer()``)."""
    tool_call_id: str
    line: str


@dataclass(frozen=True, slots=True)
class CloudPollingEvt:
    """A cloud-prover polling status update."""
    tool_call_id: str
    status: str
    message: str


# -- Nested workflow lifecycle (depth 1) ------------------------------------

@dataclass(frozen=True, slots=True)
class NestedStart:
    """A depth-1 nested workflow began."""
    thread_id: str
    workflow_name: str


@dataclass(frozen=True, slots=True)
class NestedEnd:
    """A depth-1 nested workflow finished."""
    thread_id: str


@dataclass(frozen=True, slots=True)
class NestedEvt:
    """Wraps any event produced by a depth-1 nested workflow."""
    thread_id: str
    inner: BufferEvent


# -- Silent workflow lifecycle (depth >= 2) ---------------------------------

@dataclass(frozen=True, slots=True)
class SilentStart:
    """A deeply-nested (depth >= 2) workflow began."""
    thread_id: str
    workflow_name: str


@dataclass(frozen=True, slots=True)
class SilentEnd:
    """A deeply-nested (depth >= 2) workflow finished."""
    thread_id: str


# -- Errors -----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ErrorEvt:
    """An unrecoverable error occurred during workflow execution."""
    message: str
    traceback: str | None = None


# ---------------------------------------------------------------------------
# Union of all event types
# ---------------------------------------------------------------------------

BufferEvent = Union[
    CheckpointEvt,
    NodeUpdateEvt,
    ProverOutputEvt,
    CloudPollingEvt,
    NestedStart,
    NestedEnd,
    NestedEvt,
    SilentStart,
    SilentEnd,
    ErrorEvt,
]


# ---------------------------------------------------------------------------
# LangGraph custom-event parser (single translation boundary)
# ---------------------------------------------------------------------------

def parse_custom_event(payload: dict) -> ProverOutputEvt | CloudPollingEvt | None:
    """Translate a LangGraph ``stream_mode='custom'`` dict into a typed event.

    Returns ``None`` for unrecognised event types.
    """
    match payload.get("type"):
        case "prover_output":
            return ProverOutputEvt(
                tool_call_id=payload["tool_call_id"],
                line=payload["line"],
            )
        case "cloud_polling":
            return CloudPollingEvt(
                tool_call_id=payload["tool_call_id"],
                status=payload["status"],
                message=payload["message"],
            )
        case _:
            return None
