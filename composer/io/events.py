"""
Event types emitted by graph execution and consumed by the drainer.

Every ``run_graph()`` call pushes events into an ``EventQueue`` via
an event sink.  The drainer unpacks them and dispatches to
``IOHandler`` (structural events) or ``EventHandler`` (custom
payloads).

When graph executions are nested, each event is wrapped in one or
more ``Nested`` envelopes carrying the parent's thread ID.  The
drainer peels these off to reconstruct the full path.
"""

import time
from dataclasses import dataclass, field


@dataclass
class StateUpdate:
    """A graph node produced new state (messages, tool results, etc.)."""
    payload: dict
    thread_id: str
    ts: float = field(default_factory=time.time)
    """``time.time()`` at emission — set when the event is pushed to the sink."""

@dataclass
class NextCheckpoint:
    """A new checkpoint was persisted."""
    thread_id: str
    checkpoint_id: str
    ts: float = field(default_factory=time.time)
    """``time.time()`` at emission — set when the event is pushed to the sink."""

@dataclass
class CustomUpdate:
    """A tool called ``get_stream_writer()`` with a domain-specific payload."""
    payload: dict
    thread_id: str
    checkpoint_id: str
    ts: float = field(default_factory=time.time)
    """``time.time()`` at emission — set when the event is pushed to the sink."""

@dataclass
class Start:
    """Graph execution began."""
    thread_id: str
    description: str
    tool_id: str | None = None
    started_at_wall: float = 0.0
    """``time.time()`` at start — for human-readable timestamps."""
    started_at_mono: float = 0.0
    """``time.perf_counter()`` at start — pair with End.duration_s for deltas."""
    ts: float = field(default_factory=time.time)
    """``time.time()`` at emission — set when the event is pushed to the sink."""

@dataclass
class End:
    """Graph execution ended (success or failure)."""
    thread_id: str
    duration_s: float = 0.0
    error: str | None = None
    """Exception class name if the graph raised; ``None`` on success."""
    ts: float = field(default_factory=time.time)
    """``time.time()`` at emission — set when the event is pushed to the sink."""

@dataclass
class ProgressEvent:
    payload: dict
    ts: float = field(default_factory=time.time)
    """``time.time()`` at emission — set when the event is pushed to the sink."""

InnerEvent = StateUpdate | NextCheckpoint | CustomUpdate | Start | End

@dataclass
class Nested:
    """Wrapper indicating the inner event originated from a nested ``run_graph()`` call.

    The drainer collects ``parent_id`` values into a path list so
    handlers know which nested execution produced the event.
    """
    inner: "GraphEvents"
    parent_id: str

type GraphEvents = InnerEvent | Nested

type AllEvents = GraphEvents | ProgressEvent
