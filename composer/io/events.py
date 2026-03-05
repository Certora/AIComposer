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

from dataclasses import dataclass


@dataclass
class StateUpdate:
    """A graph node produced new state (messages, tool results, etc.)."""
    payload: dict
    thread_id: str

@dataclass
class NextCheckpoint:
    """A new checkpoint was persisted."""
    thread_id: str
    checkpoint_id: str

@dataclass
class CustomUpdate:
    """A tool called ``get_stream_writer()`` with a domain-specific payload."""
    payload: dict
    thread_id: str
    checkpoint_id: str

@dataclass
class Start:
    """Graph execution began."""
    thread_id: str
    description: str
    tool_id: str | None = None

@dataclass
class End:
    """Graph execution ended (success or failure)."""
    thread_id: str

@dataclass
class ToolOutput:
    """Streaming output line from a tool (e.g. subprocess stdout)."""
    tool_id: str
    output_line: str

InnerEvent = StateUpdate | NextCheckpoint | CustomUpdate | Start | End | ToolOutput

@dataclass
class Nested:
    """Wrapper indicating the inner event originated from a nested ``run_graph()`` call.

    The drainer collects ``parent_id`` values into a path list so
    handlers know which nested execution produced the event.
    """
    inner: "AllEvents"
    parent_id: str

type AllEvents = InnerEvent | Nested
