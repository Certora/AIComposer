from dataclasses import dataclass


@dataclass
class StateUpdate:
    payload: dict
    thread_id: str

@dataclass
class NextCheckpoint:
    thread_id: str
    checkpoint_id: str

@dataclass
class CustomUpdate:
    payload: dict
    thread_id: str
    checkpoint_id: str

@dataclass
class Start:
    thread_id: str
    description: str
    tool_id: str | None = None

@dataclass
class End:
    thread_id: str

@dataclass
class ToolOutput:
    tool_id: str
    output_line: str

InnerEvent = StateUpdate | NextCheckpoint | CustomUpdate | Start | End | ToolOutput

@dataclass
class Nested:
    inner: "AllEvents"
    parent_id: str

type AllEvents = InnerEvent | Nested
