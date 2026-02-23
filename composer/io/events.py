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
class NestedStart:
    thread_id: str
    child_thread_id: str
    tool_context_id: str | None

@dataclass
class NestedEnd:
    thread_id: str
    child_thread_id: str

@dataclass
class ToolOutput:
    tool_id: str
    output_line: str

type AllEvents = StateUpdate | NextCheckpoint | CustomUpdate | NestedStart | NestedEnd | ToolOutput
