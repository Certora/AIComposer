"""
JSONL serialization for graph events.

Renders each ``GraphEvents`` instance as a single-line JSON object suitable
for streaming to the rotating ``composer.events`` logger.

The serializer is best-effort: any unserializable payload is logged with
``kind=serialize_error`` rather than crashing the drainer.
"""

import json
import logging
from typing import Any, NotRequired, TypedDict

from composer.diagnostics.logging_setup import EVENTS_LOGGER_NAME
from composer.io.events import (
    Start, End, StateUpdate, NextCheckpoint, CustomUpdate, ProgressEvent, InnerEvent,
)


_events_logger = logging.getLogger(EVENTS_LOGGER_NAME)


class _StateEntry(TypedDict):
    node: str
    tool_calls: NotRequired[list[str]]
    list_len: NotRequired[int]


def _compact_state(payload: dict) -> list[_StateEntry]:
    """Render a StateUpdate payload as a compact summary of nodes + tool calls."""
    out: list[_StateEntry] = []
    for node_name, update in payload.items():
        msgs_to_scan: list[Any] = []
        list_len: int | None = None
        if isinstance(update, dict):
            msgs_to_scan = list(update.get("messages", []) or [])
        elif isinstance(update, list):
            list_len = len(update)
            for item in update:
                if isinstance(item, dict):
                    msgs_to_scan.extend(item.get("messages", []) or [])
                else:
                    msgs_to_scan.append(item)
        else:
            out.append({"node": str(node_name)})
            continue
        tool_names: list[str] = []
        for msg in msgs_to_scan:
            tc = getattr(msg, "tool_calls", None)
            if tc:
                tool_names.extend(c["name"] for c in tc if isinstance(c, dict) and "name" in c)
        entry: _StateEntry = {"node": str(node_name)}
        if tool_names:
            entry["tool_calls"] = tool_names
        if list_len is not None:
            entry["list_len"] = list_len
        out.append(entry)
    return out


def _to_jsonable(value: Any) -> Any:
    """Best-effort coercion of arbitrary values to something json.dumps can handle."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_jsonable(v) for v in value]
        return repr(value)


def render(event: InnerEvent | ProgressEvent, path: list[str]) -> dict[str, Any]:
    """Render *event* (already path-unwrapped) into a JSON-serializable dict."""
    base: dict[str, Any] = {
        "ts": event.ts,
        "path": path,
    }
    match event:
        case Start():
            base.update(
                kind="start",
                description=event.description,
                tool_id=event.tool_id,
                started_at_wall=event.started_at_wall,
            )
        case End():
            base.update(
                kind="end",
                duration_s=event.duration_s,
                error=event.error,
            )
        case NextCheckpoint():
            base.update(kind="checkpoint", checkpoint_id=event.checkpoint_id)
        case StateUpdate():
            base.update(kind="state_update", nodes=_compact_state(event.payload))
        case CustomUpdate():
            payload = _to_jsonable(event.payload)
            base.update(
                kind="custom",
                checkpoint_id=event.checkpoint_id,
                payload=payload,
            )
        case ProgressEvent():
            base.update(kind="progress", payload=_to_jsonable(event.payload))
    return base


def emit(event: InnerEvent | ProgressEvent, path: list[str]) -> None:
    """Serialize *event* and append a single JSON line to the events log."""
    try:
        record = render(event, path)
        _events_logger.info(json.dumps(record, default=str))
    except Exception as exc:
        try:
            _events_logger.info(
                json.dumps({
                    "ts": event.ts,
                    "kind": "serialize_error",
                    "path": path,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            )
        except Exception:
            pass
