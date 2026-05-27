"""
JSONL serialization for graph events.

Renders each ``GraphEvents`` instance as a single-line JSON object suitable
for streaming to the rotating ``composer.events`` logger. Large string
fields in custom payloads are truncated so a chatty prover doesn't blow
the rotation budget on a single rule.

The serializer is best-effort: any unserializable payload is logged with
``kind=serialize_error`` rather than crashing the drainer.
"""

import json
import logging
import time
from typing import Any

from composer.io.events import (
    Start, End, StateUpdate, NextCheckpoint, CustomUpdate, ProgressEvent, InnerEvent,
)


_MAX_STR = 2048
_events_logger = logging.getLogger("composer.events")


def _truncate(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STR:
        return value[:_MAX_STR] + f"…[+{len(value) - _MAX_STR} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v) for v in value]
    if isinstance(value, tuple):
        return [_truncate(v) for v in value]
    return value


def _compact_state(payload: dict) -> list[dict]:
    """Render a StateUpdate payload as a compact summary of nodes + tool calls."""
    out: list[dict] = []
    for node_name, update in payload.items():
        if not isinstance(update, dict):
            out.append({"node": str(node_name)})
            continue
        tool_names: list[str] = []
        for msg in update.get("messages", []) or []:
            tc = getattr(msg, "tool_calls", None)
            if tc:
                tool_names.extend(c["name"] for c in tc if isinstance(c, dict) and "name" in c)
        entry: dict[str, Any] = {"node": node_name}
        if tool_names:
            entry["tool_calls"] = tool_names
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
        "ts": time.time(),
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
            payload = _truncate(_to_jsonable(event.payload))
            base.update(
                kind="custom",
                checkpoint_id=event.checkpoint_id,
                payload=payload,
            )
        case ProgressEvent():
            base.update(kind="progress", payload=_truncate(_to_jsonable(event.payload)))
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
                    "ts": time.time(),
                    "kind": "serialize_error",
                    "path": path,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            )
        except Exception:
            pass
