"""Run-state types and the in-process registry.

Each in-flight run has a :class:`RunState` containing its tasks, SSE
subscribers, replay log, and metadata. Subscribers are
``asyncio.Queue``s of already-formatted SSE strings; the SSE endpoint
pulls from one queue, ``RunState.push`` writes to all of them.

This is intentionally process-scoped — the project memo lists
"persistent run registry across server restart" as out of v1 scope. If
the process restarts, in-flight runs are gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import json
import pathlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from composer.web.render import render_fragment


# Avoid a runtime cycle: ``Task`` lives in :mod:`composer.web.handler`
# which imports from this module. Type-only import keeps ``RunState.tasks``
# precisely typed without dragging the handler module into runtime
# initialization order.
if TYPE_CHECKING:
    from composer.web.handler import Task


# How many wire events to retain per run for SSE replay. When the buffer
# fills, the oldest events are dropped and reconnects asking for events
# older than the new low_water_mark get a "stale" signal back, prompting
# a full page reload (fresh GET re-renders shell, replay then catches up
# with whatever's still retained).
#
# 50k events is generous — at ~10 events/sec a 30-min mock run produces
# ~18k. Real autoprove flows that bump this should make us think about
# compaction (replaces of the same selector are subsumable; appends are
# the only inherently un-subsumable op).
MAX_EVENTS = 50_000


# ---------------------------------------------------------------------------
# Phase / status enums
# ---------------------------------------------------------------------------

class Phase(enum.Enum):
    HARNESS            = "Harness Setup"
    SUMMARIES          = "Summaries"
    INVARIANTS         = "Structural Invariants"
    COMPONENT_ANALYSIS = "Component Analysis"
    BUG_ANALYSIS       = "Property Extraction"
    CVL_GEN            = "CVL Generation"


PHASE_ORDER: list[Phase] = list(Phase)


class Status(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


STATUS_ICON: dict[Status, str] = {
    Status.PENDING: "○",
    Status.RUNNING: "●",
    Status.DONE:    "✓",
    Status.ERROR:   "✗",
}


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class Event(NamedTuple):
    """One wire op recorded for replay. ``seq`` is the SSE id the
    browser auto-tracks via ``Last-Event-ID`` on reconnect."""
    seq: int
    op: str
    sel: str
    html: str


def _sse_event(seq: int, op: str, payload: str) -> str:
    """Encode one SSE message frame with id + event-name + data.

    The ``id:`` field is what the browser auto-stores and replays back
    as ``Last-Event-ID`` on reconnect — the whole replay scheme leans
    on this. Multi-line payloads get one ``data:`` line each per the
    spec."""
    lines = "\n".join(f"data: {line}" for line in payload.split("\n"))
    return f"id: {seq}\nevent: {op}\n{lines}\n\n"


def serialize_event(e: Event) -> str:
    """Format a stored ``Event`` for replay. Identical wire format to a
    live broadcast — both go through this helper so one bug surfaces in
    both paths."""
    return _sse_event(e.seq, e.op, json.dumps({"sel": e.sel, "html": e.html}))


# ---------------------------------------------------------------------------
# RunState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunRequest:
    """Inputs needed to launch — or re-launch — a run.

    Captured at submit time and stored on :class:`RunState` so the
    retry endpoint can reconstruct a new run from an old one without
    asking the user to re-upload artefacts. ``cache_ns`` and
    ``root_thread_id`` are determined here (rather than later in the
    pipeline) so a retry can reuse them deterministically.

    Mock-mode runs leave ``RunState.request`` as ``None`` — they don't
    support retry. The retry banner is only rendered when ``request``
    is set."""
    project_id: str
    main_contract_raw: str
    model: str
    max_concurrent: int
    cloud: bool
    system_doc_path: pathlib.Path
    threat_model_path: pathlib.Path | None
    # Cache namespace; combined with a content hash of the inputs to
    # form the actual cache key. Stable across retries of the same
    # input so cached phases skip on re-run.
    cache_ns: str
    # Memory namespace; ``None`` falls through to thread_id inside
    # WorkflowContext. v1 doesn't surface this in the UI.
    memory_ns: str | None
    # Root langgraph thread id. Soft retry reuses; hard retry
    # generates a fresh one.
    root_thread_id: str


@dataclass
class RunState:
    run_id: str
    inputs: dict
    workspace: pathlib.Path
    started_at: datetime
    tasks: dict[str, "Task"] = field(default_factory=dict)
    phase_seen: set[Phase] = field(default_factory=set)
    subscribers: list[asyncio.Queue[str]] = field(default_factory=list)
    done: bool = False
    final_status_text: str = "Running"
    final_status_class: str = "running"
    output_files: list[dict] = field(default_factory=list)
    # Real-mode runs have a ``RunRequest``; mock-mode runs leave it
    # ``None`` (they don't support retry).
    request: "RunRequest | None" = None
    # Per-run event log for SSE replay. Order matches emission order.
    events: list[Event] = field(default_factory=list)
    # Monotonic counter for the next event's seq. Never decreases (even
    # past trim) so resumes are unambiguous across the run's lifetime.
    next_seq: int = 0
    # Seq of the oldest event still retained. After a trim, equals
    # ``events[0].seq``; equals ``next_seq`` when the log is empty.
    # Reconnects with ``Last-Event-ID + 1 < low_water_mark`` are too far
    # behind and get told to reload.
    low_water_mark: int = 0

    # ---- wire emission ------------------------------------------------

    def push(self, op: str, sel: str, html: str) -> None:
        """Append a wire op to the replay log AND broadcast it to every
        live subscriber.

        Both steps are synchronous (no ``await`` between them) so
        they're atomic w.r.t. other coroutines on the loop — the SSE
        endpoint's subscribe-then-snapshot replay logic relies on this.
        Drops messages on full subscriber queues rather than blocking;
        a slow client must not jam the pipeline."""
        seq = self.next_seq
        self.next_seq += 1
        self.events.append(Event(seq=seq, op=op, sel=sel, html=html))

        # Bounded ring buffer. Oldest events are discarded once the cap
        # is hit; ``low_water_mark`` advances to the new oldest seq.
        if len(self.events) > MAX_EVENTS:
            del self.events[: len(self.events) - MAX_EVENTS]
        self.low_water_mark = self.events[0].seq if self.events else seq + 1

        payload = _sse_event(seq, op, json.dumps({"sel": sel, "html": html}))
        for q in list(self.subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(payload)

    # ---- initial render helpers ---------------------------------------

    def render_status_html(self) -> str:
        """Render the page-header status pill from the run's current
        snapshot.

        Special-cased over SSE replay because the initial run-status is
        set in the dataclass (not via ``push``), so it never appears in
        the event log. Reading it directly here means a reload-after-
        completion shows the final status immediately rather than
        waiting for replay to finish.

        Everything else on the run page (phase sections, task rows /
        panels, log entries, outputs) is reconstructed by SSE replay
        against an empty shell."""
        return render_fragment(
            "fragments/run_status.j2",
            status_text=self.final_status_text,
            status_class=self.final_status_class,
        )


# Process-scoped registry. Single user, single process — no need for a
# lookup index keyed by anything but run_id. Server restart wipes it,
# which is OK per project memo (out of v1 scope).
RUNS: dict[str, RunState] = {}
