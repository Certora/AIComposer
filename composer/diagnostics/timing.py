"""
Run-level timing aggregator for the autoprove pipeline.

A single ``RunSummary`` is installed into a ``ContextVar`` at pipeline
entry. Phase orchestration (``run_task``) and slow operations (prover
invocations) report their wall-clock numbers into it. At end-of-run the
summary is formatted into a per-phase table.

Designed to be safe to call from any thread / any coroutine: the
underlying lock is asyncio-free (``threading.Lock``) so callbacks from
subprocess I/O are fine too.
"""

from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class PhaseRecord:
    task_id: str
    label: str
    phase: str
    wall_s: float
    queue_wait_s: float
    error: str | None = None
    prover_s: float = 0.0
    prover_calls: int = 0


@dataclass
class RunSummary:
    started_at_mono: float = field(default_factory=time.perf_counter)
    phases: list[PhaseRecord] = field(default_factory=list)
    prover_total_s: float = 0.0
    prover_total_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _active_prover_by_task: dict[str, tuple[float, int]] = field(default_factory=dict, repr=False)
    """Maps task_id -> (prover_s_accum, prover_calls) recorded while task is in flight."""

    def record_phase(
        self,
        *,
        task_id: str,
        label: str,
        phase: str,
        wall_s: float,
        queue_wait_s: float,
        error: str | None = None,
    ) -> None:
        with self._lock:
            prover_s, prover_calls = self._active_prover_by_task.pop(task_id, (0.0, 0))
            self.phases.append(PhaseRecord(
                task_id=task_id,
                label=label,
                phase=phase,
                wall_s=wall_s,
                queue_wait_s=queue_wait_s,
                error=error,
                prover_s=prover_s,
                prover_calls=prover_calls,
            ))

    def add_prover_call(self, task_id: str | None, duration_s: float) -> None:
        with self._lock:
            self.prover_total_s += duration_s
            self.prover_total_calls += 1
            if task_id is not None:
                prev_s, prev_n = self._active_prover_by_task.get(task_id, (0.0, 0))
                self._active_prover_by_task[task_id] = (prev_s + duration_s, prev_n + 1)

    def total_wall_s(self) -> float:
        return time.perf_counter() - self.started_at_mono

    def format(self) -> str:
        return _format_summary(self)


_run_summary: ContextVar[RunSummary | None] = ContextVar("_run_summary", default=None)
_current_task_id: ContextVar[str | None] = ContextVar("_current_task_id", default=None)


def get_run_summary() -> RunSummary | None:
    """Return the active run summary, if any. Returns ``None`` outside an autoprove run."""
    return _run_summary.get()


def install_run_summary(summary: RunSummary) -> None:
    """Install ``summary`` as the active aggregator for the rest of the run."""
    _run_summary.set(summary)


def get_current_task_id() -> str | None:
    """Return the task_id of the active ``run_task`` scope, if any."""
    return _current_task_id.get()


def set_current_task_id(task_id: str) -> None:
    _current_task_id.set(task_id)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_secs(s: float) -> str:
    if s < 1.0:
        return f"{s*1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {sec:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m):02d}m {sec:04.1f}s"


def _format_summary(summary: RunSummary) -> str:
    total = summary.total_wall_s()
    if not summary.phases:
        return f"Auto-prove complete in {_fmt_secs(total)} (no phase data captured)"

    headers = ("Phase / Task", "Wall", "Queue wait", "Prover", "Calls", "Status")
    rows: list[tuple[str, str, str, str, str, str]] = []
    for p in summary.phases:
        label = f"{p.phase}: {p.label}" if p.label and p.label != p.task_id else p.phase
        status = "ok" if p.error is None else p.error
        rows.append((
            label,
            _fmt_secs(p.wall_s),
            _fmt_secs(p.queue_wait_s),
            _fmt_secs(p.prover_s) if p.prover_calls else "—",
            str(p.prover_calls) if p.prover_calls else "—",
            status,
        ))

    widths = [max(len(headers[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(headers))]
    sep = "  "

    def line(cells: Iterable[str]) -> str:
        return sep.join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out: list[str] = []
    out.append(f"Auto-prove complete in {_fmt_secs(total)}")
    out.append("─" * (sum(widths) + len(sep) * (len(headers) - 1)))
    out.append(line(headers))
    out.append("─" * (sum(widths) + len(sep) * (len(headers) - 1)))
    for r in rows:
        out.append(line(r))
    out.append("─" * (sum(widths) + len(sep) * (len(headers) - 1)))
    out.append(
        f"Prover total: {_fmt_secs(summary.prover_total_s)} across "
        f"{summary.prover_total_calls} call(s)"
    )
    failures = [p for p in summary.phases if p.error is not None]
    if failures:
        out.append("")
        out.append(f"Failures ({len(failures)}):")
        for p in failures:
            out.append(f"  - {p.phase}/{p.task_id} ({p.label}): {p.error}")
    return "\n".join(out)
