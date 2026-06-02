"""
Run-level timing aggregator for the autoprove pipeline.

A single ``RunSummary`` is installed into a ``ContextVar`` at pipeline
entry. Phase orchestration (``run_task``) and slow operations (prover
invocations) report their wall-clock numbers into it. At end-of-run the
summary is formatted into a per-phase table.
"""

from contextlib import asynccontextmanager, contextmanager
from logging import Logger
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable, Protocol
import uuid


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
    final_link: str | None = None # URL (cloud) or local results directory (local) of the last prover run.


@dataclass
class RunSummary:
    started_at_mono: float = field(default_factory=time.perf_counter)
    phases: list[PhaseRecord] = field(default_factory=list)
    prover_total_s: float = 0.0
    prover_total_calls: int = 0
    _active_prover_by_task: dict[str, tuple[float, int]] = field(default_factory=dict, repr=False)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Maps task_id -> (prover_s_accum, prover_calls) recorded while task is in flight."""
    _latest_link_by_task: dict[str, str] = field(default_factory=dict, repr=False)
    """Maps task_id -> link for the most recent prover run."""
    _latest_conf_by_task: dict[str, dict] = field(default_factory=dict, repr=False)
    """Maps task_id -> the full conf dict passed to the most recent prover run."""

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
        prover_s, prover_calls = self._active_prover_by_task.pop(task_id, (0.0, 0))
        link = self._latest_link_by_task.pop(task_id, None)
        self.phases.append(PhaseRecord(
            task_id=task_id,
            label=label,
            phase=phase,
            wall_s=wall_s,
            queue_wait_s=queue_wait_s,
            error=error,
            prover_s=prover_s,
            prover_calls=prover_calls,
            final_link=link,
        ))

    def add_prover_call(self, duration_s: float, *, task_id: str | None = None) -> None:
        task_id = task_id or get_current_task_id()
        self.prover_total_s += duration_s
        self.prover_total_calls += 1
        if task_id is not None:
            prev_s, prev_n = self._active_prover_by_task.get(task_id, (0.0, 0))
            self._active_prover_by_task[task_id] = (prev_s + duration_s, prev_n + 1)

    def record_prover_link(self, link: str, *, task_id: str | None = None) -> None:
        """Stash the latest prover link (URL or local path); defaults to the active task."""
        if (task_id := task_id or get_current_task_id()) is None:
            return
        self._latest_link_by_task[task_id] = link

    def get_latest_link(self, task_id: str | None = None) -> str | None:
        """Return the most recent prover link; defaults to the active task. None if none."""
        if (task_id := task_id or get_current_task_id()) is None:
            return None
        return self._latest_link_by_task.get(task_id)

    def record_prover_conf(self, conf: dict, *, task_id: str | None = None) -> None:
        """Stash the full conf passed to the prover; defaults to the active task. Last write wins."""
        if (task_id := task_id or get_current_task_id()) is None:
            return
        self._latest_conf_by_task[task_id] = dict(conf)

    def get_latest_conf(self, task_id: str | None = None) -> dict | None:
        """Return a copy of the most recent prover conf; defaults to the active task. None if none."""
        if (task_id := task_id or get_current_task_id()) is None:
            return None
        v = self._latest_conf_by_task.get(task_id)
        return dict(v) if v is not None else None

    def total_wall_s(self) -> float:
        return time.perf_counter() - self.started_at_mono

    def format(self) -> str:
        return _format_summary(self)


_run_summary: ContextVar[RunSummary | None] = ContextVar("_run_summary", default=None)
_current_task_id: ContextVar[str | None] = ContextVar("_current_task_id", default=None)


def get_run_summary() -> RunSummary:
    """Return the active run summary, or a throwaway inert one outside a run.

    Writes to the throwaway are discarded and reads return ``None``, so callers
    never need to None-check — they record unconditionally and treat any read of
    ``None`` as "no data," whether or not a run is active.
    """
    return _run_summary.get() or RunSummary()


def install_run_summary(summary: RunSummary) -> None:
    """Install ``summary`` as the active aggregator for the rest of the run."""
    _run_summary.set(summary)


def get_current_task_id() -> str | None:
    """Return the task_id of the active ``run_task`` scope, if any."""
    return _current_task_id.get()


@contextmanager
def set_current_task_id(task_id: str):
    tok = _current_task_id.set(task_id)
    try:
        yield
    finally:
        _current_task_id.reset(tok)


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
    linked = [p for p in summary.phases if p.final_link]
    if linked:
        out.append("")
        out.append("Final prover runs:")
        for p in linked:
            out.append(f"  {p.label}: {p.final_link}")
    failures = [p for p in summary.phases if p.error is not None]
    if failures:
        out.append("")
        out.append(f"Failures ({len(failures)}):")
        for p in failures:
            out.append(f"  - {p.phase}/{p.task_id} ({p.label}): {p.error}")
    return "\n".join(out)


class StartLogger(Protocol):
    def task_started(self) -> None:
        ...

@dataclass
class _TaskLog:
    t_running: float | None = None
    def task_started(self):
          self.t_running = time.perf_counter()

@asynccontextmanager
async def task_logger(
    task_id: str,
    label: str,
    phase_name: str,
    logger: Logger,
) -> AsyncIterator[StartLogger]:
    summary = _run_summary.get()
    if summary is None:
        # No active run — skip timing bookkeeping entirely.
        yield _TaskLog()
        return
    t_request = time.perf_counter()
    log = _TaskLog()
    tok = _current_task_id.set(task_id)
    try:
         yield log
    except Exception as exc:
        err_name = type(exc).__name__
        elapsed = time.perf_counter() - t_request
        queue_wait = (log.t_running - t_request) if log.t_running is not None else elapsed
        logger.exception(
            f"task failed: phase={phase_name} task_id={task_id} "
            f"wall={elapsed:.2f}s queue_wait={queue_wait:.2f}s error={err_name}"
        )
        summary.record_phase(
            task_id=task_id, label=label, phase=phase_name,
            wall_s=elapsed, queue_wait_s=queue_wait, error=err_name,
        )
        raise exc
    else:
        elapsed = time.perf_counter() - t_request
        queue_wait = (log.t_running - t_request) if log.t_running is not None else 0.0
        logger.info(
            f"task done: phase={phase_name} task_id={task_id} "
            f"wall={elapsed:.2f}s queue_wait={queue_wait:.2f}s"
        )
        summary.record_phase(
            task_id=task_id, label=label, phase=phase_name,
            wall_s=elapsed, queue_wait_s=queue_wait, error=None,
        )
    finally:
        _current_task_id.reset(tok)
