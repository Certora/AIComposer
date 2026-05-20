from typing import Any, Callable, Awaitable, AsyncIterator, cast
from contextlib import asynccontextmanager
import asyncio
import logging
import time
import traceback
import inspect


from dataclasses import dataclass
from composer.io.protocol import IOHandler
from composer.io.context import with_handler
from composer.io.event_handler import EventHandler
from composer.io.conversation import ConversationContextProvider

logger = logging.getLogger(__name__)

# Per-phase cumulative stats: phase -> (count, total_seconds, max_seconds)
_phase_stats: dict[Any, tuple[int, float, float]] = {}


def _record_phase(phase: Any, elapsed: float) -> None:
    n, total, peak = _phase_stats.get(phase, (0, 0.0, 0.0))
    _phase_stats[phase] = (n + 1, total + elapsed, max(peak, elapsed))


def phase_summary() -> str:
    """Format cumulative per-phase task timing as a table."""
    if not _phase_stats:
        return "(no tasks recorded)"
    rows = sorted(_phase_stats.items(), key=lambda kv: -kv[1][1])
    lines = [f"{'phase':<28} {'count':>5} {'total(s)':>10} {'max(s)':>8}"]
    for phase, (n, total, peak) in rows:
        name = getattr(phase, "value", str(phase))
        lines.append(f"{name:<28} {n:>5d} {total:>10.2f} {peak:>8.2f}")
    return "\n".join(lines)


def reset_phase_stats() -> None:
    _phase_stats.clear()
# ---------------------------------------------------------------------------
# Handler factory types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskInfo[P]:
    task_id: str
    label: str
    phase: P


@dataclass(frozen=True)
class TaskHandle[H]:
    """Bundles an IOHandler with lifecycle callbacks."""
    handler: IOHandler[H]
    event_handler: EventHandler
    conversation_provider: ConversationContextProvider
    on_error: Callable[[Exception, str], Awaitable[None]]
    on_start: Callable[[], None] = lambda: None
    on_done: Callable[[], None] = lambda: None


type HandlerFactory[P, H] = Callable[[TaskInfo[P]], Awaitable[TaskHandle[H]]]


# ---------------------------------------------------------------------------
# run_task helper
# ---------------------------------------------------------------------------

@asynccontextmanager
async def maybe_semaphore(
    sem: asyncio.Semaphore | None
) -> AsyncIterator[None]:
    if sem is None:
        yield
    else:
        async with sem:
            yield

async def run_task[P, T, H](
    factory: HandlerFactory[P, H],
    info: TaskInfo[P],
    fn: Callable[[], Awaitable[T]] | Callable[[ConversationContextProvider], Awaitable[T]],
    semaphore: asyncio.Semaphore | None = None,
) -> T:
    """Create a handler via *factory* and run *fn* in its ``with_handler`` scope.

    P - Type of phase markers
    T - return type of the task
    H - Type of human interaction request (routed through the handler from factory)

    Manages lifecycle callbacks (on_start/on_done/on_error).  If
    *semaphore* is provided, the task waits for acquisition before
    transitioning to RUNNING.
    """
    handle = await factory(info)
    if len(inspect.signature(fn).parameters) > 0:
        capture = cast(Callable[[ConversationContextProvider], Awaitable[T]], fn)
        inv = lambda: capture(handle.conversation_provider)
    else:
        inv = cast(Callable[[], Awaitable[T]], fn)

    phase_name = getattr(info.phase, "value", str(info.phase))
    t0 = time.perf_counter()
    logger.info("[task] start id=%s phase=%s label=%r", info.task_id, phase_name, info.label)
    try:
        async with maybe_semaphore(semaphore):
            t_acq = time.perf_counter()
            if semaphore is not None and (wait := t_acq - t0) > 0.1:
                logger.info("[task] id=%s acquired semaphore after %.2fs", info.task_id, wait)
            handle.on_start()
            async with with_handler(handle.handler, handle.event_handler):
                result = await inv()
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _record_phase(info.phase, elapsed)
        logger.warning("[task] id=%s phase=%s FAILED after %.2fs: %s", info.task_id, phase_name, elapsed, exc)
        await handle.on_error(exc, traceback.format_exc())
        raise
    else:
        elapsed = time.perf_counter() - t0
        _record_phase(info.phase, elapsed)
        logger.info("[task] id=%s phase=%s done in %.2fs", info.task_id, phase_name, elapsed)
        handle.on_done()
        return result
