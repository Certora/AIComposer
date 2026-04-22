from typing import Callable, Awaitable, AsyncIterator, cast
from contextlib import asynccontextmanager
import asyncio
import traceback
import inspect


from dataclasses import dataclass
from composer.io.protocol import IOHandler
from composer.io.context import with_handler
from composer.io.event_handler import EventHandler
from composer.io.conversation import ConversationContextProvider
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
    try:
        async with maybe_semaphore(semaphore):
            handle.on_start()
            async with with_handler(handle.handler, handle.event_handler):
                result = await inv()
    except Exception as exc:
        await handle.on_error(exc, traceback.format_exc())
        raise
    else:
        handle.on_done()
        return result
