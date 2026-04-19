"""
Append-only event buffer consumed by the background drainer.

One ``EventQueue`` is created per ``with_handler()`` scope.  All
event sinks within that scope (one per ``run_graph()`` call) push
to the same queue.  A single ``_queue_drainer`` task consumes
events via ``stream_events()``.

The queue never blocks writers — ``push()`` is synchronous.  The
consumer blocks on an ``asyncio.Event`` until new items arrive.
"""

from typing import AsyncIterator
from dataclasses import dataclass
from composer.io.events import AllEvents
import asyncio

@dataclass
class AsyncDataQueue[T]:
    """Multi-producer, single-consumer async event buffer.

    Construct with ``EventQueue(asyncio.Event(), [])``.
    """
    _ready: asyncio.Event
    _event_stream: list[T]
    _cursor: int = 0

    def push(self, event: T) -> None:
        """Append an event and signal the consumer.  Non-blocking."""
        self._event_stream.append(event)
        self._ready.set()

    async def stream_events(self) -> AsyncIterator[T]:
        """Yield events as they arrive.  Blocks when caught up."""
        while True:
            await self._ready.wait()
            self._ready.clear()
            while self._cursor < len(self._event_stream):
                yield self._event_stream[self._cursor]
                self._cursor += 1
            assert self._cursor == len(self._event_stream)
            self._cursor = 0
            self._event_stream = []


EventQueue = AsyncDataQueue[AllEvents]
