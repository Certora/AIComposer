from typing import AsyncIterator
from dataclasses import dataclass
from composer.io.events import AllEvents
import asyncio


@dataclass
class EventQueue:
    _ready: asyncio.Event
    _event_stream: list[AllEvents]
    _cursor: int = 0

    def push(self, event: AllEvents):
        self._event_stream.append(event)
        self._ready.set()

    async def stream_events(
        self
    ) -> AsyncIterator[AllEvents]:
        while True:
            await self._ready.wait()
            self._ready.clear()
            while self._cursor < len(self._event_stream):
                yield self._event_stream[self._cursor]
                self._cursor += 1
