"""
Lock-protected shared artifacts backed by BaseStore.

Provides an asyncio.Lock to serialize reads and writes to shared state
(e.g., a master CVL spec being edited by multiple property agents).
The lock is held for the entire read-merge-write cycle so that only one
agent merges at a time.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from langgraph.store.base import BaseStore


@dataclass
class SharedArtifact:
    """A shared string artifact with lock-based concurrent update protection.

    Use ``locked()`` to acquire exclusive access. It yields the current
    content and a setter — all reads and writes within the context are
    serialized.
    """
    _store: BaseStore
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _namespace: tuple[str, ...] = ()
    _key: str = ""

    @staticmethod
    def create(
        store: BaseStore,
        namespace: tuple[str, ...],
        key: str,
        initial_content: str = "",
    ) -> "SharedArtifact":
        """Create a new SharedArtifact, writing the initial value to the store."""
        artifact = SharedArtifact(
            _store=store,
            _namespace=namespace,
            _key=key,
        )
        store.put(namespace, key, {"content": initial_content})
        return artifact

    def _read(self) -> str | None:
        item = self._store.get(self._namespace, self._key)
        if item is None:
            return None
        return item.value["content"]

    def _write(self, content: str) -> None:
        self._store.put(self._namespace, self._key, {"content": content})

    def read_unsync(self) -> str | None:
        """Read current content without acquiring the lock.

        Safe when no concurrent writes are possible (e.g., after all agents have finished).
        """
        return self._read()

    @asynccontextmanager
    async def locked(self) -> AsyncIterator[tuple[str | None, Callable[[str], None]]]:
        """Acquire exclusive access for a read-merge-write cycle.

        Yields ``(current_content, write)`` — read once, write at most once.
        ``current_content`` is ``None`` if no value has been written yet.
        """
        async with self._lock:
            yield self._read(), self._write
