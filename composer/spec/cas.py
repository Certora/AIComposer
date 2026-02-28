"""
CAS-protected shared artifacts backed by BaseStore.

Provides compare-and-swap semantics for concurrent agent access to shared
state (e.g., a master CVL spec being edited by multiple property agents).
Atomicity is achieved via an asyncio.Lock around the read-compare-write
cycle — the lock is only held during the atomic operation, not during agent
reasoning.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum

from langgraph.store.base import BaseStore


class CASStatus(Enum):
    SUCCESS = "success"
    CONFLICT = "conflict"


@dataclass
class CASResult:
    """Result of a compare-and-swap update."""
    status: CASStatus
    content: str
    version: int

    @property
    def success(self) -> bool:
        return self.status == CASStatus.SUCCESS


@dataclass
class SharedArtifact:
    """A shared string artifact with CAS-based concurrent update protection.

    Version is stored in the BaseStore value dict as {"content": "...", "version": N}.
    The asyncio.Lock serializes read-compare-write cycles.
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
        """Create a new SharedArtifact, writing the initial version to the store."""
        artifact = SharedArtifact(
            _store=store,
            _namespace=namespace,
            _key=key,
        )
        store.put(namespace, key, {"content": initial_content, "version": 0})
        return artifact

    async def read(self) -> tuple[str, int]:
        """Read current content and version (no lock needed for reads)."""
        item = self._store.get(self._namespace, self._key)
        if item is None:
            return "", 0
        return item.value["content"], item.value["version"]

    async def cas_update(self, expected_version: int, new_content: str) -> CASResult:
        """Atomically update content if version matches expected_version.

        Returns CASResult with success and new version, or conflict with current state.
        """
        async with self._lock:
            item = self._store.get(self._namespace, self._key)
            if item is None:
                current_version = 0
                current_content = ""
            else:
                current_content = item.value["content"]
                current_version = item.value["version"]

            if current_version != expected_version:
                return CASResult(
                    status=CASStatus.CONFLICT,
                    content=current_content,
                    version=current_version,
                )

            new_version = current_version + 1
            self._store.put(
                self._namespace,
                self._key,
                {"content": new_content, "version": new_version},
            )
            return CASResult(
                status=CASStatus.SUCCESS,
                content=new_content,
                version=new_version,
            )
