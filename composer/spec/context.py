"""
Workflow context, services protocol, builder type aliases, and cache infrastructure.

Ported from composer/spec/context.py on the jtoman/auto-prover branch.
WorkspaceContext has been factored into WorkflowContext: workflow-specific data
(project root, contract name, etc.) is no longer part of the context and must
be passed explicitly to agents that need it.
"""

from dataclasses import dataclass
import base64
from pathlib import Path
from typing import Annotated, Callable, overload, Awaitable

from pydantic import BaseModel, ValidationError

from langgraph.store.base import BaseStore
from langchain_core.tools import BaseTool

from graphcore.graph import Builder


# ---------------------------------------------------------------------------
# Workflow input types
# ---------------------------------------------------------------------------

@dataclass
class SystemDoc:
    """Input when only a design document is available (natspec)."""
    content: str | dict  # text or base64-encoded PDF


@dataclass
class SourceCode(SystemDoc):
    """Input when source code is also available (source_spec)."""
    project_root: str
    contract_name: str
    relative_path: str
    forbidden_read: str


# ---------------------------------------------------------------------------
# Services protocol
# ---------------------------------------------------------------------------

type MemoryFactory = Callable[[str], BaseTool]
type WorkflowServices = MemoryFactory

# ---------------------------------------------------------------------------
# Builder phantom markers and type aliases
# ---------------------------------------------------------------------------

class SOURCE_TOOLS:
    """Builder has fs_tools bound (source code file access)."""

class CVL_TOOLS:
    """Builder has cvl_manual_tools bound (CVL manual RAG search)."""

type SourceBuilder = Annotated[Builder[None, None, None], SOURCE_TOOLS]
type CVLOnlyBuilder = Annotated[Builder[None, None, None], CVL_TOOLS]


type PlainBuilder = Builder[None, None, None]


# ---------------------------------------------------------------------------
# Cache hierarchy types
# ---------------------------------------------------------------------------

type CacheTypes = None | BaseModel | Marker

SNAPSHOT_NAMESPACE = ("snapshots",)


class CacheKey[Parent: CacheTypes, Curr: CacheTypes]:
    def __init__(self, key: str):
        self.key = key

    def __str__(self) -> str:
        return self.key


# Phantom marker types for the cache hierarchy.
class InvJudge:
    """Invariant formulation feedback judge step."""

class InvFormal:
    """Grouping step for individual invariant formalization."""

class Properties:
    """Grouping step for property-level analysis."""

class ComponentGroup:
    """A single application component under analysis."""

class CVLJudge:
    """CVL property feedback judge step."""

class CVLGeneration:
    """Abstraction for the CVL generation pipeline."""

class Contract:
    """An individual contract"""

type Abstraction = CVLGeneration

type Marker = (
    InvJudge | InvFormal | Properties | ComponentGroup
    | CVLJudge | Abstraction | Contract
)

# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------

@dataclass
class WorkflowContext[K: CacheTypes]:
    """
    Manages thread IDs, memory namespaces, and caching for workflows.

    Unlike the original WorkspaceContext, this does NOT hold workflow-specific
    data (project root, contract name, etc.). That data should be passed
    explicitly to agents that need it.

    - thread_id: Root for LangGraph checkpointing (sub-workflows derive from this)
    - memory_namespace: String namespace for persistent memory (memory_tool)
    - cache_namespace: Tuple namespace for store caching (None = no caching)
    """
    _services: WorkflowServices
    thread_id: str
    memory_namespace: str
    cache_namespace: tuple[str, ...] | None
    _store: BaseStore

    def abstract[T: Abstraction](self, ty: type[T]) -> "WorkflowContext[T]":
        return self  # type: ignore[return-value]

    @staticmethod
    def create(
        services: WorkflowServices,
        thread_id: str,
        store: BaseStore,
        memory_namespace: str | None = None,
        cache_namespace: tuple[str, ...] | None | str = None,
    ) -> "WorkflowContext[None]":
        cache_ns: tuple[str, ...] | None
        if isinstance(cache_namespace, str):
            cache_ns = (cache_namespace,)
        else:
            cache_ns = cache_namespace
        return WorkflowContext(
            _services=services,
            thread_id=thread_id,
            memory_namespace=memory_namespace or thread_id,
            cache_namespace=cache_ns,
            _store=store,
        )
    
    @overload
    def child[NXT: CacheTypes](self, name_key: CacheKey[K, NXT]) -> "WorkflowContext[NXT]":
        ...

    @overload
    def child[NXT: CacheTypes](self, name_key: CacheKey[K, NXT], tag: dict) -> Awaitable["WorkflowContext[NXT]"]:
        ...

    def _child_pure[NXT: CacheTypes](
        self, name_key: CacheKey[K, NXT],    
    ) -> tuple["WorkflowContext[NXT]", tuple[str, ...] | None]:
        name = name_key.key
        child_cache_ns = (*self.cache_namespace, name) if self.cache_namespace else None
        return (WorkflowContext(
            _services=self._services,
            thread_id=f"{self.thread_id}-{name}",
            memory_namespace=f"{self.memory_namespace}-{name}",
            cache_namespace=child_cache_ns,
            _store=self._store,
        ), child_cache_ns)

    async def _child_async[NXT: CacheTypes](
        self, name_key: CacheKey[K, NXT], tag: dict
    ) -> "WorkflowContext[NXT]":
        (nxt, cache_key) = self._child_pure(name_key)
        if cache_key is not None:
            await self._store.aput(cache_key, "_desc", tag)
        return nxt
        
    def _child_sync[NXT: CacheTypes](
        self, name_key: CacheKey[K, NXT]
    ) -> "WorkflowContext[NXT]":
        return self._child_pure(name_key)[0]

    def child[NXT: CacheTypes](self, name_key: CacheKey[K, NXT], tag: dict | None = None) -> "WorkflowContext[NXT] | Awaitable[WorkflowContext[NXT]]":
        """Create a child context with derived namespaces."""
        if tag is None:
            return self._child_sync(name_key)
        else:
            return self._child_async(name_key, tag)

    async def cache_get(self, ty: type[K]) -> K | None:
        """Get a typed value from the cache. Returns None if caching disabled or not found."""
        if not issubclass(ty, BaseModel):
            raise ValueError(f"Cannot use cache with non-basemodel keys {ty}")
        if self.cache_namespace is None:
            return None
        if len(self.cache_namespace) < 1:
            raise ValueError("Cache prefix too small")
        full_key = self.cache_namespace[:-1]
        result = await self._store.aget(full_key, self.cache_namespace[-1])
        if result is None:
            return None
        try:
            return ty.model_validate(result.value)
        except ValidationError:
            await self._store.adelete(full_key, self.cache_namespace[-1])
            return None

    async def cache_put(self, value: K) -> None:
        """Put a typed value in the cache. No-op if caching disabled."""
        if not isinstance(value, BaseModel):
            raise ValueError("Caching not allowed for non-basemodel keys")
        if self.cache_namespace is None:
            return
        if len(self.cache_namespace) < 1:
            raise ValueError("Cache prefix too small")
        full_key = self.cache_namespace[:-1]
        await self._store.aput(full_key, self.cache_namespace[-1], value.model_dump())

    def get_memory_tool(self) -> BaseTool:
        """Get a memory tool for this context's memory namespace."""
        return self._services(self.memory_namespace)

    async def save_snapshot(self, key: str, data: BaseModel) -> None:
        """Store a snapshot in the global snapshots namespace."""
        await self._store.aput(SNAPSHOT_NAMESPACE, key, data.model_dump())


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_system_doc(sys_path: Path) -> dict | str | None:
    """Load a system document from a file path, returning base64-encoded PDF or text."""
    if not sys_path.is_file():
        return None
    if sys_path.suffix == ".pdf":
        file_data = base64.standard_b64encode(sys_path.read_bytes()).decode("utf-8")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": file_data
            }
        }
    else:
        return sys_path.read_text()
