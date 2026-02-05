from dataclasses import dataclass

from langgraph.store.postgres import PostgresStore
from langchain_core.tools import BaseTool

from graphcore.tools.memory import memory_tool

from composer.workflow.services import get_memory


@dataclass
class ContractSpec:
    relative_path: str
    contract_name: str

@dataclass
class JobSpec(ContractSpec):
    project_root: str
    system_doc: str
    fs_filter: str

@dataclass
class WorkspaceContext:
    """
    Manages thread IDs, memory namespaces, and caching for workflows.

    Wraps a JobSpec and delegates property access to it for type safety.

    - thread_id: Root for LangGraph checkpointing (sub-workflows derive from this)
    - memory_namespace: String namespace for persistent memory (memory_tool)
    - cache_namespace: Tuple namespace for store caching (None = no caching)
    """
    _job_spec: JobSpec
    thread_id: str
    memory_namespace: str
    cache_namespace: tuple[str, ...] | None
    _store: PostgresStore

    # Delegate JobSpec properties for type safety
    @property
    def project_root(self) -> str:
        return self._job_spec.project_root

    @property
    def system_doc(self) -> str:
        return self._job_spec.system_doc

    @property
    def relative_path(self) -> str:
        return self._job_spec.relative_path

    @property
    def contract_name(self) -> str:
        return self._job_spec.contract_name

    @property
    def fs_filter(self) -> str:
        return self._job_spec.fs_filter

    @staticmethod
    def create(
        js: JobSpec,
        thread_id: str,
        store: PostgresStore,
        memory_namespace: str | None = None,
        cache_namespace: tuple[str, ...] | None | str = None,
    ) -> "WorkspaceContext":
        cache_ns: tuple[str, ...] | None
        if isinstance(cache_namespace, str):
            cache_ns = (cache_namespace,)
        else:
            cache_ns = cache_namespace
        return WorkspaceContext(
            _job_spec=js,
            thread_id=thread_id,
            memory_namespace=memory_namespace or thread_id,
            cache_namespace=cache_ns,
            _store=store,
        )

    def child(self, name: str, tag: dict | None = None) -> "WorkspaceContext":
        """Create a child context with derived namespaces."""
        child_cache_ns = (*self.cache_namespace, name) if self.cache_namespace else None
        if child_cache_ns is not None and tag is not None:
            self._store.put(
                child_cache_ns,
                "_desc",
                tag
            )
        return WorkspaceContext(
            _job_spec=self._job_spec,
            thread_id=f"{self.thread_id}-{name}",
            memory_namespace=f"{self.memory_namespace}-{name}",
            cache_namespace=child_cache_ns,
            _store=self._store,
        )

    def indexed_child(self, name: str, index: int) -> "WorkspaceContext":
        """Create an indexed child context."""
        return self.child(f"{name}-{index}")

    def cache_get(self) -> dict | None:
        """Get a value from the cache. Returns None if caching disabled or not found."""
        if self.cache_namespace is None:
            return None
        
        if len(self.cache_namespace) < 1:
            raise ValueError("Cache prefix too small")
        
        full_key = self.cache_namespace[:-1]
        result = self._store.get(full_key, self.cache_namespace[-1])
        return result.value if result else None

    def cache_put(self, value: dict) -> None:
        """Put a value in the cache. No-op if caching disabled."""
        if self.cache_namespace is None:
            return
        if len(self.cache_namespace) < 1:
            raise ValueError("Cache prefix too small")
        
        full_key = self.cache_namespace[:-1]
        self._store.put(full_key, self.cache_namespace[-1], value)

    def get_memory_tool(self) -> BaseTool:
        """Get a memory tool for this context's memory namespace."""
        return memory_tool(get_memory(self.memory_namespace))
