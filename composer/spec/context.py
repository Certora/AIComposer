from dataclasses import dataclass
import uuid
from pathlib import Path
import base64

from typing import Annotated, Protocol

from pydantic import BaseModel

from langgraph.store.postgres import PostgresStore
from langchain_core.tools import BaseTool

from graphcore.graph import LLM, Builder, FlowInput
from graphcore.tools.memory import memory_tool
from graphcore.tools.vfs import VFSState, VFSAccessor


# Phantom tool-capability markers. These tag Builder type aliases to document
# which tool sets have been bound. The type checker erases them (via Annotated)
# but they are visible in signatures for documentation.
class SOURCE_TOOLS:
    """Builder has fs_tools bound (source code file access)."""

class CVL_TOOLS:
    """Builder has cvl_manual_tools bound (CVL manual RAG search)."""

type SourceBuilder = Annotated[Builder[None, None, FlowInput], SOURCE_TOOLS]
type CVLOnlyBuilder = Annotated[Builder[None, None, FlowInput], CVL_TOOLS]
type CVLBuilder = Annotated[Builder[None, None, FlowInput], SOURCE_TOOLS, CVL_TOOLS]


@dataclass
class Builders:
    """Pre-configured builder variants with different tool sets."""
    source: SourceBuilder
    cvl: CVLBuilder
    cvl_only: CVLOnlyBuilder

from composer.workflow.services import get_memory


@dataclass
class ContractSpec:
    relative_path: str
    contract_name: str

@dataclass
class JobSpec(ContractSpec):
    project_root: str
    system_doc: str

class Services(Protocol):
    def llm(self) -> LLM:
        ...

    def kb_tools(self, read_only: bool) -> list[BaseTool]:
        ...
    
    def fs_tools(self) -> list[BaseTool]:
        ...

    def vfs_tools[S: VFSState](
        self,
        ty: type[S],
        forbidden_write: str | None = None,
        put_doc_extra: str | None = None
    ) -> tuple[list[BaseTool], VFSAccessor[S]]:
        ...

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
    _services: Services
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

    def kb_tools(self, read_only: bool) -> list[BaseTool]:
        return self._services.kb_tools(read_only)

    def fs_tools(self) -> list[BaseTool]:
        return self._services.fs_tools()

    def llm(self) -> LLM:
        return self._services.llm()

    def vfs_tools[S: VFSState](
        self,
        ty: type[S],
        forbidden_write: str | None = None,
        put_doc_extra: str | None = None
    ) -> tuple[list[BaseTool], VFSAccessor[S]]:
        return self._services.vfs_tools(ty, forbidden_write, put_doc_extra)

    def uniq_thread_id(self) -> str:
        suff = uuid.uuid4().hex[:16]
        return f"{self.thread_id}-{suff}"

    @staticmethod
    def create(
        js: JobSpec,
        services: Services,
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
            _services=services,
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
            _services=self._services,
            _job_spec=self._job_spec,
            thread_id=f"{self.thread_id}-{name}",
            memory_namespace=f"{self.memory_namespace}-{name}",
            cache_namespace=child_cache_ns,
            _store=self._store,
        )

    def indexed_child(self, name: str, index: int) -> "WorkspaceContext":
        """Create an indexed child context."""
        return self.child(f"{name}-{index}")

    def cache_get[T: BaseModel](self, ty: type[T]) -> T | None:
        """Get a typed value from the cache. Returns None if caching disabled or not found."""
        if self.cache_namespace is None:
            return None

        if len(self.cache_namespace) < 1:
            raise ValueError("Cache prefix too small")

        full_key = self.cache_namespace[:-1]
        result = self._store.get(full_key, self.cache_namespace[-1])
        if result is None:
            return None
        return ty.model_validate(result.value)

    def cache_put(self, value: BaseModel) -> None:
        """Put a typed value in the cache. No-op if caching disabled."""
        if self.cache_namespace is None:
            return
        if len(self.cache_namespace) < 1:
            raise ValueError("Cache prefix too small")

        full_key = self.cache_namespace[:-1]
        self._store.put(full_key, self.cache_namespace[-1], value.model_dump())

    def get_memory_tool(self) -> BaseTool:
        """Get a memory tool for this context's memory namespace."""
        return memory_tool(get_memory(self.memory_namespace))


def get_system_doc(sys_path: Path) -> dict | str | None:
    """Load a system document from a file path, returning base64-encoded PDF or text."""
    if not sys_path.is_file():
        print("System file not found")
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
