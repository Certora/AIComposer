import asyncio
import hashlib
import os
import unicodedata
from dataclasses import dataclass
from typing import TypedDict, Unpack, cast, overload, override
from abc import abstractmethod, ABC

from langgraph.store.base import BaseStore
from graphcore.tools.schemas import WithAsyncDependencies
from pydantic import Field

class AgentResult(TypedDict):
    question: str
    answer: str

class KeyedAgentResult(AgentResult):
    ref_string: str

class IndexedAgentResult(KeyedAgentResult):
    score: float


@dataclass(frozen=True)
class AgentIndexConfig:
    """Read/write policy for an ``AgentIndex``.

    Three modes, expressed as combinations of ``tenant_id`` and
    ``read_only``:

    - ``tenant_id`` set, ``read_only`` False — **tiered**: writes land in
      the per-tenant namespace; reads check tenant first, then global.
      The default for customer-facing deployments. A poisoned answer
      written by tenant A is invisible to tenant B (writes don't escape
      the tenant pool); only the offline promotion pipeline can move
      entries from a tenant ns into the global ns.
    - ``tenant_id`` None, ``read_only`` False — **trusted write-through**:
      writes go directly to the global ns. Intended for internal /
      curating operators whose outputs are trusted to seed the shared
      pool.
    - ``tenant_id`` None, ``read_only`` True — **read-only**: writes are
      silently dropped, reads consult global only. Useful for clients
      that should consume the shared knowledge but never augment it.
    """

    tenant_id: str | None = None
    read_only: bool = False


def agent_index_config_from_env() -> AgentIndexConfig:
    """Build an ``AgentIndexConfig`` from process env.

    ``AUTOPROVER_AGENT_INDEX_MODE`` selects between ``tiered``,
    ``trusted``, and ``readonly`` (default: ``trusted`` — preserves the
    current write-everything-to-global behavior so internal workflows
    don't change without opting in). ``tiered`` additionally requires
    ``AUTOPROVER_USER_ID`` to be set.
    """
    mode = os.environ.get("AUTOPROVER_AGENT_INDEX_MODE", "tiered").lower()
    uid = os.environ.get("AUTOPROVER_USER_ID")

    if mode == "trusted":
        return AgentIndexConfig(tenant_id=None, read_only=False)
    if mode == "readonly":
        return AgentIndexConfig(tenant_id=None, read_only=True)
    if mode == "tiered":
        if not uid:
            raise ValueError(
                "AUTOPROVER_AGENT_INDEX_MODE=tiered requires "
                "AUTOPROVER_USER_ID to be set."
            )
        return AgentIndexConfig(tenant_id=uid, read_only=False)
    raise ValueError(
        f"Unknown AUTOPROVER_AGENT_INDEX_MODE: {mode!r}. "
        "Expected one of: tiered, trusted, readonly."
    )


class AgentIndex:
    """Two-tier semantic cache.

    The index has a ``global_ns`` (shared, curated pool) and an optional
    ``tenant_ns`` (per-caller pool). Reads consult the tenant ns first
    on exact-key lookup and merge both pools on vector search. Writes
    target the tenant ns when set, or fall back to the global ns when
    not. ``read_only=True`` drops writes entirely.

    Within any single ns, ``aput`` is first-write-wins on the
    normalized-question key — the same semantics as the original
    single-pool implementation. The expectation is that a separate
    offline pipeline curates promotions from tenant pools to the
    global pool.
    """

    WITH_INDEX_SYS_COMMON = """
  When prior findings are provided alongside your task:

  1. If a prior finding directly answers your question, use it as-is. Do not rephrase, re-investigate, or "confirm" what has already been established.
  2. If prior findings partially address your question, build on them. Use the established facts as your starting point and only investigate what remains unanswered.
  3. If no prior findings are relevant, proceed with fresh analysis.

  Prior findings are prefixed with the original question that prompted them so you can judge their relevance to your current task."""

    def __init__(
        self,
        store: BaseStore,
        global_ns: tuple[str, ...],
        *,
        tenant_ns: tuple[str, ...] | None = None,
        read_only: bool = False,
    ):
        self.store = store
        self.global_ns = global_ns
        self.tenant_ns = tenant_ns if tenant_ns != global_ns else None
        self.read_only = read_only

        self._write_ns = None if self.read_only else (self.tenant_ns if self.tenant_ns is not None else self.global_ns)

    @property
    def _read_pools(self) -> list[tuple[str, ...]]:
        # Tenant first so a per-tenant entry takes precedence on exact-key
        # lookup. Global is always consulted as the shared fallback.
        if self.tenant_ns is None:
            return [self.global_ns]
        return [self.tenant_ns, self.global_ns]

    @classmethod
    def with_config(
        cls,
        store: BaseStore,
        global_ns: tuple[str, ...],
        config: AgentIndexConfig,
    ) -> "AgentIndex":
        """Build an ``AgentIndex`` from a global namespace plus an
        ``AgentIndexConfig``. The config's ``tenant_id`` (when set) is
        appended to ``global_ns`` to derive the tenant namespace."""
        tenant_ns = (
            global_ns + (config.tenant_id,) if config.tenant_id else None
        )
        return cls(
            store=store,
            global_ns=global_ns,
            tenant_ns=tenant_ns,
            read_only=config.read_only,
        )

    def _normalize(self, text: str) -> str:
        nfkc = unicodedata.normalize("NFKC", text).casefold()
        stripped = "".join(c for c in nfkc if not unicodedata.category(c).startswith("P"))
        return " ".join(stripped.split())

    def _question_key(
        self, question: str
    ) -> str:
        return hashlib.sha256(self._normalize(question).encode()).hexdigest()[18:]

    async def aput(
        self,
        **doc: Unpack[AgentResult]
    ) -> str | None:
        """Persist *doc* and return its lookup key, or ``None`` in read-only
        mode (because the entry isn't durable and a Document-Ref pointing
        at it would dangle the moment a downstream ``cvl_document_ref`` /
        ``code_document_ref`` tried to resolve it)."""
        write_ns = self._write_ns
        if write_ns is None:
            return None
        key = self._question_key(doc["question"])
        existing = await self.store.aget(write_ns, key)
        if existing is not None:
            # First-write-wins within the chosen ns.
            return key
        await self.store.aput(
            write_ns, key, {**doc}, index=["answer"]
        )
        return key

    async def aget(
        self, key: str
    ) -> AgentResult | None:
        # Tenant ns shadows global on exact-key match.
        for ns in self._read_pools:
            r = await self.store.aget(ns, key)
            if r is not None:
                return cast(AgentResult, r.value)
        return None
    
    @dataclass
    class _ListIter[T]:
        wrapped: list[T]
        ptr: int = 0

        def peek(self) -> T | None:
            if self.ptr >= len(self.wrapped):
                return None
            return self.wrapped[self.ptr]
        
        def pop(self) -> T:
            to_ret = self.wrapped[self.ptr]
            self.ptr += 1
            return to_ret

    async def asearch(
        self, question: str
    ) -> list[IndexedAgentResult] | KeyedAgentResult:
        key = self._question_key(question)
        cached = await self.aget(key)
        if cached is not None:
            return KeyedAgentResult(ref_string=key,  **cached)
        # Vector search runs in parallel across both pools. Scores share
        # the same metric (cosine similarity), so merging by score is
        # meaningful. Dedup defends against the same key existing in both
        # pools (which a manual / offline promotion may produce).
        pool_results = await asyncio.gather(*[
            self.store.asearch(ns, query=question, limit=5)
            for ns in self._read_pools
        ])
        result_pointers = [
            AgentIndex._ListIter(l) for l in pool_results
        ]
        context : list[IndexedAgentResult] = []
        seen = set()
        while True:
            query = ((i, peeked) for (i, it) in enumerate(result_pointers) if (peeked := it.peek()) is not None)
            m = max(query, key=lambda r: cast(float, r[1].score), default=None)
            if m is None:
                return context
            popped = result_pointers[m[0]].pop()
            if popped.key in seen:
                continue
            seen.add(popped.key)
            context.append({
                **cast(AgentResult, popped.value),
                "score": cast(float, popped.score),
                "ref_string": popped.key
            })
            if len(context) == 5:
                return context
    
    @overload
    @staticmethod
    def format_document(
        doc: str,
        ref_key: str
    ) -> str:
        ...

    @overload
    @staticmethod
    def format_document(
        doc: KeyedAgentResult
    ) -> str:
        ...

    @staticmethod
    def format_document(
        doc: str | KeyedAgentResult,
        ref_key: str | None = None
    ) -> str:
        if isinstance(doc, dict):
            ref_key = doc["ref_string"]
            doc = doc["answer"]
        
        return f"{doc}\n\nDocument-Ref: {ref_key}"

    @staticmethod
    def format_context(
        corpus: list[IndexedAgentResult],
        empty_res: str = "No matching prior results found",
        include_ref: bool = False
    ) -> list[str]:
        if len(corpus) == 0:
            return [empty_res]
        
        docs = []
        for (i, d) in enumerate(corpus):
            ref = f"\nDocument-Ref: {d["ref_string"]}" if include_ref else ""
            docs.append(
f"""
---- Match {i}
{ref}
**Similarity**: {d["score"]}
**Question**: {d["question"]}

**Answer**:

{d["answer"]}

---- END Match {i}
"""
)
            
        return docs

class WithAgentIndex(TypedDict):
    ind: AgentIndex

class IndexedTool[T: WithAgentIndex | AgentIndex](WithAsyncDependencies[str, T], ABC):
    @abstractmethod
    def get_question(self) -> str:
        ...

    @abstractmethod
    async def answer_question(self, context: list[str]) -> str:
        ...

    async def run(self) -> str:
        with self.tool_deps() as ind:
            if isinstance(ind, dict):
                ind = ind["ind"]
            q = self.get_question()
            prior_match = await ind.asearch(
                question=q
            )
            if isinstance(prior_match, dict):
                return f"""
{prior_match['answer']}

Document-Ref: {prior_match['ref_string']}
"""
            context = AgentIndex.format_context(prior_match)
            answer = await self.answer_question(
                context
            )

            ref_key = await ind.aput(
                question=q,
                answer=answer
            )
            if ref_key is None:
                # Read-only mode: the answer isn't durable, so no
                # Document-Ref can be surfaced.
                return answer
            return f"{answer}\n\nDocument-Ref: {ref_key}"
        
class RetrieveDocumentTool(WithAsyncDependencies[str, AgentIndex]):
    """
    Retrieve the document associated with the provided document ref
    """
    ref: str = Field(description="The document reference id")

    @override
    async def run(self) -> str:
        with self.tool_deps() as dep:
            res = await dep.aget(self.ref)
            if res is None:
                return "Document not found"
            return f"**Question**: {res["question"]}\n\n**Answer**:\n{res["answer"]}"
