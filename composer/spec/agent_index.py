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

from composer.core.user import get_uid, user_data_ns

class AgentResult(TypedDict):
    question: str
    answer: str

class KeyedAgentResult(AgentResult):
    ref_string: str

class IndexedAgentResult(KeyedAgentResult):
    score: float

_UNSAFE_DISABLE_CACHE = False

@dataclass(frozen=True)
class AgentIndexConfig:
    """Full specification for an ``AgentIndex``.

    ``base_layer`` is mandatory and is always consulted on reads. It's
    also the default write target. ``write_layer`` is an optional
    overlay: when set, it's consulted first on reads and is the sole
    write target. ``read_only`` drops writes entirely.

    Three useful modes:

    - ``write_layer`` set, ``read_only`` False — **tiered**: writes
      isolated to the overlay; reads see overlay-then-base. The overlay
      is typically a per-tenant slot built via :func:`user_data_ns`.
    - ``write_layer`` None, ``read_only`` False — **write-through**:
      writes land in ``base_layer`` directly. The trusted-operator
      case.
    - ``write_layer`` None, ``read_only`` True — **read-only**: writes
      silently dropped, reads consult ``base_layer`` only.
    """

    base_layer: tuple[str, ...]
    write_layer: tuple[str, ...] | None = None
    read_only: bool = False


def agent_index_config_from_env(data_ns: tuple[str, ...]) -> AgentIndexConfig:
    """Build a config from env, suitable for indexes that have a shared
    ``base_layer`` plus an optional per-user overlay.

    ``AUTOPROVER_AGENT_INDEX_MODE`` selects between ``tiered``,
    ``trusted``, and ``readonly`` (default: ``trusted``). ``tiered``
    additionally requires ``AUTOPROVER_USER_ID``; the overlay is
    constructed as ``user_data_ns(uid) + data_ns``. ``data_ns`` is the
    *kind* of data this index stores (e.g. ``("cvl_research",
    "cached")``); typically it's the same tuple the caller will pass
    as the index's ``base_layer``.
    """
    mode = os.environ.get("AUTOPROVER_AGENT_INDEX_MODE", "tiered").lower()

    if mode == "trusted":
        return AgentIndexConfig(base_layer=data_ns, read_only=False, write_layer=None)
    if mode == "readonly":
        return AgentIndexConfig(base_layer=data_ns, read_only=True, write_layer=None)
    if mode == "tiered":
        return AgentIndexConfig(
            base_layer=data_ns,
            read_only=False,
            write_layer=user_data_ns() + data_ns,
        )
    raise ValueError(
        f"Unknown AUTOPROVER_AGENT_INDEX_MODE: {mode!r}. "
        "Expected one of: tiered, trusted, readonly."
    )


class AgentIndex:
    """Two-layer semantic cache.

    ``base_layer`` is always consulted on reads and is the default
    write target. When ``config.write_layer`` is set, that overlay is
    consulted first on reads and is the sole write target. When
    ``config.read_only`` is true, writes are dropped silently.

    Within any single layer, ``aput`` is first-write-wins on the
    normalized-question key. The expectation is that a separate
    offline pipeline curates promotions from per-user overlays into a
    shared ``base_layer`` when applicable.

    The choice of ``base_layer`` and ``write_layer`` namespaces is the
    caller's responsibility — see :func:`user_data_ns` and
    :func:`agent_index_config_from_env` for the conventional way to
    build them. Note that langgraph store backends do prefix-matching
    on ``asearch`` (InMemoryStore via tuple slicing, Postgres via
    ``prefix LIKE``), so ``write_layer`` should not be a descendant of
    ``base_layer`` — otherwise overlay rows leak into base-layer search
    results.
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
        config: AgentIndexConfig,
    ):
        self.store = store
        self.base_layer = config.base_layer
        # Treat a write_layer that equals the base as "no overlay" so
        # the read path doesn't double-consult the same namespace.
        self.write_layer = (
            config.write_layer
            if config.write_layer is not None and config.write_layer != config.base_layer
            else None
        )
        self.read_only = config.read_only

        self._write_ns = (
            None
            if self.read_only
            else (self.write_layer if self.write_layer is not None else self.base_layer)
        )

    @property
    def _read_pools(self) -> list[tuple[str, ...]]:
        # write_layer first so an overlay entry takes precedence on
        # exact-key lookup; base_layer is always consulted as fallback.
        if self.write_layer is None:
            return [self.base_layer]
        return [self.write_layer, self.base_layer]

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
        if cached is not None and not _UNSAFE_DISABLE_CACHE:
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
