from typing import TypedDict, Unpack, cast, overload, override
import unicodedata
from langgraph.store.base import BaseStore
from graphcore.tools.schemas import WithAsyncDependencies
from abc import abstractmethod, ABC
from pydantic import Field

import hashlib

class AgentResult(TypedDict):
    question: str
    answer: str

class KeyedAgentResult(AgentResult):
    ref_string: str

class IndexedAgentResult(KeyedAgentResult):
    score: float


class AgentIndex:
    WITH_INDEX_SYS_COMMON = """
  When prior findings are provided alongside your task:                                                                                                                                         
                                                                                                                                                                                              
  1. If a prior finding directly answers your question, use it as-is. Do not rephrase, re-investigate, or "confirm" what has already been established.                                          
  2. If prior findings partially address your question, build on them. Use the established facts as your starting point and only investigate what remains unanswered.                         
  3. If no prior findings are relevant, proceed with fresh analysis.
                                                                                                                                                                                              
  Prior findings are prefixed with the original question that prompted them so you can judge their relevance to your current task."""
    

    def __init__(self, store: BaseStore, cache_ns: tuple[str, ...]):
        self.store = store
        self.cache_ns = cache_ns

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
    ) -> str:
        key = self._question_key(doc["question"])
        r = await self.store.aget(self.cache_ns, key)
        if r is not None:

            # silently dropping
            return key
        await self.store.aput(
            self.cache_ns, key, {**doc}, index=["answer"]
        )
        return key
    
    async def aget(
        self, key: str
    ) -> AgentResult | None:
        r = await self.store.aget(self.cache_ns, key)
        if r is None:
            return None
        return cast(AgentResult, r.value)
    
    async def asearch(
        self, question: str
    ) -> list[IndexedAgentResult] | KeyedAgentResult:
        key = self._question_key(question)
        cached = await self.aget(key)
        if cached is not None:
            return KeyedAgentResult(ref_string=key,  **cached)
        res = await self.store.asearch(
            self.cache_ns,
            query=question,
            limit=5
        )
        return [
            {
                **cast(AgentResult, r.value),
                "score": r.score, #type: ignore
                "ref_string": r.key
            } for r in res
        ]
    
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
