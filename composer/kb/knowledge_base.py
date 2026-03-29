from typing import override, TypedDict, cast, TYPE_CHECKING, Literal
from enum import StrEnum

from pydantic import Field

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from graphcore.tools.schemas import WithAsyncImplementation

from composer.workflow.services import Embeddings

from composer.rag.models import get_model

# tell the type checker we always import ST
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    # we're probably in test, in which case just gracefully pretend ST doesn't exist
    try:
        from sentence_transformers import SentenceTransformer #type: ignore
    except ImportError:
        pass

# tqdm tries to create a multiprocessing.RLock on first use, which calls
# fork_exec to start a resource tracker process.  In an async event loop
# with open DB connections this fails with "bad value(s) in fds_to_keep"
# and eventually hangs.  Pre-set a threading lock so tqdm never attempts
# the fork.  This is the narrowest fix: no env-var side effects, and
# sentence_transformers (which uses tqdm internally) just works.
import threading
from tqdm import tqdm as _tqdm_cls
_tqdm_cls.set_lock(threading.RLock())

DEFAULT_KB_NS = ("cvl",)

class ReviewStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"

    @property
    def sort_key(self) -> int:
        return {ReviewStatus.APPROVED: 0, ReviewStatus.PENDING_REVIEW: 1, ReviewStatus.REJECTED: 2}[self]

    @property
    def display_tag(self) -> str:
        """Tag string for agent-facing output. Empty for approved articles."""
        if self == ReviewStatus.APPROVED:
            return ""
        return f" [{self.name.replace('_', ' ')}]"

ArticleSource = Literal["human", "agent"]

def get_review_status(article: dict) -> ReviewStatus: # type: ignore[type-arg]
    try:
        return ReviewStatus(article.get("review_status", "pending_review"))
    except ValueError:
        return ReviewStatus.PENDING_REVIEW

class DefaultEmbedder(Embeddings):
    def __init__(self, model: "SentenceTransformer | None" = None):
        self.model : "SentenceTransformer" = get_model() if not model else model

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode_document(
            texts
        ).tolist() #type: ignore

    @override
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode_query(
            [text]
        ).tolist()[0] #type: ignore

class KnowledgeBaseArticle(TypedDict):
    title: str
    symptom: str
    body: str
    review_status: ReviewStatus
    source: ArticleSource

def kb_tools(store: BaseStore, kb_ns: tuple[str, ...], read_only: bool) -> list[BaseTool]:
    kb = kb_ns + ("agent", "knowledge")
    class KBScan(WithAsyncImplementation[str]):
        """
        Scan the knowledge base articles for help with a given problem
        """
        symptom: str | None = Field(description="A short, description of the problem you are facing. None if you want to search all knowledge base articles")
        limit: int | None = Field(description="Limit the number of results to return. Defaults to 10 if not provided")
        offset: int | None = Field(description="Page through the results. Defaults to 0")

        @override
        async def run(self) -> str:
            lim = self.limit if self.limit else 10
            offset = self.offset if self.offset else 0
            if self.symptom is None:
                r = await store.asearch(
                    kb,
                    offset=offset,
                    limit=lim
                )
                r = [it for it in r if get_review_status(it.value) != ReviewStatus.REJECTED]
                r.sort(key=lambda it: get_review_status(it.value).sort_key)
                to_ret = []
                for it in r:
                    as_name = cast(KnowledgeBaseArticle, it.value)
                    status = get_review_status(it.value)
                    to_ret.extend([
                        f"Title: {as_name['title']}{status.display_tag}",
                        f"Symptom: {as_name['symptom']}"
                    ])
                return "\n".join(to_ret)
            else:
                r = await store.asearch(
                    kb,
                    query=self.symptom,
                    limit=lim,
                    offset=offset
                )
                to_ret = []
                for it in r:
                    as_name = cast(KnowledgeBaseArticle, it.value)
                    status = get_review_status(it.value)
                    if status == ReviewStatus.REJECTED:
                        continue
                    to_ret.extend([
                        f"Title: {as_name['title']}{status.display_tag}",
                        f"Symptom: {as_name['symptom']}",
                        f"Similarity: {it.score}"
                    ])
                return "\n".join(to_ret)

    class KBGet(WithAsyncImplementation[str]):
        """
        Retrieve the contents of a knowledge base article
        """
        title: str = Field(description="The title of the article")

        @override
        async def run(self) -> str:
            r = await store.aget(kb, self.title)
            if r is None:
                return f"No such article with title '{self.title}'"
            status = get_review_status(r.value)
            if status == ReviewStatus.REJECTED:
                return f"No such article with title '{self.title}'"
            art = cast(KnowledgeBaseArticle, r.value)
            warning = ""
            if status == ReviewStatus.PENDING_REVIEW:
                warning = "\n> **Note:** This article has not yet been reviewed by a human. Treat its advice with caution.\n"
            return f"""
## {art['title']}
{warning}
*Symptom*: {art['symptom']}

{art['body']}
"""

    class KBPut(WithAsyncImplementation[str]):
        """
        Add a novel, non-trivial insight to the knowledge base.

        Use this when you have synthesized a non-obvious answer to a non-trivial problem you have been facing.
        Put another way, use this to summarize conclusions you have drawn from authoritative sources
        (the CVL manual, prover output, human feedback, etc.)

        IMPORTANT: Only use this tool for factual, verified solutions. Do *NOT* store any knowledge which is
        speculative, or unsupported by verifiable, empirical findings.

        *DO NOT* use this tool to store information around: syntax reminders, information already present and easily found in the CVL manual,
        problems you solved with trial-and-error without a strong understanding of _why_ the fix worked.

        Before submitting your article, search the knowledge base for other similar articles to avoid duplication. Articles are keyed based
        on the "symptom" field.

        *BAD* Example: "You need to put semi-colons at the end of function block declarations in CVL" (solution is trivially discoverable)
        *BAD* Example: "The prover didn't work, so I added `require` statements the rules until the spec passed" (vague, speculative, non-actionable)
        *GOOD* Example: "Invariants consistently failed due to spurious counter examples; the solution was strengthening the property
          to assume invariants that ruled out the spurious states" (specific, backed up by empirical experience with the prover)
        """
        title: str = Field(description="A descriptive title of the knowledge base article")
        symptom: str = Field(description="The observable behavior another agent might encounter (error messages, unexpected prover output, verification failures), " \
        "not the abstract concept. Write it as if the reader is searching for help with a specific problem")
        body: str = Field(description="A markdown formatted article body describing your solution. IMPORTANT: The advice you give must be backed up by verifiable facts.")

        @override
        async def run(self) -> str:
            it = await store.aget(kb, self.title)
            if it is not None:
                return f"An article with the title {self.title} already exists"
            article : KnowledgeBaseArticle = {
                "title": self.title,
                "symptom": self.symptom,
                "body": self.body,
                "review_status": ReviewStatus.PENDING_REVIEW,
                "source": "agent"
            }
            await store.aput(kb, self.title, cast(dict, article), index=["symptom"])
            return "Contribution accepted (pending human review)."

    to_ret : list[BaseTool] = [KBScan.as_tool("scan_knowledge_base"), KBGet.as_tool("get_knowledge_base_article")]
    if not read_only:
        to_ret.append(KBPut.as_tool("knowledge_base_contribute"))
    return to_ret
