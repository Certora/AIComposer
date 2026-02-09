from typing import override, TypedDict, cast

from pydantic import Field

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from graphcore.tools.schemas import WithImplementation

from composer.workflow.services import Embeddings

from composer.rag.models import get_model

from sentence_transformers import SentenceTransformer #type: ignore

class DefaultEmbedder(Embeddings):
    def __init__(self, model: SentenceTransformer | None = None):
        self.model : SentenceTransformer = get_model() if not model else model

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

def kb_tools(store: BaseStore, kb_ns: tuple[str, ...], read_only: bool) -> list[BaseTool]:
    kb = kb_ns + ("agent", "knowledge")
    class KBScan(WithImplementation[str]):
        """
        Scan the knowledge base articles for help with a given problem
        """
        symptom: str | None = Field(description="A short, description of the problem you are facing. None if you want to search all knowledge base articles")
        limit: int | None = Field(description="Limit the number of results to return. Defaults to 10 if not provided")
        offset: int | None = Field(description="Page through the results. Defaults to 0")

        @override
        def run(self) -> str:
            lim = self.limit if self.limit else 10
            offset = self.offset if self.offset else 0
            if self.symptom is None:
                r = store.search(
                    kb,
                    offset=offset,
                    limit=lim
                )
                to_ret = []
                for it in r:
                    as_name = cast(KnowledgeBaseArticle, it.value)
                    to_ret.extend([
                        f"Title: {as_name['title']}",
                        f"Symptom: {as_name['symptom']}"
                    ])
                return "\n".join(to_ret)
            else:
                r = store.search(
                    kb,
                    query=self.symptom,
                    limit=lim,
                    offset=offset
                )
                to_ret = []
                for it in r:
                    as_name = cast(KnowledgeBaseArticle, it.value)
                    to_ret.extend([
                        f"Title: {as_name['title']}",
                        f"Symptom: {as_name['symptom']}",
                        f"Similarity: {it.score}"
                    ])
                return "\n".join(to_ret)
        
    class KBGet(WithImplementation[str]):
        """
        Retrieve the contents of a knowledge base article
        """
        title: str = Field(description="The title of the article")

        @override
        def run(self) -> str:
            r = store.get(kb, self.title)
            if r is None:
                return f"No such article with title '{self.title}'"
            art = cast(KnowledgeBaseArticle, r.value)
            return f"""
## {art['title']}

*Symptom*: {art['symptom']}

{art['body']}
"""
        
    class KBPut(WithImplementation[str]):
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
        def run(self) -> str:
            it = store.get(kb, self.title)
            if it is not None:
                return f"An article with the title {self.title} already exists"
            store.put(kb, self.title, {
                "title": self.title,
                "description": self.symptom,
                "body": self.body
            }, index=["description"])
            return "Contribution accepted."
    
    to_ret : list[BaseTool] = [KBScan.as_tool("scan_knowledge_base"), KBGet.as_tool("get_knowledge_base_article")]
    if not read_only:
        to_ret.append(KBPut.as_tool("knowledge_base_contribute"))
    return to_ret