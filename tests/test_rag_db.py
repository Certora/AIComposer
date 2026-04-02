"""Tests for composer.rag.db — PostgreSQLRAGDatabase insert and retrieval."""

from typing import AsyncIterator, TYPE_CHECKING, Iterator
from dataclasses import dataclass, field

import numpy as np
import pytest
import pytest_asyncio
from numpy import ndarray
from psycopg_pool.pool_async import AsyncConnectionPool

from composer.rag.db import PostgreSQLRAGDatabase, ChromaRAGDatabase, ComposerRAGDB, BlockChunk
from composer.rag.text import code_ref_tag
from composer.rag.types import BlockChunk

from .conftest import MockSentenceTransformer, EMBEDDING_DIM

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# ── Test corpus ──

@dataclass
class MockManualSection:
    text: str
    question: str | None = None
    code_refs: list[str] = field(default_factory=list)

type ManualSectionPart = str | tuple[str, str] | MockManualSection

type TestCorpus = dict[
    tuple[str, ...],
    ManualSectionPart | list[ManualSectionPart]
]

_TEST_CORPUS : TestCorpus  = {
    ("CVL", "Invariants"): "Invariants must hold across all transactions in the smart contract.",
    ("CVL", "Invariants", "Checking"): (
         "How are invariants verified?",
         "Invariants are properties that must hold before and after every transaction. They are assumed in the prestate and asserted in the post state"
    ),
    ("CVL", "Rules"): "Rules describe expected behavior of specific contract functions.",
    ("CVL", "Rules", "Example"): MockManualSection(
        text = f"Example usage: {code_ref_tag(0)}",
        question = "show me a CVL example",
        code_refs=["rule sanity { assert true; }"]
    ),
    ("CVL", "Ghosts"): "Ghost variables track abstract state not present in the contract.",
    ("CVL", "Long Topic"): [
        "Part one of the long section.",
        "Part two of the long section.",
    ],
    ("CVL", "Hooks", "Syntax"): MockManualSection(
        code_refs=["hook Sstore uint256 uint256 {}"],
        text=f"Hook example: {code_ref_tag(0)}",
    ),
    ("CVL", "Hooks"): (
        "What are hooks in CVL?",
        "What are hooks in CVL?"
    )
}

def _random_unit_vector(rng: np.random.RandomState) -> ndarray:
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _perturb(vec: ndarray, rng: np.random.RandomState, epsilon: float = 0.05) -> ndarray:
    """Small perturbation of a unit vector — stays close in cosine distance."""
    noise = rng.randn(EMBEDDING_DIM).astype(np.float32)
    noise /= np.linalg.norm(noise)
    perturbed = vec + epsilon * noise
    perturbed /= np.linalg.norm(perturbed)
    return perturbed

def content_iter() -> Iterator[tuple[tuple[str, ...], int, MockManualSection]]:
    for (k, sec) in _TEST_CORPUS.items():
        if not isinstance(sec, list):
            sec = [sec]
        for (part, content) in enumerate(sec):
            if isinstance(content, tuple):
                yield (k, part, MockManualSection(text=content[1], question=content[0]))
            elif isinstance(content, str):
                yield (k, part, MockManualSection(text=content))
            else:
                yield (k, part, content)

@pytest.fixture(scope="session")
def mock_model() -> "SentenceTransformer":
    to_ret = MockSentenceTransformer()
    rng = np.random.RandomState(seed=123456789)
    for (_, _, sec) in content_iter():
        content_vector = _random_unit_vector(rng)
        content : str = sec.text
        if sec.question is not None:
            to_ret.register(f"search_query: {sec.question}", _perturb(content_vector, rng))
        to_ret.register(f"search_document: {content}", content_vector)
    return to_ret #type: ignore

@pytest_asyncio.fixture(scope="session")
async def postgres_rag(session_pg_database: AsyncConnectionPool | None, mock_model: "SentenceTransformer") -> AsyncIterator[PostgreSQLRAGDatabase | None]:
    if session_pg_database is None:
        yield None
        return
    to_ret = PostgreSQLRAGDatabase(
        conn_string=session_pg_database,
        model=mock_model
    )
    await to_ret.test_connection()
    yield to_ret

@pytest_asyncio.fixture(scope="session")
async def chroma_rag(tmp_path_factory: pytest.TempPathFactory, mock_model: "SentenceTransformer") -> AsyncIterator[ChromaRAGDatabase]:
    chroma_dir = tmp_path_factory.mktemp("chroma")
    async with ChromaRAGDatabase.rag_context(
        str(chroma_dir), mock_model
    ) as conn:
        yield conn

@pytest_asyncio.fixture(scope="session", params=["chroma", "postgres"])
async def rag_db_raw(
    request: pytest.FixtureRequest,
    postgres_rag: PostgreSQLRAGDatabase | None,
    chroma_rag: ChromaRAGDatabase
) -> AsyncIterator[ComposerRAGDB]:
    if request.param == "chroma":
        yield chroma_rag
    else:
        if postgres_rag is None:
            pytest.skip("PG Containers not installed")
        yield postgres_rag

@pytest_asyncio.fixture(scope="session")
async def rag_db(
    rag_db_raw: ComposerRAGDB
) -> AsyncIterator[ComposerRAGDB]:
    for (sec, part, content) in content_iter():
        mock_chunk = BlockChunk(
            code_refs=content.code_refs,
            part=part,
            chunk=content.text,
            headers=list(sec)
        )
        await rag_db_raw.add_manual_section(
            mock_chunk
        )
        await rag_db_raw.add_chunks_batch([mock_chunk])
    yield rag_db_raw

# ── Manual section tests (no embeddings needed) ──


@pytest.mark.asyncio
class TestManualSections:
    async def test_retrieve_by_headers(self, rag_db: ComposerRAGDB) -> None:
        result = await rag_db.get_manual_section(["CVL", "Invariants"])
        assert result is not None
        assert "Invariants must hold across all transactions" in result

    async def test_retrieve_nonexistent(self, rag_db: ComposerRAGDB) -> None:
        result = await rag_db.get_manual_section(["Nonexistent", "Section"])
        assert result is None

    async def test_multipart_section(self, rag_db: ComposerRAGDB) -> None:
        result = await rag_db.get_manual_section(["CVL", "Long Topic"])
        assert result is not None
        assert "Part one" in result
        assert "Part two" in result

    async def test_code_refs_expanded(self, rag_db: ComposerRAGDB) -> None:
        result = await rag_db.get_manual_section(["CVL", "Hooks", "Syntax"])
        assert result is not None
        assert "hook Sstore uint256 uint256 {}" in result
        assert code_ref_tag(0) not in result

    async def test_keyword_search(self, rag_db: ComposerRAGDB) -> None:
        hits = await rag_db.search_manual_keywords("invariants")
        assert len(hits) > 0
        assert hits[0].headers == ["CVL", "Invariants"]


# ── Embedding / vector search tests ──


@pytest.mark.asyncio
class TestFindRefs:
    async def test_find_matching_document(self, rag_db: ComposerRAGDB) -> None:
        # "How are invariants verified?" is a registered query near the
        # Invariants/Checking doc vector
        results = await rag_db.find_refs("How are invariants verified?", similarity_cutoff=0.9)
        assert len(results) >= 1
        assert results[0].similarity > 0.99
        assert "Invariants are properties" in results[0].content

    async def test_best_match_ranked_first(self, rag_db: ComposerRAGDB) -> None:
        results = await rag_db.find_refs(
            "How are invariants verified?", similarity_cutoff=0.3, top_k=10
        )
        assert len(results) >= 1
        # The closest doc should be the invariants entry
        assert "Invariants" in results[0].content
        assert results[0].similarity > 0.99
        # Any additional results should be less similar
        for r in results[1:]:
            assert r.similarity < results[0].similarity

    async def test_similarity_cutoff_filters(self, rag_db: ComposerRAGDB) -> None:
        # Even the best Q/A pair has similarity ~0.999, not 1.0 —
        # a cutoff of 0.9999 should exclude everything
        results = await rag_db.find_refs(
            "How are invariants verified?", similarity_cutoff=0.9999
        )
        assert len(results) == 0

    async def test_code_refs_inlined_in_results(self, rag_db: ComposerRAGDB) -> None:
        # "show me a CVL example" is near the Rules/Example doc which has a code ref
        results = await rag_db.find_refs("show me a CVL example", similarity_cutoff=0.9)
        assert len(results) >= 1
        assert "rule sanity { assert true; }" in results[0].content
        assert code_ref_tag(0) not in results[0].content
