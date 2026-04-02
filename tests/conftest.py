"""
Shared fixtures for composer tool infrastructure tests.
"""


import uuid
from typing import Any, AsyncIterator, Iterator, Callable, Iterable, TYPE_CHECKING, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import psycopg
import pytest
import pytest_asyncio
from numpy import ndarray

from langchain_core.tools import BaseTool
from langchain_core.language_models.fake import FakeListLLM

from psycopg.rows import dict_row
from psycopg.connection_async import AsyncConnection
from psycopg.sql import SQL, Identifier
from psycopg_pool.pool_async import AsyncConnectionPool as PGAsyncPool

from composer.prover.core import SummarizedReport, RawReport
from composer.spec.source.prover import get_prover_tool, LLM

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from testcontainers.postgres import PostgresContainer

try:
    from testcontainers.postgres import PostgresContainer

    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False

needs_postgres = pytest.mark.skipif(
    not _HAS_TESTCONTAINERS,
    reason="testcontainers[postgres] not installed",
)


# =========================================================================
# Mock embedding model
# =========================================================================

EMBEDDING_DIM = 768

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


class MockSentenceTransformer:
    """Fake SentenceTransformer that returns pre-registered vectors, falling
    back to deterministic hash-based vectors for unknown text.

    Handles both calling conventions:
    - RAG DB: ``encode_query(str) -> 1-D ndarray``
    - DefaultEmbedder: ``encode_query([str, ...]) -> 2-D ndarray``
    """

    def __init__(self) -> None:
        self._vectors: dict[str, ndarray] = {}

    def register(self, text: str, vector: ndarray) -> None:
        self._vectors[text] = vector

    def _resolve(self, text: str) -> ndarray:
        if text in self._vectors:
            return self._vectors[text]
        import hashlib
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4])
        rng = np.random.RandomState(seed)
        vec = _random_unit_vector(rng)
        return vec

    def encode_query(self, text: str | list[str], **_: Any) -> ndarray:
        if isinstance(text, list):
            return np.array([self._resolve(t) for t in text])
        return self._resolve(text)

    def encode_document(self, texts: list[str], **_: Any) -> ndarray:
        return np.array([self._resolve(t) for t in texts])

@dataclass
class QnATransformer:
    underlying: MockSentenceTransformer
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState())

    @property
    def as_transformer(self) -> "SentenceTransformer":
        return self.underlying #type: ignore[trust me bro]

    def register(self, doc: str, questions: Sequence[str] = ()):
        answer_doc = _random_unit_vector(self.rng)
        self.underlying.register(doc, answer_doc)
        for q in questions:
            self.underlying.register(q, _perturb(answer_doc, self.rng))

@pytest.fixture(scope="session")
def qna_factory() -> Callable[[], QnATransformer]:
    return lambda: QnATransformer(MockSentenceTransformer())

# =========================================================================
# Testcontainers: Postgres + indexed store
# =========================================================================


@pytest.fixture(scope="session")
def pg_container() -> Iterator["PostgresContainer | None"]:
    if not _HAS_TESTCONTAINERS:
        return None
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg

@asynccontextmanager
async def get_test_database(pg_container: "PostgresContainer", for_rag: bool = False) -> AsyncIterator[PGAsyncPool | None]:
    uniq_db = "test_store_" + uuid.uuid4().hex[:16]
    admin_url = pg_container.get_connection_url(driver=None)

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("CREATE DATABASE {}").format(Identifier(uniq_db)))

    conn_string = (
        f"postgresql://{pg_container.username}:{pg_container.password}"
        f"@{pg_container.get_container_host_ip()}"
        f":{pg_container.get_exposed_port(5432)}/{uniq_db}"
    )

    res = PGAsyncPool(
        conn_string,
        connection_class=AsyncConnection,
        kwargs={"autocommit": True, "row_factory": dict_row} if not for_rag else {},
    )    
    async with res:
        yield res

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("DROP DATABASE {}").format(Identifier(uniq_db)))


@pytest_asyncio.fixture
async def pg_database_opt(pg_container: "PostgresContainer | None") -> AsyncIterator[PGAsyncPool | None]:
    if pg_container is None:
        yield None
        return
    async with get_test_database(pg_container) as pool:
        yield pool

@pytest_asyncio.fixture(scope="session")
async def session_pg_database(pg_container: "PostgresContainer | None") -> AsyncIterator[PGAsyncPool | None]:
    if pg_container is None:
        yield None
        return
    async with get_test_database(pg_container, for_rag=True) as pool:
        yield pool


@pytest_asyncio.fixture
async def pg_database(pg_database_opt: PGAsyncPool | None) -> AsyncIterator[PGAsyncPool]:
    if not _HAS_TESTCONTAINERS:
        pytest.skip("No pgcontainers")
    assert pg_database_opt is not None
    yield pg_database_opt

type ProverToolResponse = SummarizedReport | RawReport | str
type ProverMock = Callable[[Iterable[ProverToolResponse]], BaseTool]

@pytest.fixture
def fake_llm():
    return FakeListLLM(responses=["Foo", "Bar"])

@pytest.fixture
def certora_prover(
    tmp_path,
    fake_llm: LLM,
    monkeypatch
) -> ProverMock:
    response_script : list[ProverToolResponse] | None = None
    response_ptr = 0

    async def mock_prover(
        *args, **kwargs
    ) -> ProverToolResponse:
        assert response_script is not None
        nonlocal response_ptr
        assert response_ptr < len(response_script)
        to_ret = response_script[response_ptr]
        response_ptr += 1
        return to_ret
    
    monkeypatch.setattr("composer.spec.source.prover.run_prover", mock_prover)
    monkeypatch.setattr("composer.spec.source.prover.get_stream_writer", lambda: (
        lambda _: None
    ))

    the_tool = get_prover_tool(
        cloud=None,
        llm=fake_llm,
        main_contract="Dummy",
        project_root=str(tmp_path),
    )

    def bind_tool(l: Iterable[ProverToolResponse]) -> BaseTool:
        nonlocal response_script
        response_script = list(l)
        return the_tool

    return bind_tool