"""
Shared fixtures for composer tool infrastructure tests.
"""


import uuid
from typing import AsyncIterator, Iterator, Callable, Iterable, TYPE_CHECKING
from contextlib import asynccontextmanager

import psycopg
import pytest
import pytest_asyncio

from langgraph.store.postgres.aio import AsyncPostgresStore
from langchain_core.tools import BaseTool
from langchain_core.language_models.fake import FakeListLLM

from psycopg.rows import dict_row
from psycopg.connection_async import AsyncConnection
from psycopg.sql import SQL, Identifier
from psycopg_pool.pool_async import AsyncConnectionPool as PGAsyncPool

from composer.kb.knowledge_base import DefaultEmbedder
from composer.spec.agent_index import AgentIndex
from composer.prover.core import SummarizedReport, RawReport
from composer.spec.source.prover import get_prover_tool, LLM

if TYPE_CHECKING:
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
# Testcontainers: Postgres + indexed store
# =========================================================================


@pytest.fixture(scope="session")
def pg_container() -> Iterator["PostgresContainer | None"]:
    if not _HAS_TESTCONTAINERS:
        return None
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg

@asynccontextmanager
async def _get_test_database(pg_container: "PostgresContainer", for_rag: bool = False) -> AsyncIterator[PGAsyncPool | None]:
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
    async with _get_test_database(pg_container) as pool:
        yield pool
    
@pytest_asyncio.fixture(scope="session")
async def session_pg_database(pg_container: "PostgresContainer | None") -> AsyncIterator[PGAsyncPool | None]:
    if pg_container is None:
        yield None
        return
    async with _get_test_database(pg_container, for_rag=True) as pool:
        yield pool

@pytest_asyncio.fixture
async def pg_database(pg_database_opt: PGAsyncPool | None) -> AsyncIterator[PGAsyncPool]:
    if _HAS_TESTCONTAINERS:
        pytest.skip("No pgcontainers")
    assert pg_database_opt is not None
    yield pg_database_opt

@pytest_asyncio.fixture
async def indexed_store(pg_database: PGAsyncPool) -> AsyncIterator[AsyncPostgresStore]:
    if _HAS_TESTCONTAINERS:
        pytest.skip("No pgcontainers")
    assert pg_database is not None
    store = AsyncPostgresStore(
        pg_database,
        index={
            "embed": DefaultEmbedder(),
            "dims": 768,
            "fields": None,
        },
    )
    await store.setup()
    yield store


@pytest_asyncio.fixture
async def index(indexed_store: AsyncPostgresStore) -> AgentIndex:
    return AgentIndex(indexed_store, ("test", "indexed_tool", uuid.uuid4().hex))

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