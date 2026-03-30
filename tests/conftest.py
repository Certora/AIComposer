"""
Shared fixtures for composer tool infrastructure tests.
"""


import uuid
from typing import AsyncIterator, Iterator, Callable, Iterable

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
def pg_container() -> Iterator[PostgresContainer]:
    if not _HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed")
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg


@pytest_asyncio.fixture
async def indexed_store(pg_container: PostgresContainer) -> AsyncIterator[AsyncPostgresStore]:
    uniq_db = "test_store_" + uuid.uuid4().hex[:16]
    admin_url = pg_container.get_connection_url(driver=None)

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("CREATE DATABASE {}").format(Identifier(uniq_db)))

    conn_string = (
        f"postgresql://{pg_container.username}:{pg_container.password}"
        f"@{pg_container.get_container_host_ip()}"
        f":{pg_container.get_exposed_port(5432)}/{uniq_db}"
    )

    pool = PGAsyncPool(
        conn_string,
        connection_class=AsyncConnection,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    async with pool:
        store = AsyncPostgresStore(
            pool,
            index={
                "embed": DefaultEmbedder(),
                "dims": 768,
                "fields": None,
            },
        )
        await store.setup()
        yield store

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(SQL("DROP DATABASE {}").format(Identifier(uniq_db)))


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
    monkeypatch.setattr("composer.spec.source.prover.get_stream_writer", lambda _: None)

    the_tool = get_prover_tool(
        cloud=None,
        llm=fake_llm,
        main_contract="Dummy",
        project_root=str(tmp_path),
        semaphore=None
    )

    def bind_tool(l: Iterable[ProverToolResponse]) -> BaseTool:
        nonlocal response_script
        response_script = list(l)
        return the_tool

    return bind_tool