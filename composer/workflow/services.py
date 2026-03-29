import psycopg
from typing import Any, Callable, TypedDict, Literal, overload, AsyncContextManager
from typing_extensions import TypeVar
import inspect
import os
from dataclasses import dataclass
from psycopg.rows import dict_row, RowFactory, DictRow, Row
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_anthropic import ChatAnthropic
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from psycopg.rows import AsyncRowFactory
from psycopg_pool.pool_async import AsyncConnectionPool as PGAsyncPool
from psycopg.connection_async import AsyncConnection
from graphcore.tools.memory import PostgresMemoryBackend, AsyncPostgresBackend

from composer.input.types import ModelOptions, ModelOptionsBase

T = TypeVar("T")

def _adapt_async(obj: T, pairs: list[tuple[str, str]]) -> T:
    """
    Patch async methods to forward to their sync counterparts.
    
    Args:
        obj: Object to patch
        pairs: List of (async_name, sync_name) tuples
        
    Raises:
        AttributeError: If method names don't exist on obj
        TypeError: If async method is not a coroutine or sync method is a coroutine
        ValueError: If method signatures don't match
    """
    for async_name, sync_name in pairs:
        # Step 1: Fetch attributes
        try:
            async_method = getattr(obj, async_name)
        except AttributeError:
            raise AttributeError(
                f"Object {obj} does not have async method '{async_name}'"
            )
        
        try:
            sync_method = getattr(obj, sync_name)
        except AttributeError:
            raise AttributeError(
                f"Object {obj} does not have sync method '{sync_name}'"
            )

        # Step 2: Verify that async_method is a coroutine function
        if not inspect.iscoroutinefunction(async_method) and not inspect.isasyncgenfunction(async_method):
            raise TypeError(
                f"Method '{async_name}' is not a coroutine function"
            )
        
        # Verify that sync_method is NOT a coroutine function
        if inspect.iscoroutinefunction(sync_method):
            raise TypeError(
                f"Method '{sync_name}' is a coroutine function but should be sync"
            )
        
        # Get signatures
        async_sig = inspect.signature(async_method)
        sync_sig = inspect.signature(sync_method)
        
        # Compare parameters (names and annotations)
        async_params = list(async_sig.parameters.values())
        sync_params = list(sync_sig.parameters.values())
        
        if len(async_params) != len(sync_params):
            raise ValueError(
                f"Parameter count mismatch: {async_name} has {len(async_params)} "
                f"parameters, {sync_name} has {len(sync_params)}"
            )
        
        for async_param, sync_param in zip(async_params, sync_params):
            if async_param.name != sync_param.name:
                raise ValueError(
                    f"Parameter name mismatch: {async_name} has '{async_param.name}', "
                    f"{sync_name} has '{sync_param.name}'"
                )
            
            if async_param.annotation != sync_param.annotation:
                raise ValueError(
                    f"Parameter annotation mismatch for '{async_param.name}': "
                    f"{async_name} has {async_param.annotation}, "
                    f"{sync_name} has {sync_param.annotation}"
                )
            
            if async_param.default != sync_param.default:
                raise ValueError(
                    f"Parameter default mismatch for '{async_param.name}': "
                    f"{async_name} has {async_param.default}, "
                    f"{sync_name} has {sync_param.default}"
                )
        
        # Step 3: Create wrapper that forwards to sync implementation
        def make_wrapper(sync_fn: Callable) -> Callable:
            async def async_wrapper(*args, **kwargs):
                # Call the sync function
                return sync_fn(*args, **kwargs)
            
            # Preserve the original signature
            setattr(async_wrapper, "__signature__", inspect.signature(sync_fn))
            async_wrapper.__name__ = sync_fn.__name__
            async_wrapper.__doc__ = sync_fn.__doc__
            
            return async_wrapper
        
        # Patch the object
        new_async_method = make_wrapper(sync_method)
        setattr(obj, async_name, new_async_method)
    return obj

def _get_composer_connection_string(
     *,
    user: str,
    password: str,
    database: str,
) -> str:
    host = os.environ.get("CERTORA_AI_COMPOSER_PGHOST", "localhost")
    port = os.environ.get("CERTORA_AI_COMPOSER_PGPORT", "5432")
    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return conn_string

def _get_composer_connection(
    *,
    user: str,
    password: str,
    database: str,
    autocommit: bool = False,
    row_factory: RowFactory[DictRow] | None = None
) -> psycopg.Connection[Any]:
    """Create a PostgreSQL connection for composer services.

    Args:
        user: Database user name
        password: Database password
        database: Database name
        autocommit: Whether to enable autocommit mode (default: False)
        row_factory: Row factory for result formatting (default: None)

    Returns:
        psycopg.Connection: Configured database connection
    """
    conn_string = _get_composer_connection_string(
        user=user,
        password=password,
        database=database
    )
    if row_factory is not None:
        return psycopg.connect(conn_string, autocommit=autocommit, row_factory=row_factory)
    return psycopg.connect(conn_string, autocommit=autocommit)

async def _get_async_composer_pool(
    *,
    user: str,
    password: str,
    database: str,
    autocommit : bool = False,
    row_factory : AsyncRowFactory[Row] | None = None
) -> PGAsyncPool[AsyncConnection[Row]]:
    conn_string = _get_composer_connection_string(
        user=user,
        database=database,
        password=password
    )

    kwargs : dict[str, Any] = {
        "autocommit": autocommit
    }
    if row_factory is not None:
        kwargs["row_factory"] = row_factory
    

    pool = PGAsyncPool(
        conn_string,
        connection_class=AsyncConnection[Row],
        kwargs=kwargs
    )
    await pool.open()
    return pool


from typing import AsyncIterable, AsyncIterator, Iterable, Iterator, NotRequired, TYPE_CHECKING, Generic

class _ConnInfo(TypedDict):
    user: str
    password: str
    database: str


type LG_DBClass = Literal["store", "checkpoint"]
type DBClass = LG_DBClass | Literal["memory"]

_DATABASE_CONFIGS : dict[DBClass, _ConnInfo] = {
    "checkpoint": {
        "user": "langgraph_checkpoint_user",
        "password": "langgraph_checkpoint_password",
        "database": "langgraph_checkpoint_db"
    },
    "memory": {
        "user": "memory_tool_user",
        "password": "memory_tool_password",
        "database": "memory_tool_db"
    },
    "store": {
        "user": "langgraph_store_user",
        "password": "langgraph_store_password",
        "database": "langgraph_store_db"
    }
}

@asynccontextmanager
async def _async_lg_pool(
    l: LG_DBClass
) -> AsyncIterator[PGAsyncPool[AsyncConnection[DictRow]]]:
    async with _async_pool_context_inner(
        l, True, dict_row
    ) as p:
        yield p

@asynccontextmanager
async def _async_memory_pool(
) -> AsyncIterator[PGAsyncPool[AsyncConnection]]:
    async with _async_pool_context_inner(
        "memory", False
    ) as p:
        yield p

@asynccontextmanager
async def _async_pool_context_inner(
    l: DBClass,
    autocommit: bool,
    row_factory: AsyncRowFactory[Row] | None = None
) -> AsyncIterator[PGAsyncPool[AsyncConnection[Row]]]:
    config = _DATABASE_CONFIGS[l]
    conn_string = _get_composer_connection_string(
        **config
    )
    kwargs : dict[str, Any] = {
        "autocommit": autocommit
    }
    if row_factory is not None:
        kwargs["row_factory"] = row_factory
    pool = PGAsyncPool(
        conn_string,
        connection_class=AsyncConnection[Row],
        kwargs=kwargs
    )
    async with pool:
        yield pool

@asynccontextmanager
async def checkpointer_context() -> AsyncIterator[AsyncPostgresSaver]:
    async with _async_lg_pool("checkpoint") as p:
        checkpointer = AsyncPostgresSaver(p)
        await checkpointer.setup()
        yield checkpointer

@asynccontextmanager
async def store_context() -> AsyncIterator[AsyncPostgresStore]:
    async with _async_lg_pool("store") as p:
        store = AsyncPostgresStore(p)
        await store.setup()
        async with store:
            yield store

from typing_extensions import deprecated

@deprecated("Use async code")
def get_checkpointer() -> PostgresSaver:
    conn = _get_composer_connection(
        **_DATABASE_CONFIGS["checkpoint"],
        autocommit=True,
        row_factory=dict_row
    )
    checkpointer = _adapt_async(
        PostgresSaver(conn),
        [("aget", "get"),
         ("aput", "put"),
         ("aget_tuple", "get_tuple"),
         ("alist", "list"),
         ("adelete_thread", "delete_thread"),
         ("aput_writes", "put_writes")
         ]
    )
    checkpointer.setup()
    return checkpointer

async def get_async_checkpointer() -> AsyncPostgresSaver:
    conn =  await _get_async_composer_pool(
        **_DATABASE_CONFIGS["checkpoint"],
        autocommit=True,
        row_factory=dict_row
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    return checkpointer

@deprecated("Use async code")
def get_store() -> PostgresStore:
    conn = _get_composer_connection(
        **_DATABASE_CONFIGS["store"],
        autocommit=True,
        row_factory=dict_row
    )
    store = PostgresStore(conn)
    store.setup()
    return store

async def get_async_store() -> AsyncPostgresStore:
    conn = await _get_async_composer_pool(
        **_DATABASE_CONFIGS["store"],
        autocommit=True,
        row_factory=dict_row
    )
    store = AsyncPostgresStore(
        conn
    )
    await store.setup()
    return store

@deprecated("Use async code")
def get_indexed_store(embedder: Embeddings) -> PostgresStore:
    conn = _get_composer_connection(
        **_DATABASE_CONFIGS["store"],
        autocommit=True,
        row_factory=dict_row
    )
    store = PostgresStore(
        conn,
        index={
            "embed": embedder,
            "dims": 768,
            "fields": None
        }
    )
    store.setup()
    return store

@asynccontextmanager
async def indexed_store_context(embedder: Embeddings, dims: int = 768) -> AsyncIterator[AsyncPostgresStore]:
    async with _async_lg_pool("store") as p:
        store = AsyncPostgresStore(
            p,
            index={
                "embed": embedder,
                "dims": dims,
                "fields": None
            }
        )
        await store.setup()
        yield store


async def get_async_indexed_store(embedder: Embeddings) -> AsyncPostgresStore:
    conn = await _get_async_composer_pool(
        **_DATABASE_CONFIGS["store"],
        autocommit=True,
        row_factory=dict_row
    )
    store = AsyncPostgresStore(
        conn,
        index={
            "embed": embedder,
            "dims": 768,
            "fields": None
        }
    )
    await store.setup()
    return store


def get_memory(ns: str, init_from: str | None = None) -> PostgresMemoryBackend:
    conn = _get_composer_connection(
        **_DATABASE_CONFIGS["memory"]
    )
    return PostgresMemoryBackend(ns, conn, init_from)

async def get_async_memory(ns : str, init_from : str | None = None) -> AsyncPostgresBackend:
    conn = await _get_async_composer_pool(
        **_DATABASE_CONFIGS["memory"]
    )
    to_ret = AsyncPostgresBackend(ns, conn)
    if init_from is not None:
        await to_ret.init_from(init_from)
    return to_ret

type MemoryBackendGenerator = Callable[[str], AsyncPostgresBackend]

@asynccontextmanager
async def memory_backend_context() -> AsyncIterator[MemoryBackendGenerator]:
    async with _async_memory_pool() as p:
        yield (lambda ns: AsyncPostgresBackend(ns, p))

_ADAPTIVE_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6"}


def create_llm_base(args: ModelOptionsBase) -> BaseChatModel:
    """Create LLM; thinking disabled when args.thinking_tokens is None."""
    thinking: dict[str, Any] | None
    if args.thinking_tokens is None:
        thinking = None
    elif args.model in _ADAPTIVE_MODELS:
        thinking = {"type": "adaptive"}
    else:
        thinking = {"type": "enabled", "budget_tokens": args.thinking_tokens}

    return ChatAnthropic(
        model_name=args.model,
        max_tokens_to_sample=args.tokens,
        temperature=1,
        timeout=None,
        max_retries=2,
        stop=None,
        thinking=thinking,
        betas=([
            "files-api-2025-04-14",
            "context-management-2025-06-27"
        ] if args.memory_tool else [
            "files-api-2025-04-14"
        ])
    )


def create_llm(args: ModelOptions) -> BaseChatModel:
    """Create and configure the LLM. Backwards-compatible; thinking always enabled."""
    return create_llm_base(args)


@dataclass
class StandardConnections:
    checkpointer: AsyncPostgresSaver
    store: AsyncPostgresStore
    memory: Callable[[str], AsyncPostgresBackend]

@dataclass
class IndexedConnections(StandardConnections):
    indexed_store: AsyncPostgresStore


@overload
def standard_connections() -> AsyncContextManager[StandardConnections]:
    ...

@overload
def standard_connections(*, embedder : Embeddings) -> AsyncContextManager[IndexedConnections]:
    ...

@asynccontextmanager
async def standard_connections(
    *,
    embedder: Embeddings | None = None
) -> AsyncIterator[StandardConnections | IndexedConnections]:
    async with (
        checkpointer_context() as check,
        memory_backend_context() as mem,
        store_context() as store
    ):
        if embedder is not None:
            async with indexed_store_context(embedder) as ind:
                yield IndexedConnections(
                    checkpointer=check,
                    indexed_store=ind,
                    store=store,
                    memory=mem
                )
                return
        yield StandardConnections(
            checkpointer=check,
            store=store,
            memory=mem
        )