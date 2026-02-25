import psycopg
from typing import Any, TypeVar, Callable
import inspect
import os
from psycopg.rows import dict_row, RowFactory, DictRow

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_anthropic import ChatAnthropic
from langgraph.store.postgres import PostgresStore

from graphcore.tools.memory import PostgresMemoryBackend

from composer.input.types import ModelOptions


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
    host = os.environ.get("CERTORA_AI_COMPOSER_PGHOST", "localhost")
    port = os.environ.get("CERTORA_AI_COMPOSER_PGPORT", "5432")
    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    if row_factory is not None:
        return psycopg.connect(conn_string, autocommit=autocommit, row_factory=row_factory)
    return psycopg.connect(conn_string, autocommit=autocommit)


def get_checkpointer() -> PostgresSaver:
    conn = _get_composer_connection(
        user="langgraph_checkpoint_user",
        password="langgraph_checkpoint_password",
        database="langgraph_checkpoint_db",
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

def get_store() -> PostgresStore:
    conn = _get_composer_connection(
        user="langgraph_store_user",
        password="langgraph_store_password",
        database="langgraph_store_db",
        autocommit=True,
        row_factory=dict_row
    )
    store = PostgresStore(conn)
    store.setup()
    return store

def get_indexed_store(embedder: Embeddings) -> PostgresStore:
    conn = _get_composer_connection(
        user="langgraph_store_user",
        password="langgraph_store_password",
        database="langgraph_store_db",
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

def get_memory(ns: str, init_from: str | None = None) -> PostgresMemoryBackend:
    conn = _get_composer_connection(
        user="memory_tool_user",
        password="memory_tool_password",
        database="memory_tool_db"
    )
    return PostgresMemoryBackend(ns, conn, init_from)

def create_llm(args: ModelOptions) -> BaseChatModel:
    """Create and configure the LLM."""
    return ChatAnthropic(
        model_name=args.model,
        max_tokens_to_sample=args.tokens,
        temperature=1,
        timeout=None,
        max_retries=2,
        stop=None,
        thinking={"type": "enabled", "budget_tokens": args.thinking_tokens},
        betas=([
            "files-api-2025-04-14",
            "context-management-2025-06-27"
        ] if args.memory_tool else [
            "files-api-2025-04-14"
        ])
    )
