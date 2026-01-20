import psycopg
from typing import Any
import os
from psycopg.rows import dict_row, RowFactory, DictRow

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langgraph.store.postgres import PostgresStore

from graphcore.tools.memory import PostgresMemoryBackend

from composer.input.types import ModelOptions
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
    checkpointer = PostgresSaver(conn)
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
