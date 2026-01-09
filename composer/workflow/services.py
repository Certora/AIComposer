import psycopg
from psycopg.rows import dict_row

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langgraph.store.postgres import PostgresStore

from graphcore.tools.memory import PostgresMemoryBackend

from composer.input.types import ModelOptions

def get_checkpointer() -> PostgresSaver:
    conn_string = "postgresql://langgraph_checkpoint_user:langgraph_checkpoint_password@localhost:5432/langgraph_checkpoint_db"
    conn = psycopg.connect(conn_string, autocommit=True, row_factory=dict_row)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer

def get_store() -> PostgresStore:
    conn_string = "postgresql://langgraph_store_user:langgraph_store_password@localhost:5432/langgraph_store_db"
    conn = psycopg.connect(conn_string, autocommit=True, row_factory=dict_row)
    store = PostgresStore(conn)
    store.setup()
    return store

def get_memory(ns: str, init_from: str | None = None) -> PostgresMemoryBackend:
    conn_string = "postgresql://memory_tool_user:memory_tool_password@localhost:5432/memory_tool_db"
    conn = psycopg.connect(conn_string)
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
