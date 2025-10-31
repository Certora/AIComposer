import psycopg
from psycopg.rows import dict_row

from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.store.postgres import PostgresStore

from graphcore.graph import build_workflow, BoundLLM
from graphcore.tools.vfs import vfs_tools, VFSAccessor, VFSToolConfig
from graphcore.tools.memory import PostgresMemoryBackend, memory_tool

from verisafe.workflow.types import Input, PromptParams
from verisafe.core.context import CryptoContext
from verisafe.core.state import CryptoStateGen
from verisafe.input.types import ModelOptions

from verisafe.tools import *
from verisafe.templates.loader import load_jinja_template
from verisafe.workflow.summarization import SummaryGeneration


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

def get_memory() -> PostgresMemoryBackend:
    conn_string = "postgresql://memory_tool_user:memory_tool_password@localhost:5432/memory_tool_db"
    conn = psycopg.connect(conn_string)
    return PostgresMemoryBackend('verisafe', conn)

def get_system_prompt() -> str:
    """Load and render the system prompt from Jinja template"""
    return load_jinja_template("system_prompt.j2")

def get_initial_prompt(prompt: PromptParams) -> str:
    """Load and render the initial prompt from Jinja template"""
    return load_jinja_template("synthesis_prompt.j2", **prompt)


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

def get_cryptostate_builder(
    llm: BaseChatModel,
    prompt_params: PromptParams,
    fs_layer: str | None,
    summarization_threshold : int | None
) -> tuple[StateGraph[CryptoStateGen, CryptoContext, Input, Any], BoundLLM, VFSAccessor[CryptoStateGen]]:
    (vfs_tooling, mat) = vfs_tools(VFSToolConfig(
        fs_layer=fs_layer,
        immutable=False,
        forbidden_write="^rules.spec$",
        put_doc_extra= \
"""
By convention, every Solidity file placed into the virtual filesystem should contain exactly one contract/interface/library definitions.
Further, the name of the contract/interface/library defined in that file should name the name of the solidity source file sans extension.
For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`.

IMPORTANT: You may not use this tool to update the specification, nor should you attempt to
add new specification files.
"""
    ), CryptoStateGen)

    crypto_tools = [certora_prover, propose_spec_change, human_in_the_loop, code_result, cvl_manual_search, *vfs_tooling]

    if "context-management-2025-06-27" in getattr(llm, "betas"):
        memory = memory_tool(get_memory())
        crypto_tools.append(memory)

    conf : SummaryGeneration | None = SummaryGeneration(
        max_messages=summarization_threshold
    ) if summarization_threshold else None

    workflow_builder: tuple[StateGraph[CryptoStateGen, CryptoContext, Input, Any], BoundLLM] = build_workflow(
        state_class=CryptoStateGen,
        input_type=Input,
        tools_list=crypto_tools,
        sys_prompt=get_system_prompt(),
        initial_prompt=get_initial_prompt(prompt_params),
        output_key="generated_code",
        unbound_llm=llm,
        context_schema=CryptoContext,
        summary_config=conf
    )

    return workflow_builder + (mat,)
