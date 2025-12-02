import psycopg
from psycopg.rows import dict_row

from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.store.postgres import PostgresStore

from graphcore.graph import build_workflow, BoundLLM
from graphcore.tools.vfs import vfs_tools, VFSAccessor, VFSToolConfig, VFSState
from graphcore.tools.memory import PostgresMemoryBackend, memory_tool

from composer.workflow.types import Input, PromptParams
from composer.core.context import AIComposerContext
from composer.core.state import AIComposerState
from composer.input.types import ModelOptions

from composer.tools import *
from composer.templates.loader import load_jinja_template
from composer.workflow.summarization import SummaryGeneration


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

def get_memory_ns(thread_id: str, ns: str) -> str:
    return f"ai-composer-{thread_id}-{ns}"

def get_memory(ns: str, init_from: str | None = None) -> PostgresMemoryBackend:
    conn_string = "postgresql://memory_tool_user:memory_tool_password@localhost:5432/memory_tool_db"
    conn = psycopg.connect(conn_string)
    return PostgresMemoryBackend(ns, conn, init_from)

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

def get_vfs_tools(
    fs_layer: str | None,
    immutable: bool
) -> tuple[list[BaseTool], VFSAccessor[VFSState]]:
    if immutable:
        return vfs_tools(VFSToolConfig(
            fs_layer=fs_layer,
            immutable=True
        ), VFSState)
    else:
        return vfs_tools(VFSToolConfig(
            fs_layer=fs_layer,
            immutable=False,
            forbidden_write="^rules.spec$",
            put_doc_extra= \
    """
    By convention, every Solidity file placed into the virtual filesystem should contain exactly one contract/interface/library definitions.
    Further, the name of the contract/interface/library defined in that file should name the name of the solidity source file sans extension.
    For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`"

    IMPORTANT: You may not use this tool to update the specification, nor should you attempt to
    add new specification files.
    """
        ), AIComposerState)

def get_cryptostate_builder(
    llm: BaseChatModel,
    prompt_params: PromptParams,
    fs_layer: str | None,
    summarization_threshold : int | None,
    extra_tools: list[BaseTool] = []
) -> tuple[StateGraph[AIComposerState, AIComposerContext, Input, Any], BoundLLM, VFSAccessor[VFSState]]:
    (vfs_tooling, mat) = get_vfs_tools(fs_layer=fs_layer, immutable=False)

    crypto_tools = [certora_prover, propose_spec_change, human_in_the_loop, code_result, cvl_manual_search, *vfs_tooling]
    crypto_tools.extend(extra_tools)

    conf : SummaryGeneration | None = SummaryGeneration(
        max_messages=summarization_threshold
    ) if summarization_threshold else None

    workflow_builder: tuple[StateGraph[AIComposerState, AIComposerContext, Input, Any], BoundLLM] = build_workflow(
        state_class=AIComposerState,
        input_type=Input,
        tools_list=crypto_tools,
        sys_prompt=get_system_prompt(),
        initial_prompt=get_initial_prompt(prompt_params),
        output_key="generated_code",
        unbound_llm=llm,
        context_schema=AIComposerContext,
        summary_config=conf
    )

    return workflow_builder + (mat,)
