
import sqlite3
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph

from graphcore.graph import build_workflow, BoundLLM
from graphcore.tools.vfs import vfs_tools, VFSAccessor, VFSToolConfig
from graphcore.summary import SummaryConfig

from verisafe.workflow.types import Input
from verisafe.core.context import CryptoContext
from verisafe.core.state import CryptoStateGen
from verisafe.input.types import ModelOptions

from verisafe.tools import *
from verisafe.templates.loader import load_jinja_template

def get_checkpointer() -> SqliteSaver:
    state_db = sqlite3.connect("cryptosafe.db", check_same_thread=False)
    checkpointer = SqliteSaver(state_db)
    return checkpointer

def get_system_prompt() -> str:
    """Load and render the system prompt from Jinja template"""
    return load_jinja_template("system_prompt.j2")

def get_initial_prompt() -> str:
    """Load and render the initial prompt from Jinja template"""
    return load_jinja_template("synthesis_prompt.j2")


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
        betas=["files-api-2025-04-14"],
    )

class SummaryGeneration(SummaryConfig[CryptoStateGen]):
    def get_resume_prompt(self, state: CryptoStateGen, summary: str) -> str:
        res = super().get_resume_prompt(state, summary)

        res += "\n You may use the VFS tools to query the current state of your implementation."
        return res


def get_cryptostate_builder(llm: BaseChatModel, summarization_threshold: int | None, fs_layer: str | None) -> tuple[StateGraph[CryptoStateGen, CryptoContext, Input, Any], BoundLLM, VFSAccessor[CryptoStateGen]]:

    system_prompt = get_system_prompt()
    initial_prompt = get_initial_prompt()
    
    conf : SummaryGeneration | None = None
    if summarization_threshold is not None:
        conf = SummaryGeneration(
            max_messages=summarization_threshold
        )

    (vfs_tooling, mat) = vfs_tools(VFSToolConfig(
        fs_layer=fs_layer,
        immutable=False,
        forbidden_write="^rules.spec$",
        put_doc_extra= \
"""
By convention, every Solidity placed into the virtual filesystem should contain exactly one contract/interface/library defitions.
Further, the name of the contract/interface/library defined in that file should name the name of the solidity source file sans extension.
For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`"

IMPORTANT: You may not use this tool to update the specification, nor should you attempt to
add new specification files.
"""
    ), CryptoStateGen)

    crypto_tools = [certora_prover, propose_spec_change, human_in_the_loop, code_result, cvl_manual_search, *vfs_tooling]


    workflow_builder: tuple[StateGraph[CryptoStateGen, CryptoContext, Input, Any], BoundLLM] = build_workflow(
        state_class=CryptoStateGen,
        input_type=Input,
        tools_list=crypto_tools,
        sys_prompt=system_prompt,
        initial_prompt=initial_prompt,
        output_key="generated_code",
        unbound_llm=llm,
        context_schema=CryptoContext,
        summary_config=conf
    )

    return workflow_builder + (mat,)
