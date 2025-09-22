from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import Tuple, Any
from langchain_core.language_models.chat_models import BaseChatModel
from verisafe.workflow.types import Input
from verisafe.core.context import CryptoContext
from verisafe.core.state import CryptoStateGen
from graphcore.graph import build_workflow, BoundLLM, SummarizationConfig
from langgraph.graph import StateGraph
from verisafe.input.types import ModelOptions
from langchain_anthropic import ChatAnthropic

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

def get_cryptostate_builder(llm: BaseChatModel, summarization_threshold: int = None) -> Tuple[StateGraph[CryptoStateGen, CryptoContext, Input, Any], BoundLLM]:
    crypto_tools = [certora_prover, propose_spec_change, human_in_the_loop, code_result, cvl_manual_search, put_file]
    summarization_config=SummarizationConfig(max_messages=summarization_threshold
        ) if summarization_threshold is not None else None

    workflow_builder: Tuple[StateGraph[CryptoStateGen, CryptoContext, Input, Any], BoundLLM] = build_workflow(
        state_class=CryptoStateGen,
        input_type=Input,
        tools_list=crypto_tools,
        sys_prompt=get_system_prompt(),
        initial_prompt=get_initial_prompt(),
        output_key="generated_code",
        unbound_llm=llm,
        context_schema=CryptoContext,
        summarization_config=summarization_config
    )

    return workflow_builder
