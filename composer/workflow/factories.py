from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph

from graphcore.graph import build_workflow, BoundLLM
from graphcore.tools.vfs import vfs_tools, VFSAccessor, VFSToolConfig, VFSState

from composer.workflow.types import Input, PromptParams
from composer.core.context import AIComposerContext
from composer.core.state import AIComposerState

from composer.templates.loader import load_jinja_template
from composer.workflow.summarization import SummaryGeneration

from composer.workflow.services import get_checkpointer, get_store, get_memory, create_llm


def get_memory_ns(thread_id: str, ns: str) -> str:
    return f"ai-composer-{thread_id}-{ns}"

def get_system_prompt(platform: str = "evm") -> str:
    """Load and render the system prompt from Jinja template"""
    return load_jinja_template({
        "evm": "system_prompt.j2",
        "svm": "svm_system_prompt.j2",
    }[platform])

def get_initial_prompt(prompt: PromptParams, platform: str = "evm") -> str:
    """Load and render the initial prompt from Jinja template"""
    return load_jinja_template({
        "evm": "synthesis_prompt.j2",
        "svm": "svm_synthesis_prompt.j2",
    }[platform], **prompt)

def get_vfs_tools(
    fs_layer: str | None,
    immutable: bool,
    platform: str = "evm",
) -> tuple[list[BaseTool], VFSAccessor[VFSState]]:
    if immutable:
        return vfs_tools(VFSToolConfig(
            fs_layer=fs_layer,
            immutable=True
        ), VFSState)
    else:
        match platform:
            case "evm":
                kwargs = {
                    "forbidden_write": "^rules\\.spec$",
                    "put_doc_extra": \
    """
    By convention, every Solidity file placed into the virtual filesystem should contain exactly one contract/interface/library definitions.
    Further, the name of the contract/interface/library defined in that file should name the name of the solidity source file sans extension.
    For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`"

    IMPORTANT: You may not use this tool to update the specification, nor should you attempt to
    add new specification files.
    """
                }
            case "svm" | "rust":
                kwargs = {
                    "forbidden_write": "^rules\\.rs$",
                    "put_doc_extra":
    """
    By convention, Rust source files should follow standard Rust module conventions. The main library entry point should be in src/lib.rs.
    You MUST create a valid Cargo.toml for the project to compile and run tests.  The Cargo.toml should include necessary dependencies
    like `cvlr` for CVLR macros.

    IMPORTANT: You may not use this tool to update the specification files. If changes to spec files are necessary, use the
    propose_spec_change tool or consult the user.
    """
                }

        return vfs_tools(VFSToolConfig(
            fs_layer=fs_layer,
            immutable=False,
            **kwargs
        ), AIComposerState)

def get_cryptostate_builder(
    llm: BaseChatModel,
    prompt_params: PromptParams,
    fs_layer: str | None,
    summarization_threshold : int | None,
    extra_tools: list[BaseTool] = [],
    platform: str = "evm",
) -> tuple[StateGraph[AIComposerState, AIComposerContext, Input, Any], BoundLLM, VFSAccessor[VFSState]]:
    (vfs_tooling, mat) = get_vfs_tools(fs_layer=fs_layer, immutable=False, platform=platform)
    # import here to avoid loading these for non-composer factory uses

    from composer.tools.prover import certora_prover, solana_prover
    from composer.tools.proposal import propose_spec_change_tool
    from composer.tools.question import human_in_the_loop
    from composer.tools.result import code_result
    from composer.tools.search import cvl_manual_search

    crypto_tools: list[BaseTool] = ([solana_prover] if platform == "svm" else [certora_prover]
        ) + [propose_spec_change_tool(platform), human_in_the_loop, code_result, cvl_manual_search(AIComposerContext), *vfs_tooling]
    crypto_tools.extend(extra_tools)

    conf : SummaryGeneration | None = SummaryGeneration(
        max_messages=summarization_threshold
    ) if summarization_threshold else None

    workflow_builder: tuple[StateGraph[AIComposerState, AIComposerContext, Input, Any], BoundLLM] = build_workflow(
        state_class=AIComposerState,
        input_type=Input,
        tools_list=crypto_tools,
        sys_prompt=get_system_prompt(platform),
        initial_prompt=get_initial_prompt(prompt_params, platform),
        output_key="generated_code",
        unbound_llm=llm,
        context_schema=AIComposerContext,
        summary_config=conf
    )

    return workflow_builder + (mat,)
