
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from graphcore.graph import Builder
from graphcore.tools.vfs import vfs_tools, VFSAccessor, VFSToolConfig, VFSState

from composer.core.context import AIComposerContext
from composer.core.state import AIComposerState, AIComposerInput

from composer.templates.loader import load_jinja_template
from composer.workflow.summarization import SummaryGeneration



def get_memory_ns(thread_id: str, ns: str) -> str:
    return f"ai-composer-{thread_id}-{ns}"

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
            # Block writes to ANY spec file. Spec mutations must go through
            # propose_spec_change (for committed edits) or write_working_spec
            # + commit_working_spec (for iterative drafts).
            forbidden_write=r"^.+\.spec$|^natspec_output/.+$",
            put_doc_extra= \
    """
    By convention, every Solidity file placed into the virtual filesystem should contain exactly one contract/interface/library definitions.
    Further, the name of the contract/interface/library defined in that file should match the name of the solidity source file sans extension.
    For example, src/MyContract.sol should contain an interface/library/contract called `MyContract`.

    IMPORTANT: You may not use this tool to update, create, or delete any spec file (any path ending in `.spec`).
    All spec mutations must go through the propose_spec_change tool (for committed edits) or the
    write_working_spec / commit_working_spec flow (for iterative drafts).
    """
        ), AIComposerState)

def get_cryptostate_builder(
    llm: BaseChatModel,
    fs_layer: str | None,
    summarization_threshold : int | None,
) -> tuple[Builder[AIComposerState, AIComposerContext, AIComposerInput], VFSAccessor[VFSState]]:
    (vfs_tooling, mat) = get_vfs_tools(fs_layer=fs_layer, immutable=False)
    # import here to avoid loading these for non-composer factory uses

    from composer.tools.prover import certora_prover
    from composer.tools.proposal import propose_spec_change
    from composer.tools.question import human_in_the_loop
    from composer.tools.result import code_result
    from composer.tools.search import cvl_manual_tools
    from composer.tools.working_spec import CommitWorkingSpec, ReadWorkingSpec, WriteWorkingSpec

    crypto_tools: list[BaseTool] = [
        certora_prover,
        propose_spec_change,
        human_in_the_loop,
        code_result,
        *cvl_manual_tools(AIComposerContext),
        *vfs_tooling,
        ReadWorkingSpec.as_tool("read_working_spec"),
        WriteWorkingSpec.as_tool("write_working_spec"),
        CommitWorkingSpec.as_tool("commit_working_spec")
    ]

    builder : Builder[None, None, None] = Builder()


    conf : SummaryGeneration | None = SummaryGeneration(
        max_messages=summarization_threshold
    ) if summarization_threshold else None

    res = builder.with_context(
        AIComposerContext
    ).with_loader(
        load_jinja_template
    ).with_input(
        AIComposerInput
    ).with_tools(
        crypto_tools
    ).with_state(
        AIComposerState
    ).with_llm(
        llm
    ).with_output_key(
        "generated_code"
    )

    if conf is not None:
        res = res.with_summary_config(conf)
    
    return (res, mat)
