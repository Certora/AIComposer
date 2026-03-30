from typing import NotRequired, override, TypedDict, Literal, Annotated
import json

from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel, Discriminator

from graphcore.tools.schemas import WithAsyncImplementation, WithInjectedId, WithInjectedState
from graphcore.tools.results import result_tool_generator
from graphcore.graph import tool_state_update
from graphcore.summary import SummaryConfig

from composer.spec.cvl_generation import (
    static_tools, CVLGenerationExtra, FeedbackToolContext, FEEDBACK_VALIDATION_KEY,
    check_completion, CVL_JUDGE_KEY, run_cvl_generator, GeneratedCVL
)
from composer.spec.context import WorkflowContext, CVLGeneration, SourceCode
from composer.spec.prop import PropertyFormulation
from composer.spec.system_model import ContractComponentInstance
from composer.spec.source.prover import ProverStateExtra, DELETE_SKIP, VALIDATION_KEY as PROVER_VALIDATION_KEY
from langgraph.graph import MessagesState
from composer.spec.gen_types import CVLResource, TypedTemplate
from composer.spec.source.source_env import SourceEnvironment
from langgraph.types import Command
from composer.spec.feedback import property_feedback_judge, FeedbackTemplate

from graphcore.graph import FlowInput

class SourceCVLGenerationExtra(CVLGenerationExtra, ProverStateExtra):
    pass

class SourceCVLGenerationInput(SourceCVLGenerationExtra, FlowInput):
    pass

class SourceCVLGenerationState(SourceCVLGenerationExtra, MessagesState):
    result: NotRequired[str]

class ExpectRuleFailure(WithAsyncImplementation[Command], WithInjectedId):
    """
    Mark a rule name as expected to fail.
    """
    rule_name: str = Field(description="The name of the rule")
    reason: str = Field(description="The reason the rule is expected to fail")

    @override
    async def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Success",
            rule_skips={
                self.rule_name: self.reason
            }
        )

class ExpectRulePassage(WithAsyncImplementation[Command], WithInjectedId):
    """
    Unmark a rule as expected to fail. By default all rules/invariants are expected to pass,
    so this should only be called to revert a prior call to `expect_rule_failure`.
    """
    rule_name : str = Field(description="The name of the rule that was previously marked as expected to fail that is now expected to pass")

    @override
    async def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Success",
            rule_skips={
                self.rule_name: DELETE_SKIP
            }
        )

def result_checker(
    s: SourceCVLGenerationState,
    res: str,
    tid: str
) -> str | None:
    return check_completion(s)

class PropertyGenParams(TypedDict):
    context: ContractComponentInstance | None
    resources: list[CVLResource]
    properties: list[PropertyFormulation]
    contract_name: str

class PropertyGenerationConfig(SummaryConfig[SourceCVLGenerationState]):
    def __init__(self):
        super().__init__(max_messages=75)

    @override
    def get_summarization_prompt(self, state: SourceCVLGenerationState) -> str:
            return """
You are approaching the context limit for your task. After this point, your context will be cleared
and the task restarted from the initial prompt.

To enable you to continue to work effectively after this compaction, summarize the current state of your task. In particular, summarize:
1. Any key findings about CVL you received from the CVL researcher or your own research
2. The current state of your task, including:
   a. What properties have been formalized
   b. What properties you have skipped, and why
   c. What properties have been accepted by the feedback tool.
   d. What rules you have chosen to mark as failing, and why
3. If you have any outstanding, unaddressed feedback from your last iteration with the feedback tool, include that unaddressed feedback in your summary
4. If you have any outstanding, unaddressed tasks from the most recent iteration with the prover, include those unaddressed tasks in your summary
5. Any techniques/attempts that you or the feedback rejected or didn't work
6. Any techniques/attempts that you attempted but were rejected by the prover

In other words, your summary should include all information necessary to prevent the next iteration on this task from repeating work
or repeating mistakes.

If your current task itself began with a summary, include the salient parts of that summary in your new summary.
"""

    @override
    def get_resume_prompt(self, state: SourceCVLGenerationState, summary: str) -> str:
        return f"""
You are resuming this task already in progress. The current version of your spec (if any) is available via the `get_cvl` tool.

A summary of your work up until this point is as follows:

BEGIN SUMMARY:
{summary}

END SUMMARY

**IMPORTANT**: Absolutely *nothing* has changed since the summary was produced and now. You do *NOT* need to reverify
any information about CVL present in your summary unless you discovery something *new* with necessitates revisiting those conclusions.
If you have outstanding feedback to address, you do *NOT* need to re-invoke the feedback tool; proceed immediately with addressing
that feedback.
"""

class AddFile(BaseModel):
    """
    Add a new file to the input of the prover. If the name of the contract within the file does *NOT* match the file stem,
    specify the contract name explicitly, otherwise leave it null.
    """
    type: Literal["add_file"]
    file_path: str = Field(description="The relative path to the file to include in the prover inputs")
    contract_name: str | None = Field(description="The name of the contract within `file_path` to ingest into the prover, if does not match the file stem")

class RemoveFile(BaseModel):
    """
    Remove a file from the prover inputs. If the file is specified in the form `path/to/Contract.sol:Something`
    provide *only* the file path portion, i.e., `path/to/Contract.sol`
    """
    type: Literal["remove_file"]
    path_to_remove: str = Field(description="The path to the file to remove from prover inputs")

class AddLink(BaseModel):
    """
    Add a link from one contract to another via a storage field.

    For example, if contract A has a *top-level* storage field
    `rewardToken` that points to the instance of `B` you should register the link
    (A, rewardToken, B).

    NB that the link field *must* be at the top-level of the contract's storage. Link flags cannot be used
    to link fields in structs.
    """
    type: Literal["add_link"]
    source_contract_name: str = Field(description="The name of the contract that is the source of the link")
    link_field_name: str = Field(description="The storage field holding the link within `source_contract_name`")
    target_contract_name : str = Field(description="The contract held in `link_field_name` of `source_contract_name`")

class RemoveLink(BaseModel):
    """
    Remove a link from one contract to another.
    """
    type: Literal["remove_link"]
    source_contract_name : str = Field(description="The name of the contract whose link should be removed")
    link_field_name : str = Field(description="The storage field holding the link within `source_contract_name` that should be removed")

type ConfigEdit = Annotated[RemoveLink | AddLink | AddFile | RemoveFile, Discriminator("type")]

class ConfigEditTool(WithAsyncImplementation[Command | str], WithInjectedId, WithInjectedState[ProverStateExtra]):
    """
    Call this tool to make a edits to the prover configuration.

    Each individual edit is applied in some sequence; if the edits conflict with one another the result is undefined.
    The configuration change is atomic: if any of the edits fail to apply the configuration will remain unchanged,
    and the issue will be returned. Otherwise, the updated configuration is returned as the result of this call.
    """
    edits: list[ConfigEdit] = Field(
        description="A list of the atomic edits to make to the file."
    )

    def _parse_link(self, l) -> tuple[str, str, str]:
        base = l.split("=", 1)
        assert len(base) == 2, l
        contract_and_field = base[0].split(":", 1)
        assert len(contract_and_field), base[0]
        return (contract_and_field[0], contract_and_field[1], base[1])

    @override
    async def run(self) -> Command | str:
        curr_config = self.state["config"].copy()
        for ed in self.edits:
            match ed:
                case RemoveFile(path_to_remove=to_remove):
                    assert "files" in curr_config
                    new_files = []
                    found = False
                    for (ind, f) in enumerate(curr_config["files"]):
                        if f.startswith(to_remove):
                            new_files.extend(curr_config["files"][ind+1:])
                            found = True
                            break
                        new_files.append(f)
                    if not found:
                        return f"Path {to_remove} doesn't seem to appear in {"\n".join(curr_config["files"])}"
                    curr_config["files"] = new_files
                case AddFile(file_path=to_add, contract_name=explicit_name):
                    assert "files" in curr_config
                    if any([ x.startswith(to_add) for x in curr_config["files"] ]):
                        return f"Path {to_add} already appears in prover inputs"
                    new_files = curr_config["files"].copy()
                    if explicit_name is not None:
                        to_add += f":{explicit_name}"
                    new_files.append(
                        to_add
                    )
                    curr_config["files"] = new_files
                case AddLink(source_contract_name=src, link_field_name=fld, target_contract_name=tgt):
                    if ".sol" in src or ".sol" in tgt:
                        return ".sol extension found in source/dest of AddLink; did you accidentally provide a filename?"
                    if "link" in curr_config:
                        curr_link : list[str] = curr_config["link"]
                        for l in curr_link:
                            (curr_src, curr_fld, curr_dst) = self._parse_link(l)
                            if curr_src == src and curr_fld == fld:
                                return f"Link for field {fld} in contract {src} already exists -> {curr_dst}"
                    new_links = list(curr_config.get("link", []))
                    new_links.append(f"{src}:{fld}={tgt}")
                    curr_config["link"] = new_links
                case RemoveLink(source_contract_name=src, link_field_name=fld):
                    if "link" not in curr_config:
                        return "No links configured, nothing to remove"
                    new_links = []
                    found = False
                    curr_links = curr_config["link"]
                    for (i, l) in enumerate(curr_links):
                        (curr_src, curr_fld, _) = self._parse_link(l)
                        if curr_src == src and curr_fld == fld:
                            new_links.extend(curr_links[i+1:])
                            found = True
                            break
                    if not found:
                        return f"No existing link found that matches {src}:{fld}"
                    curr_config["link"] = new_links

        return tool_state_update(
            self.tool_call_id,
            f"Accepted, new config is:\n```json\n{json.dumps(curr_config, indent=2)}\n```",
            config=curr_config
        )


_PropertyGenTemplate = TypedTemplate[PropertyGenParams]("property_generation_prompt.j2")

async def batch_cvl_generation(
    ctx: WorkflowContext[CVLGeneration],
    init_config: dict,
    props: list[PropertyFormulation],
    component: ContractComponentInstance | None,
    resources: list[CVLResource],
    prover_tool: BaseTool,
    env: SourceEnvironment,
    description: str,
    source: SourceCode
) -> GeneratedCVL:
    result_tool = result_tool_generator(
        "result", (str, "Commentary on your generated spec"),
        "Call to signal your completed cvl generation",
        (SourceCVLGenerationState, result_checker)
    )

    bound_template = _PropertyGenTemplate.bind({
        "resources": resources,
        "context": component,
        "properties": props,
        "contract_name": source.contract_name
    })

    task_graph = env.builder.with_tools(
        env.cvl_authorship_tools
    ).with_tools(
        static_tools()
    ).with_tools(
        [prover_tool, ExpectRulePassage.as_tool("expect_rule_passage"), ExpectRuleFailure.as_tool("expect_rule_failure"), result_tool, ctx.get_memory_tool()]
    ).with_state(
        SourceCVLGenerationState
    ).with_output_key(
        "result"
    ).with_input(
        SourceCVLGenerationInput
    ).with_context(
        FeedbackToolContext
    ).with_sys_prompt_template(
        "property_generation_system_prompt.j2"
    ).inject(
        lambda d: bound_template.render_to(d.with_initial_prompt_template)
    ).with_summary_config(PropertyGenerationConfig()).compile_async()

    feedback_env = property_feedback_judge(
        ctx.child(CVL_JUDGE_KEY), env, FeedbackTemplate.bind({
            "has_source": True,
            "context": component
        }), props
    )

    res_state = await run_cvl_generator(
        ctx = ctx,
        d = task_graph,
        description=description,
        ctxt=feedback_env,
        in_state=SourceCVLGenerationInput(
            curr_spec=None,
            config=init_config,
            input=[],
            required_validations=[FEEDBACK_VALIDATION_KEY, PROVER_VALIDATION_KEY],
            rule_skips={},
            skipped=[],
            validations={}
        )
    )

    d = res_state["curr_spec"]
    assert d is not None and "result" in res_state
    return GeneratedCVL(
        commentary=res_state["result"],
        cvl=d,
        skipped=res_state["skipped"]
    )
