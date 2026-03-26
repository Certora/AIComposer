from typing import NotRequired, override, TypedDict

from langchain_core.tools import BaseTool
from pydantic import Field

from graphcore.tools.schemas import WithAsyncImplementation, WithInjectedId
from graphcore.tools.results import result_tool_generator
from graphcore.graph import tool_state_update
from graphcore.summary import SummaryConfig

from composer.spec.cvl_generation import (
    static_tools, CVLGenerationExtra, FeedbackToolContext, FEEDBACK_VALIDATION_KEY,
    check_completion, CVL_JUDGE_KEY, run_cvl_generator, GeneratedCVL
)
from composer.spec.context import WorkflowContext, CVLGeneration, SourceCode, CVLJudge
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
            rule_skip={
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
    pass

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


_PropertyGenTemplate = TypedTemplate[PropertyGenParams]("property_generation_prompt.j2")

async def batch_cvl_generation(
    ctx: WorkflowContext[CVLGeneration],
    init_config: dict,
    props: list[PropertyFormulation],
    component: ContractComponentInstance | None,
    resources: list[CVLResource],
    prover_tool: BaseTool,
    env: SourceEnvironment,
    source: SourceCode,
    description: str
) -> GeneratedCVL:
    result_tool = result_tool_generator(
        "result", (str, "Commentary on your generated spec"),
        "Call to signal your completed cvl generation",
        (SourceCVLGenerationState, result_checker)
    )

    bound_template = _PropertyGenTemplate.bind({})

    task_graph = env.builder.with_tools(
        env.cvl_authorship_tools
    ).with_tools(
        static_tools()
    ).with_tools(
        [prover_tool, ExpectRulePassage.as_tool("expect_rule_passage"), ExpectRuleFailure.as_tool("expect_rule_failure"), result_tool]
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
    ).with_summary_config(PropertyGenerationConfig()).compile()

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
