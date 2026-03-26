from composer.spec.cvl_generation import (static_tools, run_cvl_generator, CVLGenerationInput, CVLGenerationState,
    FeedbackToolContext
)

from dataclasses import dataclass
from typing import Callable, override, Protocol


from langchain_core.tools import BaseTool


from graphcore.summary import SummaryConfig

from composer.spec.context import (
    WorkflowContext, CVLGeneration, SystemDoc
)
from composer.spec.prop import PropertyFormulation
from composer.spec.feedback import property_feedback_judge, Properties, FeedbackTemplate, FeedbackEnv
from composer.spec.gen_types import TypedTemplate
from composer.spec.system_model import ContractComponentInstance
from composer.spec.cvl_generation import CVL_JUDGE_KEY, FeedbackToolContext, static_tools, SkippedProperty

class GenerationEnv(FeedbackEnv, Protocol):
    @property
    def cvl_authorship_tools(self) -> tuple[BaseTool, ...]:
        ...

class SourceGenerationParams(Properties):
    context: ContractComponentInstance

NoSourceGen = TypedTemplate[SourceGenerationParams]("nosource_property_generation_prompt.j2")

class _CVLConfig(SummaryConfig[CVLGenerationState]):
    def __init__(self, reader: Callable[[], str], contract_name: str):
        super().__init__(max_messages=50, enabled=True)
        self.reader = reader
        self.contract_name = contract_name

    @override
    def get_summarization_prompt(self, state: CVLGenerationState) -> str:
        return """
You are approaching the context limit for your task. After this point, your context will be cleared
and the task restarted from the initial prompt.

To enable you to continue to work effectively after this compaction, summarize the current state of your task. In particular, summarize:
1. Any key findings about CVL you received from the CVL researcher or your own research
2. The current state of your task, including:
   a. What properties have been formalized
   b. What properties you have skipped, and why
   c. What properties have been accepted by the feedback tool.
3. If you have any outstanding, unaddressed feedback from your last iteration with the feedback tool, include that unaddressed feedback in your summary
4. Any techniques/attempts that you or the feedback rejected or didn't work

In other words, your summary should include all information necessary to prevent the next iteration on this task from repeating work
or repeating mistakes.

If your current task itself began with a summary, include the salient parts of that summary in your new summary.
"""

    @override
    def get_resume_prompt(self, state: CVLGenerationState, summary: str) -> str:
        return f"""
You are resuming this task already in progress. The current version of your spec (if any) is available via the `get_cvl` tool.

The current content of the type checking stub for {self.contract_name} is as follows:
```solidity
{self.reader()}
```

A summary of your work up until this point is as follows:

BEGIN SUMMARY:
{summary}

END SUMMARY

**IMPORTANT**: Absolutely *nothing* has changed since the summary was produced and now. You do *NOT* need to reverify
any information about CVL present in your summary unless you discovery something *new* with necessitates revisiting those conclusions.
If you have outstanding feedback to address, you do *NOT* need to re-invoke the feedback tool; proceed immediately with addressing
that feedback.
"""
    

@dataclass
class GenerationSuccess:
    commentary: str
    skipped: list[SkippedProperty]

@dataclass
class GaveUp:
    reason: str

async def generate_cvl_batch(
    ctx: WorkflowContext[CVLGeneration],
    env: GenerationEnv,
    system_doc: SystemDoc,

    props: list[PropertyFormulation],
    component: ContractComponentInstance,
    contract_name: str,

    injected_tools: list[BaseTool],
    stub_reader: Callable[[], str]
) -> GenerationSuccess | GaveUp:
    feedback_ctxt = property_feedback_judge(
        ctx=ctx.child(CVL_JUDGE_KEY), env=env, prompt=FeedbackTemplate.bind({
            "context": component,
            "has_source": False
        }).depends(Properties), props=props, extra_inputs=lambda: [
            f"The current typechecking stub for the {contract_name} stub is",
            stub_reader(),
            "For reference, the system document for the application is",
            system_doc.content
        ]
    )

    g = env.builder.with_tools(
        env.cvl_authorship_tools
    ).with_tools(
        injected_tools
    ).with_tools(
        static_tools()
    ).with_output_key(
        "result"
    ).with_input(
        CVLGenerationInput
    ).with_state(
        CVLGenerationState
    ).with_context(
        FeedbackToolContext
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).inject(
        lambda b: NoSourceGen.bind({
            "context": component,
            "properties": props
        }).render_to(b.with_initial_prompt_template)
    ).with_summary_config(_CVLConfig(stub_reader, contract_name)).compile_async()

    res = await run_cvl_generator(
        ctx, g, CVLGenerationInput(
            curr_spec=None,
            input=[
                f"The current stub implementation of the {contract_name} is",
                stub_reader()
            ],
            required_validations=["feedback"],
            skipped=[],
            validations={}
        ),
        ctxt=feedback_ctxt,
        description = f"{contract_name} {component.component.name} ({len(props)} properties)"
    )
    assert "result" in res
    # I hate this, but, well, I give up
    if res["result"].startswith("GAVE_UP:"):
        return GaveUp(reason=res["result"])
    else:
        return GenerationSuccess(
            commentary=res["result"],
            skipped=res["skipped"]
        )
