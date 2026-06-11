
from typing import Callable, NotRequired, Protocol, Sequence, Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool

from graphcore.graph import FlowInput

from composer.spec.context import (
    WorkflowContext, CVLJudge
)
from composer.spec.prop import PropertyFormulation
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.cvl.tools import get_cvl
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.gen_types import TemplateInstantiation, InjectedTemplate, TypedTemplate
from composer.spec.cvl_generation import FeedbackToolContext, Rebuttal, SkippedProperty
from composer.spec.tool_env import BasicAgentTools
from composer.spec.system_model import ContractComponentInstance
from composer.spec.util import uniq_thread_id

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")

class FeedbackEnv(BasicAgentTools, Protocol):
    @property
    def feedback_tools(self) -> tuple[BaseTool, ...]:
        ...

class Properties(TypedDict):
    properties: list[PropertyFormulation]

class FeedbackInherentParams(TypedDict):
    context: ContractComponentInstance | None
    has_source: bool

FeedbackTemplate = TypedTemplate[FeedbackInherentParams]("property_judge_prompt.j2")

# Default judge system prompt for the no-source (natspec) flow: same template the
# source-mode judge uses, with the source-tool sections compiled out. Source-mode
# callers override with has_source=True (see composer/spec/source/author.py).
FeedbackSystemTemplate = TypedTemplate[dict[str, Any]]("property_judge_system_prompt.j2").bind({"has_source": False})

def property_feedback_judge(
    ctx: WorkflowContext[CVLJudge],
    env: FeedbackEnv,
    prompt: InjectedTemplate[Properties] | TemplateInstantiation,
    props: list[PropertyFormulation],
    *,
    extra_inputs: list[str | dict] | Callable[[], list[str | dict]] | None = None,
    system_prompt: TemplateInstantiation = FeedbackSystemTemplate
) -> FeedbackToolContext:

    builder = env.builder.with_tools(
        env.feedback_tools
    )

    class JudgeExtra(RoughDraftState):
        curr_spec: str

    class ST(MessagesState, JudgeExtra):
        result: NotRequired[PropertyFeedback]

    class SpecJudgeInput(FlowInput, JudgeExtra):
        pass

    rough_draft_tools = get_rough_draft_tools(ST)

    def did_rough_draft_read(s: ST, _) -> str | None:
        if not s["did_read"]:
            return "Completion REJECTED: never read rough draft for review"
        return None

    mem = ctx.get_memory_tool()

    final_prompt = prompt if isinstance(prompt, TemplateInstantiation) else prompt.inject({"properties": props})

    workflow = bind_standard(
        builder, ST, validator=did_rough_draft_read
    ).with_input(
        SpecJudgeInput
    ).inject(
        lambda b: final_prompt.render_to(b.with_initial_prompt_template)
    ).inject(
        lambda g: system_prompt.render_to(g.with_sys_prompt_template)
    ).with_tools([*rough_draft_tools, mem, get_cvl(ST), ]).compile_async()

    async def the_tool(
        cvl: str,
        skipped: Sequence[SkippedProperty],
        rebuttals: Sequence[Rebuttal],
        within_tool: str,
    ) -> PropertyFeedback:
        input_parts: list[str | dict] = []
        if extra_inputs:
            if isinstance(extra_inputs, list):
                input_parts.extend(extra_inputs)
            else:
                input_parts.extend(extra_inputs())

        input_parts.append("The proposed CVL file is")
        input_parts.append(cvl)
        if skipped:
            input_parts.append("The following properties were explicitly skipped by the author:")
            for s in skipped:
                input_parts.append(f"  Property {s.property_title}: {s.reason}")
        if rebuttals:
            input_parts.append(
                "The author has filed the following rebuttals against feedback from "
                "prior rounds. Evaluate each per the Step 1 rebuttal rule (and the "
                "Criteria 7 exception for skip-related rebuttals). Empirical evidence "
                "types (`typecheck_failure`, `counterexample`, `manual_citation`) "
                "carry near-binding weight; `reasoned` rebuttals are a conversation, "
                "not a veto."
            )
            for i, r in enumerate(rebuttals, 1):
                input_parts.append(
                    f"  Rebuttal {i} [{r.evidence_type}]\n"
                    f"    Addressing: {r.prior_feedback_reference}\n"
                    f"    Evidence: {r.evidence}"
                )
        res = await run_to_completion(
            workflow,
            SpecJudgeInput(input=input_parts, curr_spec=cvl, memory=None, did_read=False),
            thread_id=uniq_thread_id("feedback"),
            recursion_limit=ctx.recursion_limit,
            description="Property feedback judge",
            within_tool=within_tool,
        )
        assert "result" in res
        return res["result"]

    return FeedbackToolContext(feedback_thunk=the_tool, titles=[p.title for p in props])

