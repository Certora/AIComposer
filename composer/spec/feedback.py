
import sqlite3
from typing import Callable, Awaitable, NotRequired, Protocol, Sequence

from pydantic import BaseModel, Field


from langgraph.graph import MessagesState

from graphcore.graph import FlowInput
from graphcore.tools.memory import SqliteMemoryBackend, memory_tool

from composer.spec.context import (
    WorkflowContext, CVLBuilder, CVLOnlyBuilder,
    CVLJudge
)
from composer.spec.prop import PropertyFormulation
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.cvl.tools import get_cvl
from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.gen_types import GenerationPrompt

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")

class ISkippedProperty(Protocol):
    @property
    def property_index(self) -> int: ...

    @property
    def reason(self) -> str: ...


type FeedbackTool = Callable[[str, Sequence[ISkippedProperty]], Awaitable[PropertyFeedback]]

class PropertyContext(Protocol):
    @property
    def cvl_authorship(self) -> CVLBuilder | CVLOnlyBuilder: ...

    @property
    def prompt(self) -> GenerationPrompt: ...

def property_feedback_judge(
    ctx: WorkflowContext[CVLJudge],
    env: PropertyContext,
    props: list[PropertyFormulation],
    with_memory: bool,
) -> FeedbackTool:

    builder = env.cvl_authorship

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

    if with_memory:
        mem = ctx.get_memory_tool()
    else:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        mem = memory_tool(SqliteMemoryBackend("dummy", db))

    template_kwargs: dict = {
        "properties": props,
        **env.prompt.feedback_prompt.args
    }

    workflow = bind_standard(
        builder, ST, validator=did_rough_draft_read
    ).with_input(
        SpecJudgeInput
    ).with_initial_prompt_template(
        str(env.prompt.feedback_prompt.template),
        **template_kwargs
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_tools([*rough_draft_tools, mem, get_cvl(ST)]).compile_async(
        checkpointer=ctx.checkpointer
    )

    async def the_tool(
        cvl: str,
        skipped: Sequence[ISkippedProperty],
    ) -> PropertyFeedback:
        input_parts: list[str | dict] = env.prompt.feedback_prompt_extras()
        input_parts.append("The proposed CVL file is")
        input_parts.append(cvl)
        if skipped:
            input_parts.append("The following properties were explicitly skipped by the author:")
            for s in skipped:
                input_parts.append(f"  Property {s.property_index}: {s.reason}")
        res = await run_to_completion(
            workflow,
            SpecJudgeInput(input=input_parts, curr_spec=cvl, memory=None, did_read=False),
            thread_id=ctx.uniq_thread_id(),
            description="Property feedback judge",
        )
        assert "result" in res
        return res["result"]

    return the_tool

