from typing import Literal
import asyncio

from pydantic import Field, BaseModel

from typing import Literal, Annotated, override, NotRequired
import sys

from pydantic import Field, BaseModel

from langgraph.types import Command
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage

from graphcore.tools.schemas import WithInjectedId, WithAsyncImplementation
from graphcore.tools.vfs import fs_tools
from graphcore.graph import Builder, FlowInput

from composer.spec.trunner import run_to_completion_sync, run_to_completion
from composer.spec.context import WorkspaceContext
from composer.workflow.services import get_checkpointer
from composer.spec.graph_builder import bind_standard
from composer.spec.harness import Configuration


class BaseInvariant(BaseModel):
    """
    A single invariant
    """
    name: str = Field(description="A unique, descriptive name of the invariant")
    description : str = Field(description="A semi-formal, language language description of the invariant to formalize.")

class Invariant(BaseInvariant):
    """
    A single invariant from your analysis with dependencies to other invariants.
    """
    dependencies: list[str] = Field(description="The names of other invariants that are likely to be required to formalize this invariant")

class Invariants(BaseModel):
    """
    The structural invariants you identified in your analysis
    """
    inv: list[Invariant] = Field(description="The invariants you identified")

type InvFeedbackSort = Literal[
    "GOOD",
    "NOT_STRUCTURAL",
    "NOT_INDUCTIVE",
    "UNLIKELY_TO_HOLD",
    "NOT_FORMAL"
]

class InvariantFeedback(BaseModel):
    """
    Your feedback on the given invariant
    """
    sort: InvFeedbackSort = Field(description="Your classification on the invariant")
    explanation: str = Field(description="An explanation of your finding, including any suggestions for improvement.")

def _get_invariant_formulation(
    inv_ctx: WorkspaceContext,
    builder: Builder[None, None, None]
) -> Invariants:
    if (cached := inv_ctx.cache_get()) is not None:
        return Invariants.model_validate(cached)

    def merge_invariant_feedback(left: dict[str, tuple[str, InvFeedbackSort]], right: dict[str, tuple[str, InvFeedbackSort]]) -> dict:
        to_ret = left.copy()
        for (k,v) in right.items():
            to_ret[k] = v

        return to_ret

    class InvInput(FlowInput):
        invariant_data: dict

    class ST(MessagesState):
        result: NotRequired[Invariants]
        invariant_data: Annotated[dict[str, tuple[str, InvFeedbackSort]], merge_invariant_feedback]

    fs = fs_tools(inv_ctx.project_root, forbidden_read=inv_ctx.fs_filter)

    memory = inv_ctx.get_memory_tool()

    judge_ctx = inv_ctx.child(
        "judge"
    )

    def validate_invariants(
        _: ST,
        i: Invariants
    ) -> str | None:
        all_invariant_names = set()
        for inv in i.inv:
            if inv.name in all_invariant_names:
                return f"Multiple definitions for {inv.name}"
            all_invariant_names.add(inv.name)
        
        for inv in i.inv:
            for d in inv.dependencies:
                if d not in all_invariant_names:
                    return f"Invariant {inv.name} references {d}, but no invariant with that name exists."

    class FeedbackST(MessagesState):
        result: NotRequired[InvariantFeedback]

    feedback_graph = bind_standard(
        builder,
        FeedbackST
    ).with_sys_prompt(
        "You are a methodical formal verification expert working at Certora, Inc."
    ).with_initial_prompt_template(
        "invariant_judge_prompt.j2",
        contract_spec=inv_ctx
    ).with_tools(
        [*fs, judge_ctx.get_memory_tool()]
    ).with_input(
        FlowInput
    ).build_async()[0].compile(
        checkpointer=get_checkpointer()
    )

    sem = asyncio.Semaphore(3)

    class InvariantFeedbackTool(WithInjectedId, WithAsyncImplementation[Command]):
        """
        Receive feedback on one of your invariants
        """
        inv: BaseInvariant = Field(description="The invariant to receive feedback on")
        
        @override
        async def run(self) -> Command:
            async with sem:
                res = await run_to_completion(
                    feedback_graph,
                    FlowInput(input=[
                        f"The invariant is called: {self.inv.name}\nStatement: {self.inv.description}"
                    ]),
                    "invariant-judge"
                )
                assert "result" in res
                feedback = res["result"]
                update = {
                    "messages": [ToolMessage(
                        tool_call_id=self.tool_call_id,
                        content=f"Judgment: {feedback.sort}\nExplanation: {feedback.explanation}"
                    )],
                    "invariant_data": {
                        self.inv.name: (self.inv.description, feedback.sort)
                    }
                }
                return Command(update=update)

    d = bind_standard(
        builder,
        ST,
        doc="The structural/state invariants you identified",
        validator=validate_invariants
    ).with_sys_prompt(
        "You are a methodical formal verification expert working at Certora, Inc."
    ).with_initial_prompt_template(
        "structural_invariant_prompt.j2",
        contract_spec=inv_ctx
    ).with_tools(
        [*fs, memory, InvariantFeedbackTool.as_tool("invariant_feedback")]
    ).with_input(InvInput).build_async()[0].compile(checkpointer=get_checkpointer())

    s = run_to_completion_sync(
        graph=d,
        input=InvInput(input=[], invariant_data={}),
        thread_id=inv_ctx.thread_id,
    )
    assert "result" in s
    to_ret = s["result"]
    inv_ctx.cache_put(to_ret.model_dump())
    return to_ret


def structural_invariants_flow(
    ctx: WorkspaceContext,
    conf: Configuration,
    builder: Builder[None, None, None]
) -> str:
    s = _get_invariant_formulation(
        ctx.child("structural-inv"),
        builder
    )
    print(s)

    

    sys.exit(1)
