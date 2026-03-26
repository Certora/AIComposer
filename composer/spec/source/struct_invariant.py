"""
Structural invariant formulation.

Runs an LLM agent that identifies structural invariants for a contract,
with a feedback sub-agent that validates each candidate invariant.
The resulting invariants are converted to ``PropertyFormulation`` instances
and fed into ``generate_batch_cvl`` by the pipeline.
"""

import asyncio
from typing import Literal, Annotated, NotRequired, override

from typing_extensions import TypedDict
from pydantic import Field, BaseModel

from langgraph.types import Command
from langgraph.graph import MessagesState
from langchain_core.messages import ToolMessage

from graphcore.tools.schemas import WithInjectedId, WithAsyncImplementation
from graphcore.graph import FlowInput

from composer.tools.thinking import RoughDraftState, get_rough_draft_tools
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.context import WorkflowContext, SourceCode, CacheKey, InvJudge
from composer.spec.source.source_env import SourceEnvironment
from composer.spec.system_model import HarnessedApplication
from composer.spec.gen_types import TypedTemplate


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaseInvariant(BaseModel):
    """A single invariant."""
    name: str = Field(description="A unique, descriptive name of the invariant. Must not contain spaces (use snake casing if necessary)")
    description: str = Field(description="A semi-formal, natural language description of the invariant to formalize.")


class Invariants(BaseModel):
    """The structural invariants identified in the analysis."""
    inv: list[BaseInvariant] = Field(description="The invariants you identified")


type InvFeedbackSort = Literal[
    "GOOD",
    "NOT_STRUCTURAL",
    "NOT_INDUCTIVE",
    "UNLIKELY_TO_HOLD",
    "NOT_FORMAL",
]


class InvariantFeedback(BaseModel):
    """Feedback on a given invariant."""
    sort: InvFeedbackSort = Field(description="Your classification on the invariant")
    explanation: str = Field(description="An explanation of your finding, including any suggestions for improvement.")


STRUCTURAL_INV_KEY = CacheKey[None, Invariants]("structural-inv")
INV_JUDGE_KEY = CacheKey[Invariants, InvJudge]("judge")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _merge_invariant_feedback(
    left: dict[str, tuple[str, InvFeedbackSort]],
    right: dict[str, tuple[str, InvFeedbackSort]],
) -> dict[str, tuple[str, InvFeedbackSort]]:
    to_ret = left.copy()
    for k, v in right.items():
        to_ret[k] = v
    return to_ret

class InvariantParams(TypedDict):
    context: HarnessedApplication
    contract_spec: SourceCode

_typed_invariant_prompt = TypedTemplate[InvariantParams]("structural_invariant_prompt.j2")

async def get_invariant_formulation(
    ctx: WorkflowContext[None],
    source: SourceCode,
    env: SourceEnvironment,
    app: HarnessedApplication
) -> Invariants:
    """Run the structural invariant formulation agent.

    An LLM agent reads the contract source, proposes structural invariants,
    and validates each one through a feedback sub-agent. Returns invariants
    that passed all feedback criteria.

    Args:
        ctx: Workflow context for threading, memory, and checkpointing.
        source: Source code metadata (used for template rendering).
        source_tools: Builder with fs_tools for source code reading.

    Returns:
        Validated structural invariants.
    """
    inv_ctx = ctx.child(STRUCTURAL_INV_KEY)
    if (cached := inv_ctx.cache_get(Invariants)) is not None:
        return cached

    judge_ctx = inv_ctx.child(INV_JUDGE_KEY)

    class InvExtra(TypedDict):
        invariant_data: Annotated[
            dict[str, tuple[str, InvFeedbackSort]],
            _merge_invariant_feedback,
        ]

    class ST(MessagesState, InvExtra):
        result: NotRequired[Invariants]

    class InvInput(FlowInput, InvExtra):
        pass

    def _validate_invariants(s: ST, i: Invariants) -> str | None:
        all_invariant_names: set[str] = set()
        for inv in i.inv:
            if inv.name in all_invariant_names:
                return f"Multiple definitions for {inv.name}"
            all_invariant_names.add(inv.name)
            feed_rec = s["invariant_data"].get(inv.name, None)
            if feed_rec is None or feed_rec[0] != inv.description or feed_rec[1] != "GOOD":
                return f"Invariant with name {inv.name} (with description `{inv.description}`) was never accepted by feedback judge"
        return None

    # -- Feedback sub-agent --

    class FeedbackExtra(RoughDraftState):
        pass

    class FeedbackST(MessagesState, FeedbackExtra):
        result: NotRequired[InvariantFeedback]

    class FeedbackInput(FlowInput, FeedbackExtra):
        pass

    feedback_graph = bind_standard(
        env.builder,
        FeedbackST,
    ).with_sys_prompt(
        "You are a methodical formal verification expert working at Certora, Inc."
    ).with_initial_prompt_template(
        "invariant_judge_prompt.j2",
        contract_spec=source,
    ).with_tools(
        [judge_ctx.get_memory_tool(), *get_rough_draft_tools(FeedbackST), *env.source_tools]
    ).with_input(
        FeedbackInput
    ).compile_async()

    sem = asyncio.Semaphore(3)

    class InvariantFeedbackTool(WithInjectedId, WithAsyncImplementation[Command]):
        """
        Receive feedback on one of your invariants.

        You may call this tool in parallel.
        """
        inv: BaseInvariant = Field(description="The invariant to receive feedback on")

        @override
        async def run(self) -> Command:
            async with sem:
                res = await run_to_completion(
                    feedback_graph,
                    FeedbackInput(
                        input=[f"The invariant is called: {self.inv.name}\nStatement: {self.inv.description}"],
                        memory=None,
                        did_read=False,
                    ),
                    thread_id=judge_ctx.uniq_thread_id(),
                    description=f"Invariant feedback: {self.inv.name}",
                )
                assert "result" in res
                feedback: InvariantFeedback = res["result"]
                return Command(update={
                    "messages": [ToolMessage(
                        tool_call_id=self.tool_call_id,
                        content=f"Judgment: {feedback.sort}\nExplanation: {feedback.explanation}",
                    )],
                    "invariant_data": {
                        self.inv.name: (self.inv.description, feedback.sort)
                    },
                })

    # -- Main formulation agent --

    bound_template = _typed_invariant_prompt.bind({
        "context": app,
        "contract_spec": source
    })

    graph = bind_standard(
        env.builder,
        ST,
        doc="The structural/state invariants you identified",
        validator=_validate_invariants,
    ).with_sys_prompt_template(
        "cvl_summarization_system_prompt.j2"
    ).inject(
        lambda g: bound_template.render_to(g.with_initial_prompt_template)
    ).with_tools(
        [inv_ctx.get_memory_tool(), InvariantFeedbackTool.as_tool("invariant_feedback"), *env.source_tools]
    ).with_input(
        InvInput
    ).compile_async()

    st = await run_to_completion(
        graph=graph,
        input=InvInput(input=[], invariant_data={}),
        thread_id=inv_ctx.thread_id,
        description="Structural invariant formulation",
    )

    assert "result" in st
    to_ret: Invariants = st["result"]
    inv_ctx.cache_put(to_ret)
    return to_ret
