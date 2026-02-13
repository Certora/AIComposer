from typing import Literal, Annotated, override, NotRequired
import asyncio
import pathlib
import uuid

from pydantic import Field, BaseModel

from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.errors import GraphRecursionError
from langchain_core.messages import ToolMessage

from graphcore.tools.schemas import WithInjectedId, WithAsyncImplementation
from graphcore.graph import Builder, FlowInput

from composer.spec.trunner import run_to_completion_sync, run_to_completion
from composer.spec.context import WorkspaceContext, Builders
from composer.workflow.services import get_checkpointer
from composer.spec.graph_builder import bind_standard
from composer.spec.prop import PropertyFormulation
from composer.spec.cvl_generation import generate_property_cvl, CVLResource, ProverContext, GeneratedCVL
from composer.spec.draft import get_rough_draft_tools

class BaseInvariant(BaseModel):
    """
    A single invariant
    """
    name: str = Field(description="A unique, descriptive name of the invariant. Must not contain spaces (use snake casing if necessary)")
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
    if (cached := inv_ctx.cache_get(Invariants)) is not None:
        return cached

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

    fs = inv_ctx.fs_tools()

    memory = inv_ctx.get_memory_tool()

    judge_ctx = inv_ctx.child(
        "judge"
    )

    def validate_invariants(
        state: ST,
        i: Invariants
    ) -> str | None:
        all_invariant_names = set()
        for inv in i.inv:
            if inv.name in all_invariant_names:
                return f"Multiple definitions for {inv.name}"
            all_invariant_names.add(inv.name)
            if (feed_rec := state["invariant_data"].get(inv.name, None)) is None or \
                feed_rec[0] != inv.description or feed_rec[1] != "GOOD":
                return f"Invariant with name {inv.name} (with description `{inv.description}`) was never accepted by feedback judge"
        
        for inv in i.inv:
            for d in inv.dependencies:
                if d not in all_invariant_names:
                    return f"Invariant {inv.name} references {d}, but no invariant with that name exists."

    class FeedbackST(MessagesState):
        result: NotRequired[InvariantFeedback]
        did_read: NotRequired[bool]
        memory: NotRequired[str]

    feedback_graph = bind_standard(
        builder,
        FeedbackST
    ).with_sys_prompt(
        "You are a methodical formal verification expert working at Certora, Inc."
    ).with_initial_prompt_template(
        "invariant_judge_prompt.j2",
        contract_spec=inv_ctx
    ).with_tools(
        [*fs, judge_ctx.get_memory_tool(), *get_rough_draft_tools(FeedbackST)]
    ).with_input(
        FlowInput
    ).build_async()[0].compile(
        checkpointer=get_checkpointer()
    )

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
                    FlowInput(input=[
                        f"The invariant is called: {self.inv.name}\nStatement: {self.inv.description}"
                    ]),
                    judge_ctx.thread_id + uuid.uuid4().hex[:16]
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
    inv_ctx.cache_put(to_ret)
    return to_ret


def structural_invariants_flow(
    ctx: WorkspaceContext,
    conf: ProverContext,
    builder: Builder[None, None, None],
    builders: Builders,
) -> list[CVLResource]:
    s = _get_invariant_formulation(
        ctx.child("structural-inv"),
        builder
    )
    n_waiting : dict[str, int] = {}
    dependents : dict[str, set[str]]= {}
    worklist : list[str] = []

    i_map : dict[str, Invariant] = {}

    child_invs = ctx.child("inv-formal")

    for inv in s.inv:
        i_map[inv.name] = inv
        if len(inv.dependencies) == 0:
            worklist.append(inv.name)
        else:
            n_waiting[inv.name] = len(inv.dependencies)
            for dep in inv.dependencies:
                if dep not in dependents:
                    dependents[dep] = set()
                dependents[dep].add(inv.name)

    inv_to_impl : dict[str, str] = {}

    to_ret : list[CVLResource] = []

    while len(worklist) > 0:
        to_iter = worklist
        worklist = []
        for w in to_iter:
            to_generate = i_map[w]
            import hashlib

            uniq_key = hashlib.sha256(
                f"{to_generate.name}|{to_generate.description}".encode()
            ).hexdigest()[:16]

            child_key = f"{to_generate.name}-{uniq_key}"

            inv_ctx = child_invs.child(child_key, to_generate.model_dump())

            as_prop = PropertyFormulation(
                methods="invariant",
                description=to_generate.description,
                sort="invariant"
            )
            resources : list[CVLResource] = []

            for d in to_generate.dependencies:
                assert d in inv_to_impl
                resources.append(CVLResource(
                    required=False,
                    import_path=inv_to_impl[d],
                    description=f"A spec file containing the invariant `{d}` which may be necessary to prove this invariant",
                    sort="import"
                ))
            
            gen : GeneratedCVL
            if (cached := inv_ctx.cache_get(GeneratedCVL)) is not None:
                gen = cached
            else:
                try:
                    gen = generate_property_cvl(
                        inv_ctx,
                        conf.with_resources(resources),
                        as_prop,
                        None,
                        builders,
                        with_memory=True
                    )
                except GraphRecursionError:
                    continue
                inv_ctx.cache_put(gen)
            print(gen.commentary)
            inv_name = f"{to_generate.name}.spec"
            to_ret.append(CVLResource(
                import_path=inv_name,
                required=False,
                description=f"A specification file containing the invariant {to_generate.name}, which may be necessary to assume a precondition.",
                sort="import"
            ))
            (pathlib.Path(ctx.project_root) / "certora" / inv_name).write_text(gen.cvl)
            inv_to_impl[to_generate.name] = inv_name

            if to_generate.name in dependents:
                for dep in dependents[to_generate.name]:
                    assert dep in n_waiting
                    n_waiting[dep] -= 1
                    if n_waiting[dep] == 0:
                        worklist.append(dep)

    return to_ret
