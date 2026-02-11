from typing import Callable, Awaitable, NotRequired, override, TypedDict, Literal
import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel, Field

from langchain_core.messages import ToolMessage

from langgraph.types import Command
from langgraph.graph import MessagesState

from graphcore.graph import FlowInput, Builder
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId, WithAsyncImplementation
from graphcore.tools.memory import SqliteMemoryBackend, memory_tool


from composer.spec.context import WorkspaceContext
from composer.spec.prop import PropertyFormulation
from composer.spec.graph_builder import bind_standard
from composer.spec.cvl_tools import put_cvl_raw, put_cvl, get_cvl
from composer.spec.trunner import run_to_completion, run_to_completion_sync
from composer.spec.component import ComponentInst
from composer.spec.prover import get_prover_tool
from composer.workflow.services import get_checkpointer

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")

type FeedbackTool = Callable[[str], Awaitable[PropertyFeedback]]

def property_feedback_judge(
    ctx: WorkspaceContext,
    builder: Builder[None, None, FlowInput],
    inst: ComponentInst | None,
    prop: PropertyFormulation,
    with_memory: bool
) -> FeedbackTool:
    
    child = ctx.child("feedback")
    
    class ST(MessagesState):
        memory: NotRequired[str]
        result: NotRequired[PropertyFeedback]
        did_read: NotRequired[bool]
        curr_spec: NotRequired[str]

    class GetMemory(WithInjectedState[ST], WithImplementation[Command | str], WithInjectedId):
        """
        Retrieve the rough draft of the feedback
        """
        @override
        def run(self) -> str | Command:
            mem = self.state.get("memory", None)
            if mem is None:
                return "Rough draft not yet written"
            return Command(update={
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content=mem)],
                "did_read": True
            })

    class SpecJudgeInput(FlowInput):
        curr_spec: str

    class SetMemory(WithInjectedId, WithImplementation[Command]):
        """
        Write your rough draft for review
        """
        rough_draft : str = Field(description="The new rough draft of your feedback")

        @override
        def run(self) -> Command:
            return Command(update={
                "memory": self.rough_draft,
                "messages": [ToolMessage(tool_call_id=self.tool_call_id, content="Success")]
            })

    def did_rough_draft_read(s: ST, _) -> str | None:
        h = s.get("did_read", None) is None
        if h is None:
            return "Completion REJECTED: never read rough draft for review"
        return None

    if with_memory:
        memory = ctx.get_memory_tool()
    else:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        memory = memory_tool(SqliteMemoryBackend("dummy", db))

    workflow = bind_standard(
        builder, ST, validator=did_rough_draft_read
    ).with_input(
        SpecJudgeInput
    ).with_initial_prompt_template(
        "property_judge_prompt.j2",
        context=inst,
        **prop.to_template_args()
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_tools([SetMemory.as_tool("write_rough_draft"), GetMemory.as_tool("read_rough_draft"), memory, get_cvl(ST)]).compile_async(
        checkpointer=get_checkpointer()
    )

    async def the_tool(
        cvl: str
    ) -> PropertyFeedback:
        res = await run_to_completion(
            workflow,
            SpecJudgeInput(input=[
                "The proposed CVL file is",
                cvl
            ], curr_spec=cvl),
            thread_id=child.uniq_thread_id(),
        )
        assert "result" in res
        r = res["result"]
        print(f"Returning feedback \n{r.feedback}\nFor:{prop}")
        return r

    return the_tool

type ExplorationTool = Callable[[str], Awaitable[str]]

CODE_EXPLORER_SYS_PROMPT = """\
You are a code exploration assistant analyzing smart contract source code.
You have access to file tools (list_files, get_file, grep_files) to explore the project.

Your job is to answer a specific question about the codebase thoroughly and precisely.

Guidelines:
- Ground every claim in what you find in the source code.
- Include relevant function signatures, state variable declarations, or code snippets in your answer.
- If the question asks about behavior, trace through the actual implementation rather than speculating.
- Be concise: the caller needs a dense, actionable answer, not a walkthrough of your exploration process.
"""

def code_explorer(
    ctx: WorkspaceContext,
    builder: Builder[None, None, FlowInput],
) -> ExplorationTool:

    child = ctx.child("explorer")

    class ST(MessagesState):
        result: NotRequired[str]

    workflow = bind_standard(
        builder, ST, "Your findings about the source code"
    ).with_input(
        FlowInput
    ).with_sys_prompt(
        CODE_EXPLORER_SYS_PROMPT
    ).with_initial_prompt(
        "Answer the following question about the source code"
    ).compile_async(
        checkpointer=get_checkpointer()
    )

    async def explore(question: str) -> str:
        res = await run_to_completion(
            workflow,
            FlowInput(input=[question]),
            thread_id=child.uniq_thread_id(),
        )
        assert "result" in res
        return res["result"]

    return explore


class CVLResource(
    BaseModel
):
    import_path: str = Field(description="the path to the resource (relative to `certora/`)")
    required : bool = Field(description="whether this resource *must* be used in the verification process")
    description: str = Field(description="A description of this resource")
    sort: Literal["import"]

@dataclass
class ProverContext:
    prover_config: dict
    resources: list[CVLResource]

    def with_resources(
        self,
        to_add: list[CVLResource]
    ) -> "ProverContext":
        new_l = self.resources + to_add
        return ProverContext(
            self.prover_config,
            new_l
        )
    
class GeneratedCVL(TypedDict):
    commentary: str
    cvl: str

def generate_property_cvl(
    ctx: WorkspaceContext,
    prover_setup: ProverContext,
    prop: PropertyFormulation,
    feat: ComponentInst | None,
    builder: Builder[None, None, FlowInput],
    with_memory: bool
) -> GeneratedCVL:
    
    class ST(MessagesState):
        curr_spec: NotRequired[str]
        result: NotRequired[str]

    feedback = property_feedback_judge(
        ctx.child("judge"), builder, feat, prop, with_memory
    )

    explorer = code_explorer(
        ctx, builder
    )

    llm = ctx.llm()

    verifier = get_prover_tool(
        llm,
        ST,
        prover_setup.prover_config,
        ctx.contract_name,
        ctx.project_root
    )

    class FeedbackSchema(WithInjectedState[ST], WithAsyncImplementation[str]):
        """
        Receive feedback on your CVL
        """
        @override
        async def run(self) -> str:
            st = self.state
            spec = st.get("curr_spec", None)
            if spec is None:
                return "No spec put yet"
            t = await feedback(spec)
            return f"""
Good? {str(t.good)}
Feedback {t.feedback}
"""

    class ExploreCodeSchema(WithAsyncImplementation[str]):
        """
        Delegate a focused question about the source code to a code exploration sub-agent.
        The sub-agent has its own conversation thread with file tools (list_files, get_file,
        grep_files) and will return a synthesized answer. Use this instead of reading files
        directly when you need to understand a specific aspect of the codebase (e.g. how a
        function modifies state, what access control patterns are used, how two contracts
        interact).
        """
        question: str = Field(
            description="A specific, focused question about the source code. "
            "Good: 'What state variables does withdraw() modify and how?' "
            "Bad: 'Tell me about the contract' "
            "Bad: 'What is the definition of function X?' (read the source directly)"
        )

        @override
        async def run(self) -> str:
            return await explorer(self.question)

    tools = [put_cvl, put_cvl_raw, FeedbackSchema.as_tool("feedback_tool"), ExploreCodeSchema.as_tool("explore_code"), verifier, get_cvl(ST)]
    tools.extend(ctx.kb_tools(read_only=False))

    if with_memory:
        tools.append(ctx.get_memory_tool())

    d = bind_standard(
        builder, ST, "A description of your generated CVL"
    ).with_input(
        FlowInput
    ).with_tools(
        tools
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt_template(
        "property_generation_prompt.j2",
        context=feat,
        resources=prover_setup.resources,
        **prop.to_template_args(),
        memory=with_memory
    ).compile_async(
        checkpointer=get_checkpointer()
    )

    r = run_to_completion_sync(
        d,
        FlowInput(input=[]),
        thread_id=ctx.thread_id
    )
    assert "result" in r and "curr_spec" in r

    return {
        "commentary": r["result"],
        "cvl": r["curr_spec"]
    }
