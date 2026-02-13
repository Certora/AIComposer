from typing import Callable, Awaitable, NotRequired, override, TypedDict, Literal
import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel, Field

from langchain_core.messages import ToolMessage, HumanMessage

from langgraph.types import Command
from langgraph.graph import MessagesState

from graphcore.graph import FlowInput, Builder, tool_state_update
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
from composer.spec.draft import get_rough_draft_tools

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

    
    class SpecJudgeInput(FlowInput):
        curr_spec: str

    rough_draft_tools = get_rough_draft_tools(ST)

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
    ).with_tools([*rough_draft_tools, memory, get_cvl(ST)]).compile_async(
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

class ExplicitThinking(
    WithImplementation[Command],
    WithInjectedId
):
    """Use this tool to record your reasoning. It will not execute any actions
    or retrieve any information — it only logs your thought for future reference.

    Use it when you need to:
    - Synthesize findings after gathering source files or documentation
    - Plan an implementation approach before writing or modifying code
    - Analyze a prover violation before deciding on a fix
    - Evaluate tradeoffs between multiple strategies for spec changes
    - Verify that your planned changes satisfy all requirements and constraints

    Do NOT use it when:
    - The next step is obvious (e.g., fetching a file, running a test)
    - You are simply executing a known plan step by step
    - You have not yet gathered the information needed to reason usefully

    IMPORTANT: you may not call this tool in parallel with other tools.
    """
    thought: str = Field(
        description=(
            "Your structured reasoning. Include: "
            "what you have learned so far, "
            "what constraints or requirements apply, "
            "what approach you are considering and why, "
            "and any risks or edge cases to watch for."
        )
    )
    @override
    def run(self) -> Command:
        return Command(update={"messages": [
            ToolMessage(tool_call_id=self.tool_call_id, content="Thought recorded."),
            HumanMessage(content="Now, consider your current thought process and carefully evaluate how to proceed.")
        ]})


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

    tools = [put_cvl, put_cvl_raw, FeedbackSchema.as_tool("feedback_tool"), ExploreCodeSchema.as_tool("explore_code"), verifier, get_cvl(ST), ExplicitThinking.as_tool("extended_reasoning")]
    tools.extend(ctx.kb_tools(read_only=False))

    if with_memory:
        tools.append(ctx.get_memory_tool())

    extra_inputs : list[str | dict] = []

    if with_memory:
        last_attempt = ctx.child("last_attempt").cache_get()
        if last_attempt is not None:
            extra_inputs.append("Your last working draft was:")
            extra_inputs.append(last_attempt["cvl"])
        

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

    try:
        r = run_to_completion_sync(
            d,
            FlowInput(input=extra_inputs),
            thread_id=ctx.thread_id
        )
    finally:
        last_state = d.get_state({"configurable": {"thread_id": ctx.thread_id}}).values
        if "curr_spec" in last_state and with_memory:
            ctx.child("last_attempt").cache_put({"cvl": last_state["curr_spec"]})

    assert "result" in r and "curr_spec" in r

    return {
        "commentary": r["result"],
        "cvl": r["curr_spec"]
    }
