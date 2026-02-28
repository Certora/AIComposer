"""
CVL generation agent: generates CVL specifications for security properties.

Parameterized by:
- env: GenerationEnv — bundles input, builders, capabilities, and tools
- with_memory: whether to persist memory across runs
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Awaitable, NotRequired, override, Literal

from pydantic import BaseModel, Field

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import BaseTool

from langgraph.types import Command, Checkpointer
from langgraph.graph import MessagesState

from graphcore.graph import FlowInput
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId, WithAsyncImplementation
from graphcore.tools.memory import SqliteMemoryBackend, memory_tool

from composer.spec.context import (
    WorkflowContext, Builders, SourceBuilder,
    CacheKey, CVLGeneration, CVLJudge, Feedback, ThreadProvider,
    SourceCode, SystemDoc,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.cvl_tools import put_cvl_raw, put_cvl, get_cvl
from composer.spec.component import ComponentInst
from composer.tools.thinking import ExplicitThinking, get_rough_draft_tools
from composer.spec.cvl_research import cvl_researcher
from composer.templates.loader import load_jinja_template

CVL_JUDGE_KEY = CacheKey[CVLGeneration, CVLJudge]("judge")
FEEDBACK_KEY = CacheKey[CVLJudge, Feedback]("feedback")

type ExplorationTool = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# GenerationEnv — unified configuration for CVL generation
# ---------------------------------------------------------------------------

class CVLResource(BaseModel):
    import_path: str = Field(description="the path to the resource (relative to `certora/`)")
    required: bool = Field(description="whether this resource *must* be used in the verification process")
    description: str = Field(description="A description of this resource")
    sort: Literal["import"]


@dataclass
class GenerationEnv:
    """Environment configuration for CVL generation.

    Bundles input, builders, and optional capabilities. Each capability
    adds tools and template conditionals.
    """
    input: SourceCode | SystemDoc
    builders: Builders

    # Optional capabilities
    prover_tool: BaseTool | None = None
    resources: list[CVLResource] = field(default_factory=list)
    extra_tools: list[BaseTool] = field(default_factory=list)
    extra_input: list[str | dict] = field(default_factory=list)
    result_tools: tuple[BaseTool, ...] | None = None

    @property
    def has_source(self) -> bool:
        return isinstance(self.input, SourceCode)

    @property
    def has_prover(self) -> bool:
        return self.prover_tool is not None

    @property
    def has_publish(self) -> bool:
        return self.result_tools is not None

    @property
    def base_builder(self):
        return self.builders.cvl if self.has_source else self.builders.cvl_only


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")


type FeedbackTool = Callable[[str], Awaitable[PropertyFeedback]]


def property_feedback_judge(
    ctx: WorkflowContext[CVLJudge],
    env: GenerationEnv,
    inst: ComponentInst | None,
    prop: PropertyFormulation,
    with_memory: bool,
) -> FeedbackTool:

    child = ctx.child(FEEDBACK_KEY)
    has_source = env.has_source
    builder = env.builders.cvl if has_source else env.builders.cvl_only

    class ST(MessagesState):
        memory: NotRequired[str]
        result: NotRequired[PropertyFeedback]
        did_read: NotRequired[bool]
        curr_spec: NotRequired[str]

    class SpecJudgeInput(FlowInput):
        curr_spec: str

    rough_draft_tools = get_rough_draft_tools(ST)

    def did_rough_draft_read(s: ST, _) -> str | None:
        if s.get("did_read", None) is None:
            return "Completion REJECTED: never read rough draft for review"
        return None

    if with_memory:
        mem = ctx.get_memory_tool()
    else:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        mem = memory_tool(SqliteMemoryBackend("dummy", db))

    template_kwargs: dict = {
        "context": inst,
        "has_source": has_source,
        **prop.to_template_args(),
    }
    if has_source:
        assert isinstance(env.input, SourceCode)
        template_kwargs["contract_name"] = env.input.contract_name
        template_kwargs["relative_path"] = env.input.relative_path

    workflow = bind_standard(
        builder, ST, validator=did_rough_draft_read
    ).with_input(
        SpecJudgeInput
    ).with_initial_prompt_template(
        "property_judge_prompt.j2",
        **template_kwargs
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_tools([*rough_draft_tools, mem, get_cvl(ST)]).compile_async(
        checkpointer=ctx.checkpointer
    )

    async def the_tool(
        cvl: str
    ) -> PropertyFeedback:
        input_parts: list[str | dict] = []
        if not has_source:
            input_parts.append("The following is the design document for the application:")
            input_parts.append(env.input.content)
        input_parts.append("The proposed CVL file is")
        input_parts.append(cvl)
        res = await run_to_completion(
            workflow,
            SpecJudgeInput(input=input_parts, curr_spec=cvl),
            thread_id=child.uniq_thread_id(),
        )
        assert "result" in res
        return res["result"]

    return the_tool


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


def _code_explorer(
    ctx: ThreadProvider,
    builder: SourceBuilder,
    checkpointer: Checkpointer,
) -> ExplorationTool:
    """Create a code exploration sub-agent for answering source code questions."""

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
        checkpointer=checkpointer
    )

    async def explore(question: str) -> str:
        res = await run_to_completion(
            workflow,
            FlowInput(input=[question]),
            thread_id=ctx.uniq_thread_id(),
        )
        assert "result" in res
        return res["result"]

    return explore


class UnresolvedCallGuidance(WithImplementation[Command], WithInjectedId):
    """
Invoke this tool to receive guidance on how to deal with verification failures due to havocs caused by
unresolved calls.

You may NOT call this tool in parallel with other tools.
    """
    @override
    def run(self) -> Command:
        return Command(update={
            "messages": [
                ToolMessage(tool_call_id=self.tool_call_id, content="Advice is as follows:"),
                HumanMessage(load_jinja_template("unresolved_call_guidance.j2"))
            ]
        })


class GeneratedCVL(BaseModel):
    commentary: str
    cvl: str


class _LastAttemptCache(BaseModel):
    cvl: str

LAST_ATTEMPT_KEY = CacheKey[CVLGeneration, _LastAttemptCache]("last_attempt")


async def generate_property_cvl(
    ctx: WorkflowContext[CVLGeneration],
    prop: PropertyFormulation,
    feat: ComponentInst | None,
    env: GenerationEnv,
    with_memory: bool,
) -> GeneratedCVL:

    class ST(MessagesState):
        curr_spec: NotRequired[str]
        result: NotRequired[str]

    has_source = env.has_source
    has_prover = env.has_prover
    has_publish = env.has_publish
    base_builder = env.base_builder

    feedback = property_feedback_judge(
        ctx.child(CVL_JUDGE_KEY), env, feat, prop, with_memory,
    )

    researcher = cvl_researcher(ctx, env.builders.cvl_only)

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

    _cvl_research_doc = (
        "Delegate a question about CVL syntax, patterns, or techniques to a research sub-agent. "
        "The sub-agent searches the CVL manual and knowledge base, then delivers a synthesized answer.\n\n"
        "Use this when you need to understand how to express something in CVL, what patterns to "
        "use, or how a specific CVL feature works."
    )
    if has_source:
        _cvl_research_doc += " Do NOT use this for source code questions (use explore_code instead)."

    class CVLResearchSchema(WithAsyncImplementation[str]):
        __doc__ = _cvl_research_doc
        question: str = Field(
            description="A specific question about CVL. "
            "Good: 'How do I use ghost state to track cumulative token transfers?' "
            "Good: 'What is the correct syntax for a preserved block with require statements?' "
            "Bad: 'How does the withdraw function work?' (not a CVL question)"
        )
        @override
        async def run(self) -> str:
            return await researcher(self.question)

    tools: list[BaseTool] = [
        put_cvl, put_cvl_raw,
        FeedbackSchema.as_tool("feedback_tool"),
        CVLResearchSchema.as_tool("cvl_research"),
        get_cvl(ST),
        ExplicitThinking.as_tool("extended_reasoning"),
    ]

    template_kwargs: dict = {
        "context": feat,
        "resources": env.resources,
        "has_source": has_source,
        "has_prover": has_prover,
        "has_publish": has_publish,
        "has_stub_tools": len(env.extra_tools) > 0,
        **prop.to_template_args(),
        "memory": with_memory,
    }

    if has_source:
        assert isinstance(env.input, SourceCode)
        explorer = _code_explorer(ctx, env.builders.source, ctx.checkpointer)

        class ExploreCodeSchema(WithAsyncImplementation[str]):
            """
            Delegate a focused question about the source code to a code exploration sub-agent.
            The sub-agent has its own conversation thread with file tools (list_files, get_file,
            grep_files) and will return a synthesized answer. Use this instead of reading files
            directly when you need to understand a specific aspect of the codebase.
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

        tools.append(ExploreCodeSchema.as_tool("explore_code"))
        template_kwargs["contract_name"] = env.input.contract_name
        template_kwargs["relative_path"] = env.input.relative_path

    if env.prover_tool is not None:
        tools.append(env.prover_tool)
        tools.append(UnresolvedCallGuidance.as_tool("unresolved_call_guidance"))

    # Add extra tools from env (e.g., stub read, field request, typecheck)
    tools.extend(env.extra_tools)

    tools.extend(ctx.kb_tools(read_only=False))

    if with_memory:
        tools.append(ctx.get_memory_tool())

    extra_inputs: list[str | dict] = []

    # Prepend env.extra_input (e.g., current stub content)
    extra_inputs.extend(env.extra_input)

    if with_memory:
        last_attempt = ctx.child(LAST_ATTEMPT_KEY).cache_get(_LastAttemptCache)
        if last_attempt is not None:
            extra_inputs.append("Your last working draft was:")
            extra_inputs.append(last_attempt.cvl)

    # Builder configuration: if result_tools provided, use manual config
    if env.result_tools is not None:
        tools.extend(env.result_tools)
        d = base_builder.with_state(
            ST
        ).with_output_key(
            "result"
        ).with_default_summarizer(
            max_messages=50
        ).with_input(
            FlowInput
        ).with_tools(
            tools
        ).with_sys_prompt_template(
            "cvl_system_prompt.j2"
        ).with_initial_prompt_template(
            "property_generation_prompt.j2",
            **template_kwargs
        ).compile_async(
            checkpointer=ctx.checkpointer
        )
    else:
        d = bind_standard(
            base_builder, ST, "A description of your generated CVL"
        ).with_input(
            FlowInput
        ).with_tools(
            tools
        ).with_sys_prompt_template(
            "cvl_system_prompt.j2"
        ).with_initial_prompt_template(
            "property_generation_prompt.j2",
            **template_kwargs
        ).compile_async(
            checkpointer=ctx.checkpointer
        )

    try:
        r = await run_to_completion(
            d,
            FlowInput(input=extra_inputs),
            thread_id=ctx.thread_id
        )
    finally:
        last_state = d.get_state({"configurable": {"thread_id": ctx.thread_id}}).values
        if "curr_spec" in last_state and with_memory:
            ctx.child(LAST_ATTEMPT_KEY).cache_put(_LastAttemptCache(cvl=last_state["curr_spec"]))

    assert "result" in r

    if has_publish:
        # With publish tools, curr_spec may not be in final state (it was merged into master).
        # The result contains the commentary (or GAVE_UP: reason).
        return GeneratedCVL(
            commentary=r["result"],
            cvl=r.get("curr_spec", ""),
        )
    else:
        assert "curr_spec" in r
        return GeneratedCVL(
            commentary=r["result"],
            cvl=r["curr_spec"]
        )
