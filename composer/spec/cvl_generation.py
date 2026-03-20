"""
CVL generation agent: generates CVL specifications for security properties.

Parameterized by:
- env: GenerationEnv — bundles input, builders, capabilities, and tools
- with_memory: whether to persist memory across runs
"""

import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Annotated, Callable, Awaitable, NotRequired, override, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from langgraph.types import Command
from langgraph.graph import MessagesState

from graphcore.graph import FlowInput, tool_state_update, tool_return
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId, WithAsyncImplementation
from graphcore.tools.memory import SqliteMemoryBackend, memory_tool

from composer.spec.context import (
    WorkflowContext, SourceBuilder, CVLBuilder, CVLOnlyBuilder,
    CacheKey, CVLGeneration, CVLJudge, Feedback,
    SourceCode, SystemDoc,
)
from composer.spec.guidance import ERC20TokenGuidance, UnresolvedCallGuidance
from composer.core.state import merge_validation
from composer.spec.prop import PropertyFormulation
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.spec.cvl_tools import put_cvl_raw, put_cvl, get_cvl
from composer.spec.component import ComponentInst
from composer.tools.thinking import ExplicitThinking, RoughDraftState, get_rough_draft_tools
from composer.spec.cvl_research import cvl_research_tool, CVL_RESEARCH_BASE_DOC
from composer.spec.code_explorer import code_explorer_tool_from_builder
from composer.templates.loader import load_jinja_template

CVL_JUDGE_KEY = CacheKey[CVLGeneration, CVLJudge]("judge")
FEEDBACK_KEY = CacheKey[CVLJudge, Feedback]("feedback")



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

    Bundles input, role-based builders, and optional capabilities.
    Each capability adds tools and template conditionals.

    Builder roles:
    - cvl_authorship: main CVL generation agent and feedback judge
    - cvl_research: CVL research sub-agents (manual/KB search only)
    - source_tools: code exploration sub-agent (None if no source code)
    """
    input: SourceCode | SystemDoc
    cvl_authorship: CVLBuilder | CVLOnlyBuilder
    cvl_research: CVLOnlyBuilder
    source_tools: SourceBuilder | None = None

    # Optional capabilities
    prover_tool: BaseTool | None = None
    resources: list[CVLResource] = field(default_factory=list)
    extra_tools: list[BaseTool] = field(default_factory=list)
    extra_input: list[str | dict] = field(default_factory=list)
    result_tools: "ResultToolFactory | None" = None

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
        return self.cvl_authorship


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------

class PropertyFeedback(BaseModel):
    """
    The feedback on the properties
    """
    good: bool = Field(description="Whether the properties are good as is, or if there is room for improvement")
    feedback: str = Field(description="The feedback on the rule if work is needed. Can be empty if there is no feedback")


type FeedbackTool = Callable[[str, list[SkippedProperty]], Awaitable[PropertyFeedback]]


def property_feedback_judge(
    ctx: WorkflowContext[CVLJudge],
    env: GenerationEnv,
    inst: ComponentInst | None,
    props: list[PropertyFormulation],
    with_memory: bool,
) -> FeedbackTool:

    child = ctx.child(FEEDBACK_KEY)
    has_source = env.has_source
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
        "context": inst,
        "has_source": has_source,
        "properties": props,
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
        cvl: str,
        skipped: list[SkippedProperty],
    ) -> PropertyFeedback:
        input_parts: list[str | dict] = []
        if not has_source:
            input_parts.append("The following is the design document for the application:")
            input_parts.append(env.input.content)
        input_parts.append("The proposed CVL file is")
        input_parts.append(cvl)
        if skipped:
            input_parts.append("The following properties were explicitly skipped by the author:")
            for s in skipped:
                input_parts.append(f"  Property {s.property_index}: {s.reason}")
        res = await run_to_completion(
            workflow,
            SpecJudgeInput(input=input_parts, curr_spec=cvl, memory=None, did_read=False),
            thread_id=child.uniq_thread_id(),
            description="Property feedback judge",
        )
        assert "result" in res
        return res["result"]

    return the_tool



class SkippedProperty(BaseModel):
    """A property the agent explicitly decided not to formalize."""
    property_index: int = Field(description="1-indexed property number from the batch listing")
    reason: str = Field(description="Justification for why this property was skipped")


def _merge_skips(
    left: list[SkippedProperty],
    right: list[SkippedProperty],
) -> list[SkippedProperty]:
    """State reducer: merge by property_index (new justification replaces old).

    An entry with an empty reason is a sentinel for "unskipped" — it removes
    the property from the skip list.
    """
    by_index = {s.property_index: s for s in left}
    for s in right:
        by_index[s.property_index] = s
    return sorted(
        (s for s in by_index.values() if s.reason),
        key=lambda s: s.property_index,
    )


class GeneratedCVL(BaseModel):
    commentary: str
    cvl: str
    skipped: list[SkippedProperty] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Completion validation
# ---------------------------------------------------------------------------

class CVLGenerationExtra(TypedDict):
    curr_spec: str | None
    skipped: Annotated[list[SkippedProperty], _merge_skips]
    validations: Annotated[dict[str, str], merge_validation]


def _compute_digest(curr_spec: str, skipped: list[SkippedProperty]) -> str:
    digester = hashlib.md5()
    digester.update(curr_spec.encode())
    for s in skipped:
        digester.update(f"{s.property_index}:{s.reason}".encode())
    return digester.hexdigest()


def check_completion(
    state: CVLGenerationExtra, required: list[str],
) -> str | None:
    """Returns None if valid, error string if not."""
    spec = state["curr_spec"]
    if spec is None:
        return "Completion REJECTED: no spec written yet."
    digest = _compute_digest(spec, state["skipped"])
    validations = state["validations"]
    for key in required:
        if key not in validations or validations[key] != digest:
            return f"Completion REJECTED: {key} validation not satisfied or stale."
    return None


def make_validation_stamper(key: str) -> Callable[[CVLGenerationExtra], dict[str, str]]:
    """Create a stamper for future prover tool integration.

    The stamper reads curr_spec/skipped from state and returns
    a dict suitable for merging into the validations state key.
    """
    def stamp(state: CVLGenerationExtra) -> dict[str, str]:
        return {key: _compute_digest(
            state["curr_spec"] or "",
            state["skipped"],
        )}
    return stamp


class CVLGenerationInput(FlowInput, CVLGenerationExtra):
    pass

type StateValidator = Callable[[CVLGenerationExtra], str | None]
type ResultToolFactory = Callable[[StateValidator], tuple[BaseTool, ...]]


class CVLGenerationState(MessagesState, CVLGenerationExtra):
    result: NotRequired[str]


class _LastAttemptCache(BaseModel):
    cvl: str

LAST_ATTEMPT_KEY = CacheKey[CVLGeneration, _LastAttemptCache]("last_attempt")

DESCRIPTION = "CVL generation"


async def generate_batch_cvl(
    ctx: WorkflowContext[CVLGeneration],
    props: list[PropertyFormulation],
    feat: ComponentInst | None,
    env: GenerationEnv,
    with_memory: bool,
    description: str,
) -> GeneratedCVL:

    num_props = len(props)

    has_source = env.has_source
    has_prover = env.has_prover
    has_publish = env.has_publish
    base_builder = env.base_builder

    required_validations = ["feedback"]
    if has_prover:
        required_validations.append("prover")

    validator: StateValidator = lambda s: check_completion(s, required_validations)

    feedback = property_feedback_judge(
        ctx.child(CVL_JUDGE_KEY), env, feat, props, with_memory,
    )

    class FeedbackSchema(WithInjectedState[CVLGenerationState], WithInjectedId, WithAsyncImplementation[Command]):
        """
        Receive feedback on your CVL and any skip declarations.
        The judge will evaluate coverage (all properties accounted for)
        and the validity of any skip justifications.
        """
        @override
        async def run(self) -> Command:
            st = self.state
            spec = st["curr_spec"]
            if spec is None:
                return tool_return(self.tool_call_id, "No spec put yet")
            skipped = st["skipped"]
            t = await feedback(spec, skipped)
            msg = f"Good? {t.good}\nFeedback {t.feedback}"
            if t.good:
                digest = _compute_digest(spec, skipped)
                return tool_state_update(
                    self.tool_call_id, msg,
                    validations={"feedback": digest},
                )
            return tool_state_update(self.tool_call_id, msg)

    class RecordSkipSchema(WithInjectedState[CVLGenerationState], WithInjectedId, WithImplementation[Command]):
        """
        Declare that you are skipping a property from the batch.
        You must provide the 1-indexed property number and a justification.
        The feedback judge will evaluate whether your justification is valid.
        Only use this after genuinely attempting to formalize the property.
        """
        property_index: int = Field(
            description="The 1-indexed property number from the batch listing"
        )
        reason: str = Field(
            description="Justification for why this property cannot be formalized"
        )

        @override
        def run(self) -> Command:
            if not (1 <= self.property_index <= num_props):
                return tool_state_update(
                    self.tool_call_id,
                    f"Invalid property index {self.property_index}. Must be between 1 and {num_props}.",
                )
            if not self.reason.strip():
                return tool_state_update(
                    self.tool_call_id,
                    "A non-empty justification is required when skipping a property.",
                )
            skip = SkippedProperty(
                property_index=self.property_index,
                reason=self.reason,
            )
            return tool_state_update(
                self.tool_call_id,
                f"Recorded skip for property {self.property_index}.",
                skipped=[skip],
            )

    class UnskipSchema(WithInjectedId, WithImplementation[Command]):
        """
        Remove a previously declared skip for a property.
        Use this if you later find a way to formalize a property you previously skipped.
        """
        property_index: int = Field(
            description="The 1-indexed property number to un-skip"
        )

        @override
        def run(self) -> Command:
            if not (1 <= self.property_index <= num_props):
                return tool_state_update(
                    self.tool_call_id,
                    f"Invalid property index {self.property_index}. Must be between 1 and {num_props}.",
                )
            # Empty reason is the sentinel for "not skipped"
            skip = SkippedProperty(
                property_index=self.property_index,
                reason="",
            )
            return tool_state_update(
                self.tool_call_id,
                f"Removed skip for property {self.property_index}.",
                skipped=[skip],
            )

    _cvl_research_doc = CVL_RESEARCH_BASE_DOC
    if has_source:
        _cvl_research_doc += " Do NOT use this for source code questions (use explore_code instead)."

    tools: list[BaseTool] = [
        put_cvl, put_cvl_raw,
        FeedbackSchema.as_tool("feedback_tool"),
        RecordSkipSchema.as_tool("record_skip"),
        UnskipSchema.as_tool("unskip_property"),
        cvl_research_tool(ctx, env.cvl_research, _cvl_research_doc),
        get_cvl(CVLGenerationState),
        ExplicitThinking.as_tool("extended_reasoning"),
        ERC20TokenGuidance.as_tool("erc20_guidance"),
        UnresolvedCallGuidance.as_tool("unresolved_call_guidance"),
    ]

    template_kwargs: dict = {
        "context": feat,
        "resources": env.resources,
        "has_source": has_source,
        "has_prover": has_prover,
        "has_publish": has_publish,
        "has_stub_tools": len(env.extra_tools) > 0,
        "properties": props,
        "memory": with_memory,
    }

    if has_source:
        assert isinstance(env.input, SourceCode)
        assert env.source_tools is not None
        tools.append(code_explorer_tool_from_builder(env.source_tools))
        template_kwargs["contract_name"] = env.input.contract_name
        template_kwargs["relative_path"] = env.input.relative_path

    if env.prover_tool is not None:
        tools.append(env.prover_tool)

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
        tools.extend(env.result_tools(validator))
        d = base_builder.with_state(
            CVLGenerationState
        ).with_output_key(
            "result"
        ).with_default_summarizer(
            max_messages=50
        ).with_input(
            CVLGenerationInput
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
            base_builder, CVLGenerationState, "A description of your generated CVL",
            validator=lambda s, _r: validator(s),
        ).with_input(
            CVLGenerationInput
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
            CVLGenerationInput(
                input=extra_inputs,
                curr_spec=None,
                skipped=[],
                validations={},
            ),
            thread_id=ctx.thread_id,
            description=description,
        )
    finally:
        last_state = d.get_state({"configurable": {"thread_id": ctx.thread_id}}).values
        curr = last_state.get("curr_spec")
        if curr is not None and with_memory:
            ctx.child(LAST_ATTEMPT_KEY).cache_put(_LastAttemptCache(cvl=curr))

    assert "result" in r

    skipped = r["skipped"]

    if has_publish:
        # With publish tools, curr_spec may not be in final state (it was merged into master).
        # The result contains the commentary (or GAVE_UP: reason).
        return GeneratedCVL(
            commentary=r["result"],
            cvl=r["curr_spec"] or "",
            skipped=skipped,
        )
    else:
        assert r["curr_spec"] is not None
        return GeneratedCVL(
            commentary=r["result"],
            cvl=r["curr_spec"],
            skipped=skipped,
        )
