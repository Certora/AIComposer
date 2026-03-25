"""
CVL generation agent: generates CVL specifications for security properties.

Parameterized by:
- env: GenerationEnv — bundles input, builders, capabilities, and tools
- with_memory: whether to persist memory across runs
"""

import hashlib
from dataclasses import dataclass
from typing import Annotated, Callable, NotRequired, override, Awaitable
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.runtime import get_runtime

from graphcore.graph import FlowInput, tool_state_update, tool_return, Builder
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId, WithAsyncImplementation

from composer.spec.context import (
    WorkflowContext, CacheKey, CVLGeneration, CVLJudge,
)
from composer.spec.guidance import ERC20TokenGuidance, UnresolvedCallGuidance
from composer.core.state import merge_validation
from composer.spec.prop import PropertyFormulation
from composer.spec.graph_builder import bind_standard, run_to_completion
from composer.cvl.tools import put_cvl_raw, put_cvl, get_cvl
from composer.spec.cvl_research import cvl_research_tool, CVL_RESEARCH_BASE_DOC
from composer.spec.feedback import property_feedback_judge, PropertyFeedback
from composer.spec.gen_types import GenerationEnv

CVL_JUDGE_KEY = CacheKey[CVLGeneration, CVLJudge]("judge")


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------

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
    required_validations: list[str]


def _compute_digest(curr_spec: str, skipped: list[SkippedProperty]) -> str:
    digester = hashlib.md5()
    digester.update(curr_spec.encode())
    for s in skipped:
        digester.update(f"{s.property_index}:{s.reason}".encode())
    return digester.hexdigest()


def check_completion(
    state: CVLGenerationExtra,
) -> str | None:
    """Returns None if valid, error string if not."""
    spec = state["curr_spec"]
    if spec is None:
        return "Completion REJECTED: no spec written yet."
    digest = _compute_digest(spec, state["skipped"])
    validations = state["validations"]
    required = state["required_validations"]
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

@dataclass
class _CVLGenerationContext:
    feedback_thunk: Callable[[str, list[SkippedProperty]], Awaitable[PropertyFeedback]]
    num_props: int


class _FeedbackSchema(WithInjectedState[CVLGenerationState], WithInjectedId, WithAsyncImplementation[Command]):
    """
    Receive feedback on your CVL and any skip declarations.
    The judge will evaluate coverage (all properties accounted for)
    and the validity of any skip justifications.
    """
    @override
    async def run(self) -> Command:
        feedback = get_runtime(_CVLGenerationContext).context.feedback_thunk
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

class _RecordSkipSchema(WithInjectedState[CVLGenerationState], WithInjectedId, WithImplementation[Command]):
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
        num_props = get_runtime(_CVLGenerationContext).context.num_props
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

class _UnskipSchema(WithInjectedId, WithImplementation[Command]):
    """
    Remove a previously declared skip for a property.
    Use this if you later find a way to formalize a property you previously skipped.
    """
    property_index: int = Field(
        description="The 1-indexed property number to un-skip"
    )

    @override
    def run(self) -> Command:
        num_props = get_runtime(_CVLGenerationContext).context.num_props
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


async def generate_batch_cvl(
    ctx: WorkflowContext[CVLGeneration],
    props: list[PropertyFormulation],
    env: GenerationEnv,
    with_memory: bool,
    description: str,
) -> GeneratedCVL:
    required_validations = ["feedback"]

    feedback = property_feedback_judge(
        ctx.child(CVL_JUDGE_KEY), env, props, with_memory,
    )

    _cvl_research_doc = CVL_RESEARCH_BASE_DOC

    tools: list[BaseTool] = [
        put_cvl, put_cvl_raw,
        _FeedbackSchema.as_tool("feedback_tool"),
        _RecordSkipSchema.as_tool("record_skip"),
        _UnskipSchema.as_tool("unskip_property"),
        get_cvl(CVLGenerationState),
        ERC20TokenGuidance.as_tool("erc20_guidance"),
        UnresolvedCallGuidance.as_tool("unresolved_call_guidance"),
    ]

    tools.extend(env.extra_tools)

    extra_inputs: list[str | dict] = env.prompt.cvl_prompt_extras

    for (validation, tool) in env.validation_tools:
        required_validations.append(validation)
        tools.append(tool)

    tools.append(cvl_research_tool(ctx, env.cvl_research, _cvl_research_doc))

    template_kwargs: dict = {
        "properties": props,
        "memory": with_memory,
        **env.prompt.cvl_prompt.args
    }

    tools.extend(ctx.kb_tools(read_only=False))

    if with_memory:
        tools.append(ctx.get_memory_tool())

    if with_memory:
        last_attempt = ctx.child(LAST_ATTEMPT_KEY).cache_get(_LastAttemptCache)
        if last_attempt is not None:
            extra_inputs.append("Your last working draft was:")
            extra_inputs.append(last_attempt.cvl)

    # Builder configuration: if result_tools provided, use manual config
    to_build : Builder[CVLGenerationState, None, None]

    if env.output_tools:
        to_build = env.cvl_authorship.with_state(
            CVLGenerationState
        ).with_output_key(
            "result"
        ).with_default_summarizer(
            max_messages=50
        ).with_tools(
            env.output_tools
        )
    else:
        to_build = bind_standard(
            env.cvl_authorship, CVLGenerationState, "A description of your generated CVL",
            validator=lambda s, _r: check_completion(s),
        )

    d = to_build.with_input(
        CVLGenerationInput
    ).with_tools(
        tools
    ).with_sys_prompt_template(
        "cvl_system_prompt.j2"
    ).with_initial_prompt_template(
        str(env.prompt.cvl_prompt.template),
        **template_kwargs
    ).with_context(
        _CVLGenerationContext
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
                required_validations=required_validations
            ),
            thread_id=ctx.thread_id,
            context=_CVLGenerationContext(
                feedback_thunk=feedback,
                num_props=len(props)
            ),
            description=description,
        )
    finally:
        last_state = d.get_state({"configurable": {"thread_id": ctx.thread_id}}).values
        curr = last_state.get("curr_spec")
        if curr is not None and with_memory:
            ctx.child(LAST_ATTEMPT_KEY).cache_put(_LastAttemptCache(cvl=curr))

    assert "result" in r

    skipped = r["skipped"]

    return GeneratedCVL(
        commentary=r["result"],
        cvl=r["curr_spec"] or "",
        skipped=skipped,
    )
