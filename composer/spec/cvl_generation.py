"""
CVL generation agent: generates CVL specifications for security properties.

Parameterized by:
- env: GenerationEnv — bundles input, builders, capabilities, and tools
- with_memory: whether to persist memory across runs
"""

import hashlib
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, NotRequired, override, Awaitable, Any, Protocol
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool

from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import get_runtime

from graphcore.graph import FlowInput, tool_state_update, tool_return
from graphcore.tools.schemas import WithImplementation, WithInjectedState, WithInjectedId, WithAsyncImplementation

from composer.spec.context import (
    WorkflowContext, CacheKey, CVLGeneration, CVLJudge,
)
from composer.spec.guidance import ERC20TokenGuidance, UnresolvedCallGuidance
from composer.core.state import merge_validation
from composer.spec.graph_builder import run_to_completion
from composer.cvl.tools import put_cvl_raw, put_cvl, get_cvl, edit_cvl
from composer.ui.tool_display import tool_display, suppress_ack

class PropertyFeedbackProtocol(Protocol):
    @property
    def good(self) -> bool: ...

    @property
    def feedback(self) -> str: ...

CVL_JUDGE_KEY = CacheKey[CVLGeneration, CVLJudge]("judge")


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------

class SkippedProperty(BaseModel):
    """A property the agent explicitly decided not to formalize."""
    property_title: str = Field(description="The unique snake_case title of the property from the batch listing")
    reason: str = Field(description="Justification for why this property was skipped")


class PropertyRuleMapping(BaseModel):
    """The rules/invariants in the spec that verify a given property."""
    property_title: str = Field(description="The unique snake_case title of the property (from the batch listing) that these rules verify")
    rules: list[str] = Field(description="The names of the rules/invariants in the spec that verify this property")


class Rebuttal(BaseModel):
    """A rebuttal to a specific piece of feedback from a prior round, backed by evidence.

    File a rebuttal when a prior-round suggestion was tried and provably does not work —
    a typecheck error, a persistent counterexample, a CVL construct that does not parse,
    etc. Do NOT file rebuttals for feedback you merely disagree with; address those by
    revising the spec.
    """
    prior_feedback_reference: str = Field(
        description=(
            "A brief quote from, or clear pointer to, the piece of prior-round feedback "
            "this rebuttal addresses. Just enough for the judge to identify which prior "
            "suggestion you are responding to — not a full transcript."
        )
    )
    evidence_type: Literal[
        "typecheck_failure",
        "counterexample",
        "manual_citation",
        "reasoned",
    ] = Field(
        description=(
            "The basis of the rebuttal. Empirical types (`typecheck_failure`, "
            "`counterexample`, `manual_citation`) carry more weight than `reasoned`; "
            "only use `reasoned` when you genuinely cannot produce tool output or a "
            "manual citation."
        )
    )
    evidence: str = Field(
        description=(
            "The concrete artifact backing the rebuttal: typecheck error text, a "
            "counterexample summary, a manual quote with location, or a brief reasoned "
            "argument. Keep it short and specific — the judge reads this verbatim."
        )
    )


def _merge_skips(
    left: list[SkippedProperty],
    right: list[SkippedProperty],
) -> list[SkippedProperty]:
    """State reducer: merge by property_title (new justification replaces old).

    An entry with an empty reason is a sentinel for "unskipped" — it removes
    the property from the skip list.
    """
    by_title = {s.property_title: s for s in left}
    for s in right:
        by_title[s.property_title] = s
    return sorted(
        (s for s in by_title.values() if s.reason),
        key=lambda s: s.property_title,
    )


class GeneratedCVL(BaseModel):
    commentary: str
    cvl: str
    skipped: list[SkippedProperty] = Field(default_factory=list)
    property_rules: list[PropertyRuleMapping] = Field(default_factory=list)
    # The final prover conf recorded during generation, persisted so that a cache hit
    # (which skips the prover) can still write certora/confs. None for pre-existing cache
    # entries or runs where the prover never ran.
    conf: dict | None = Field(default=None)


# ---------------------------------------------------------------------------
# Completion validation
# ---------------------------------------------------------------------------

class CVLGenerationExtra(TypedDict):
    curr_spec: str | None
    skipped: Annotated[list[SkippedProperty], _merge_skips]
    property_rules: list[PropertyRuleMapping]
    validations: Annotated[dict[str, str], merge_validation]
    required_validations: list[str]


def _compute_digest(curr_spec: str, skipped: list[SkippedProperty]) -> str:
    digester = hashlib.md5()
    digester.update(curr_spec.encode())
    for s in skipped:
        digester.update(f"{s.property_title}:{s.reason}".encode())
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


def validate_property_rules(
    property_rules: list[PropertyRuleMapping],
    skipped: list[SkippedProperty],
    titles: list[str],
) -> str | None:
    """Validate the property->rules mapping declared at completion time.

    ``titles`` is the batch's full set of property titles. Returns None if valid, otherwise
    a single message enumerating all problems. A mapping is valid when every non-skipped
    property (referenced by its unique title) is mapped to at least one rule, no skipped
    property is mapped, every referenced title exists, and no title is mapped twice.
    """
    valid_titles = set(titles)
    skipped_titles = {s.property_title for s in skipped}
    errors: list[str] = []
    mapped: set[str] = set()
    for m in property_rules:
        if m.property_title not in valid_titles:
            errors.append(f"Unknown property title {m.property_title!r} (not one of the batch's properties).")
            continue
        if m.property_title in mapped:
            errors.append(f"Property {m.property_title!r} appears more than once in the mapping.")
            continue
        mapped.add(m.property_title)
        if m.property_title in skipped_titles:
            errors.append(
                f"Property {m.property_title!r} is marked as skipped and must not appear "
                "in the mapping (un-skip it or remove it)."
            )
            continue
        if not any(r.strip() for r in m.rules):
            errors.append(f"Property {m.property_title!r} must map to at least one non-empty rule name.")
    for t in titles:
        if t in skipped_titles or t in mapped:
            continue
        errors.append(f"Property {t!r} is neither skipped nor mapped to any rules.")
    if errors:
        return (
            "Completion REJECTED: the property_rules mapping is invalid. Fix all of the "
            "following and resubmit:\n- " + "\n- ".join(errors)
        )
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


class CVLGenerationState(MessagesState, CVLGenerationExtra):
    result: NotRequired[str]


class _LastAttemptCache(BaseModel):
    cvl: str

LAST_ATTEMPT_KEY = CacheKey[CVLGeneration, _LastAttemptCache]("last_attempt")

DESCRIPTION = "CVL generation"

type FeedbackToolImpl = Callable[
    [str, list[SkippedProperty], list[Rebuttal], str],
    Awaitable[PropertyFeedbackProtocol],
]
"""``(cvl, skipped, rebuttals, within_tool) -> PropertyFeedback``. ``within_tool``
is the calling ``_FeedbackSchema``'s ``tool_call_id``, plumbed through to the
sub-graph's ``run_to_completion`` so its UI panel anchors under the parent
tool widget."""

@dataclass
class FeedbackToolContext:
    feedback_thunk: FeedbackToolImpl
    # The batch's property titles (unique, enforced at extraction). Used to validate that
    # the titles named by record_skip / unskip_property / the result mapping refer to real
    # properties, and to check every non-skipped property is mapped.
    titles: list[str]

FEEDBACK_VALIDATION_KEY = "feedback"

@tool_display("Getting feedback", "Feedback")
class _FeedbackSchema(WithInjectedState[CVLGenerationState], WithInjectedId, WithAsyncImplementation[Command]):
    """
    Receive feedback on your CVL and any skip declarations.
    The judge will evaluate coverage (all properties accounted for)
    and the validity of any skip justifications.

    If a prior-round suggestion from the judge was tried and provably does not work,
    file it in `rebuttals` with concrete evidence (typecheck error text, counterexample
    summary, manual citation). Do NOT file rebuttals for feedback you merely disagree
    with — address those by revising the spec. An empty rebuttal list is the expected
    default; only populate it when you have ground-truth evidence against a prior point.
    """
    rebuttals: list[Rebuttal] = Field(
        default_factory=list,
        description=(
            "Optional rebuttals to specific pieces of prior-round feedback. Each entry "
            "identifies the prior point being rebutted, classifies the evidence "
            "(`typecheck_failure` / `counterexample` / `manual_citation` / `reasoned`), "
            "and supplies the concrete evidence text. Empirical types outweigh reasoned "
            "ones with the judge. Leave empty if you have nothing to rebut."
        ),
    )

    @override
    async def run(self) -> Command:
        feedback = get_runtime(FeedbackToolContext).context.feedback_thunk
        st = self.state
        spec = st["curr_spec"]
        if spec is None:
            return tool_return(self.tool_call_id, "No spec put yet")
        skipped = st["skipped"]
        t = await feedback(spec, skipped, self.rebuttals, self.tool_call_id)
        msg = f"Good? {t.good}\nFeedback {t.feedback}"
        if t.good:
            digest = _compute_digest(spec, skipped)
            return tool_state_update(
                self.tool_call_id, msg,
                validations={FEEDBACK_VALIDATION_KEY: digest},
            )
        return tool_state_update(self.tool_call_id, msg)

@tool_display(
    lambda p: f"Skipping property `{p.get('property_title', '?')}`",
    suppress_ack("Skip result", ("Recorded skip",)),
)
class _RecordSkipSchema(WithInjectedState[CVLGenerationState], WithInjectedId, WithImplementation[Command]):
    """
    Declare that you are skipping a property from the batch.
    You must provide the property's title and a justification.
    The feedback judge will evaluate whether your justification is valid.
    Only use this after genuinely attempting to formalize the property.
    """
    property_title: str = Field(
        description="The snake_case title of the property from the batch listing"
    )
    reason: str = Field(
        description="Justification for why this property cannot be formalized"
    )

    @override
    def run(self) -> Command:
        titles = get_runtime(FeedbackToolContext).context.titles
        if self.property_title not in titles:
            return tool_state_update(
                self.tool_call_id,
                f"Unknown property title {self.property_title!r}. Must be one of: {', '.join(titles)}.",
            )
        if not self.reason.strip():
            return tool_state_update(
                self.tool_call_id,
                "A non-empty justification is required when skipping a property.",
            )
        skip = SkippedProperty(
            property_title=self.property_title,
            reason=self.reason,
        )
        return tool_state_update(
            self.tool_call_id,
            f"Recorded skip for property {self.property_title}.",
            skipped=[skip],
        )

@tool_display(
    lambda p: f"Un-skipping property `{p.get('property_title', '?')}`",
    suppress_ack("Unskip result", ("Removed skip",)),
)
class _UnskipSchema(WithInjectedId, WithImplementation[Command]):
    """
    Remove a previously declared skip for a property.
    Use this if you later find a way to formalize a property you previously skipped.
    """
    property_title: str = Field(
        description="The snake_case title of the property to un-skip"
    )

    @override
    def run(self) -> Command:
        titles = get_runtime(FeedbackToolContext).context.titles
        if self.property_title not in titles:
            return tool_state_update(
                self.tool_call_id,
                f"Unknown property title {self.property_title!r}. Must be one of: {', '.join(titles)}.",
            )
        # Empty reason is the sentinel for "not skipped"
        skip = SkippedProperty(
            property_title=self.property_title,
            reason="",
        )
        return tool_state_update(
            self.tool_call_id,
            f"Removed skip for property {self.property_title}.",
            skipped=[skip],
        )

def static_tools() -> list[BaseTool]:
    return [
        put_cvl, put_cvl_raw,
        _FeedbackSchema.as_tool("feedback_tool"),
        _RecordSkipSchema.as_tool("record_skip"),
        _UnskipSchema.as_tool("unskip_property"),
        get_cvl(CVLGenerationState),
        edit_cvl(CVLGenerationState),
        ERC20TokenGuidance.as_tool("erc20_guidance"),
        UnresolvedCallGuidance.as_tool("unresolved_call_guidance"),
    ]


async def run_cvl_generator[S: CVLGenerationState, C: FeedbackToolContext, I: CVLGenerationInput](
    ctx: WorkflowContext[CVLGeneration],
    d: CompiledStateGraph[S, C, I, Any],
    in_state: I,
    ctxt: C,
    description: str,
    skip_mnemonic: bool = False
) -> S:
    input_copy = in_state["input"].copy()
    last_attempt = await ctx.child(LAST_ATTEMPT_KEY).cache_get(_LastAttemptCache)
    if last_attempt is not None:
        input_copy.append("Your last working draft was (you will need to re-put this onto the VFS):")
        input_copy.append(last_attempt.cvl)
    in_state_copy = in_state.copy()
    in_state_copy["input"] = input_copy
    tid : str
    desc : str
    if not skip_mnemonic:
        tid, mnem = await ctx.thread_and_mnemonic()
        desc = f"{description} ({mnem})"
    else:
        tid = ctx.thread_id
        desc = description
    try:
        r = await run_to_completion(
            d,
            in_state_copy,
            thread_id=tid,
            context=ctxt,
            description=desc,
            recursion_limit=ctx.recursion_limit,
        )
        return r
    finally:
        last_state = (await d.aget_state({"configurable": {"thread_id": ctx.thread_id}})).values
        curr = last_state.get("curr_spec")
        if curr is not None:
            await ctx.child(LAST_ATTEMPT_KEY).cache_put(_LastAttemptCache(cvl=curr))
