"""
Tests for CVL generation skip/completion machinery wired through a ReAct graph.

Uses a minimal test tool to exercise the _merge_skips reducer and
check_completion / _compute_digest validation logic end-to-end.
"""
import pytest

from typing import NotRequired, override, Iterable, Callable, Protocol


from dataclasses import dataclass


from langgraph.types import Command
from langgraph.graph import MessagesState


from composer.spec.cvl_generation import (
    CVLGenerationExtra,
    SkippedProperty,
    check_completion,
    _FeedbackSchema,
    _RecordSkipSchema,
    _UnskipSchema,
    FEEDBACK_VALIDATION_KEY,
    FeedbackToolContext,
    FeedbackToolImpl
)

from graphcore.tools.results import result_tool_generator
from graphcore.tools.schemas import WithAsyncImplementation, WithInjectedId
from graphcore.graph import tool_state_update

from graphcore.testing import Scenario, tool_call_raw, ToolCallDict

pytestmark = pytest.mark.asyncio

_RECORD_SKIP_NAME = "record_skip"
_UNSKIP_NAME = "unskip_property"
_FEEDBACK_NAME = "feedback_tool"
_RESULT_NAME = "result"
_PUT_CVL = "put_cvl"

def _skip(property_index: int, reason: str) -> ToolCallDict:
    return tool_call_raw(name=_RECORD_SKIP_NAME, property_index=property_index, reason=reason)

def _unskip(property_index: int) -> ToolCallDict:
    return tool_call_raw(_UNSKIP_NAME, property_index=property_index)

def _feedback() -> ToolCallDict:
    return tool_call_raw(_FEEDBACK_NAME)

def _result(commentary: str) -> ToolCallDict:
    return tool_call_raw(_RESULT_NAME, value=commentary)

def _cvl(the_cvl: str) -> ToolCallDict:
    return tool_call_raw(_PUT_CVL, cvl=the_cvl)

# ---------------------------------------------------------------------------
# Test state and test tools
# ---------------------------------------------------------------------------


class CVLTestState(MessagesState, CVLGenerationExtra):
    result: NotRequired[str]

result_tool = result_tool_generator(
    "result",
    (str, "The commentary"),
    "When you're done with the generation",
    validator=(CVLTestState, lambda st, _res, _id: check_completion(st))
)

class DummyPutCVL(WithAsyncImplementation[Command], WithInjectedId):
    """
    put some cvl
    """
    cvl: str

    @override
    async def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id, content="Success",
            curr_spec=self.cvl
        )

TOOLS = [
    _RecordSkipSchema.as_tool(_RECORD_SKIP_NAME),
    _UnskipSchema.as_tool(_UNSKIP_NAME),
    _FeedbackSchema.as_tool(_FEEDBACK_NAME),
    result_tool,
    DummyPutCVL.as_tool(_PUT_CVL)
]

@dataclass
class Feedback:
    good: bool
    feedback: str

async def dummy_feedback(
    spec: str,
    s: list[SkippedProperty],
) -> Feedback:
    return Feedback(good=True, feedback="")

def any_reason(
    ind: int
):
    return (ind, None)

def with_reason(
    ind: int, reason: str
):
    return (ind, reason)

def feedback_builder(
    *reason_specs: tuple[int, str | None],
    accepted_cvl: str | Iterable[str] | Callable[[str], bool] | None = None,
) -> FeedbackToolImpl:
    skippable = {
        k: v for (k,v) in reason_specs
    }

    async def impl(
        spec: str,
        skipped: list[SkippedProperty]
    ) -> Feedback:
        for sk in skipped:
            if sk.property_index not in skippable:
                return Feedback(good=False, feedback="I don't like your skips")
            reason = skippable[sk.property_index]
            if reason is None:
                continue
            if reason != sk.reason:
                return Feedback(good=False, feedback="I don't like your skip reason")
        if accepted_cvl is None:
            return Feedback(good=True, feedback="")
        elif callable(accepted_cvl):
            return Feedback(good=accepted_cvl(spec), feedback="")
        else:
            return Feedback(good=spec in accepted_cvl, feedback="")

    return impl

def scenario(
    num_props: int,
    feedback_impl: FeedbackToolImpl = dummy_feedback,
    *,
    curr_spec: str | None = None,
    skips: list[SkippedProperty] | None = None,
    required: list[str] | None = None
):
    return Scenario(CVLTestState, *TOOLS).init(
        curr_spec=curr_spec,
        validations={},
        skips=skips if skips else [],
        required_validations=required if required else [FEEDBACK_VALIDATION_KEY]
    ).with_context(FeedbackToolContext(
        feedback_impl, num_props=num_props
    ))


# =========================================================================
# _merge_skips reducer via graph
# =========================================================================


class TestMergeSkipsViaGraph:
    async def test_adds_skip(self):
        result = await scenario(1).turn(
            _skip(property_index=1, reason="too complex")
        ).run()
        assert len(result["skipped"]) == 1
        assert result["skipped"][0].property_index == 1
        assert result["skipped"][0].reason == "too complex"

    async def test_merges_two_skips(self):
        result = await scenario(4).turn(
            _skip(property_index=1, reason="complex"),
            _skip(property_index=3, reason="out of scope")
        ).run()
        assert len(result["skipped"]) == 2
        indices = sorted([s.property_index for s in result["skipped"]])
        assert indices == [1, 3]  # sorted by index

    async def test_overwrites_reason(self):
        result = await scenario(1).turn(
            _skip(1, "old reason")
        ).turn(
            _skip(1, "new reason")
        ).run()
        assert len(result["skipped"]) == 1
        assert result["skipped"][0].reason == "new reason"

    async def test_unskip(self):
        result = await scenario(1).turn(
            _skip(1, "no real reason")
        ).turn(
            _unskip(1)
        ).run()
        assert len(result["skipped"]) == 0

    async def test_skip_is_merge(self):
        result = await scenario(2).turn(
            _skip(1, "because")
        ).turn(
            _skip(2, "it's hard")
        ).run()
        assert len(result["skipped"]) == 2
        assert result["skipped"][0].property_index == 1
        assert result["skipped"][1].property_index == 2

    def skip_error_mapper(self, st: CVLTestState) -> tuple[str, list[SkippedProperty]]:
        return (
            Scenario.last_single_tool(_RECORD_SKIP_NAME, st),
            st["skipped"]
        )

    async def test_invalid_skip_reason(self):
        (msg, st) = await scenario(1).turn(
            _skip(1, "too hard")
        ).turn(
            _skip(1, "")
        ).map(self.skip_error_mapper).run()
        assert len(st) == 1 and st[0].property_index == 1
        assert "A non-empty justification is" in msg


# =========================================================================
# Validation stamping + check_completion via graph
# =========================================================================

class RejectionTest(Protocol):
    def __call__(self, st: CVLTestState, cb: Callable[[str], bool] | None = None) -> bool:
        ...

class StagedRejection(RejectionTest, Protocol):
    def check_reason(
        self, reason: str | Callable[[str], bool]
    ) -> Callable[[CVLTestState], bool]:
        ...

def _curry_fn(
    f: RejectionTest
) -> StagedRejection:
    class Wrapped:
        def __call__(self, st: CVLTestState, cb: Callable[[str], bool] | None = None) -> bool:
            return f(st, cb)
        
        def check_reason(self, reason: str | Callable[[str], bool]) -> Callable[[CVLTestState], bool]:
            if isinstance(reason, str):
                return (lambda st: f(st, lambda msg: reason in msg))
            else:
                return (lambda st: f(st, reason))
    return Wrapped()

@_curry_fn
def is_result_rejection(
    st: CVLTestState, cb: Callable[[str], bool] | None = None
) -> bool:
    return "result" not in st and (msg := Scenario.last_single_tool(
        _RESULT_NAME, st
    )).startswith("Completion REJECTED:") and (cb is None or cb(msg))

def result(
    st: CVLTestState
) -> str:
    assert "result" in st
    return st["result"]

unsat_feedback_message = f"{FEEDBACK_VALIDATION_KEY} validation not satisfied or stale"

class TestValidationLogic:
    async def test_basic_completion(
        self
    ):
        assert await scenario(1, curr_spec="anything").turn(
            _feedback()
        ).turn(
            _result("We did it")
        ).map_run(result) == "We did it"

    async def test_no_spec_no_result(self):
        assert await scenario(1).turn(
            _result("nope")
        ).map_run(is_result_rejection.check_reason("no spec written yet"))

    async def test_no_feedback_no_result(self):
        assert await scenario(1, curr_spec="anything").turn(
            _result("nope")
        ).map_run(is_result_rejection.check_reason(unsat_feedback_message))

    async def test_changed_reason_no_result(self):
        assert await scenario(1, curr_spec="whatever").turns(
            _skip(1, "reason 1"),
            _feedback(),
            _skip(1, "new reason"),
            _result("I'm done")
        ).map_run(is_result_rejection.check_reason(unsat_feedback_message))

    async def test_other_validation_key_unsat(self):
        assert await scenario(1, curr_spec="whatever", required=[FEEDBACK_VALIDATION_KEY, "something_else"]).turn(
            _feedback()
        ).turn(
            _result("I'm done")
        ).map_run(is_result_rejection.check_reason("something_else validation not satisfied or stale"))

    async def test_no_validation_rollbacks(self):
        assert await scenario(1, curr_spec="whatever").turns(
            _skip(1, "reason 1"),
            _feedback(), # stamp reason 1
            _skip(1, "reason 2"),
            _feedback(), # stamp reason 2,
            _skip(1, "reason 1"), # back to reason 1 digest
            _result("I'm done") # should fail, current accepted version is reason 2 digest
        ).map_run(is_result_rejection.check_reason(unsat_feedback_message))

    async def test_different_spec_no_result(self):
        assert await scenario(1, curr_spec="version 1").turns(
            _feedback(),
            _cvl("version 2"),
            _result("I'm done")
        ).map_run(is_result_rejection.check_reason(unsat_feedback_message))

    async def test_unskip_rollsback(self):
        assert await scenario(2, feedback_builder(
            any_reason(1)
        ), curr_spec="the spec").turns(
            _skip(1, "doesn't matter"),
            _feedback(), # should be accepted
            _skip(2, "don't like this"),
            _feedback(), # rejection
            _unskip(2),
            _result("I'm done")
        ).map_run(result) == "I'm done"

    async def test_cvl_rollback(self):
        assert await scenario(1, feedback_impl=feedback_builder(accepted_cvl="version 1"), curr_spec="version 1").turns(
            _feedback(),
            _cvl("version 2"),
            _feedback(),
            _cvl("version 1"),
            _result("I'm done")
        ).map_run(result) == "I'm done"

class TestFeedbackIntegration:
    async def test_picky_skips(self):
        assert await scenario(1, feedback_impl=feedback_builder(
            with_reason(1, "good reason")
        ), curr_spec="version 1").turns(
            _skip(1, "good reason"),
            _feedback(),
            _skip(1, "bad reason"),
            _result("complete")
        ).map_run(is_result_rejection.check_reason(unsat_feedback_message))
