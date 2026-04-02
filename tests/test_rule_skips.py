"""
Tests for the prover tool, rule skip machinery, and their interactions.

Uses a monkeypatched prover (certora_prover fixture) to test report handling,
validation stamping, and rule skip reducer behavior end-to-end.
"""
import pytest

from composer.spec.source.author import ExpectRuleFailure, ExpectRulePassage
from composer.spec.source.prover import (
    StateWithSkips, VALIDATION_KEY,
)
from composer.spec.cvl_generation import check_completion
from composer.prover.core import RawReport, SummarizedReport

from graphcore.tests.conftest import Scenario, tool_call_raw, ToolCallDict
from graphcore.tools.results import result_tool_generator

from .conftest import ProverMock, ProverToolResponse

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# State type (StateWithSkips already has `result: NotRequired[str]`)
# ---------------------------------------------------------------------------

_PROVER = "verify_spec"
_SKIP = "expect_rule_failure"
_UNSKIP = "expect_rule_passage"
_RESULT = "result"


# ---------------------------------------------------------------------------
# Tool call constructors
# ---------------------------------------------------------------------------


def _verify(rules: list[str] | None = None) -> ToolCallDict:
    return tool_call_raw(_PROVER, rules=rules)


def _verify_rules(*rules: str) -> ToolCallDict:
    return tool_call_raw(_PROVER, rules=list(rules))


def _skip(rule_name: str, reason: str) -> ToolCallDict:
    return tool_call_raw(_SKIP, rule_name=rule_name, reason=reason)


def _unskip(rule_name: str) -> ToolCallDict:
    return tool_call_raw(_UNSKIP, rule_name=rule_name)


def _result(commentary: str) -> ToolCallDict:
    return tool_call_raw(_RESULT, value=commentary)


# ---------------------------------------------------------------------------
# Prover response constructors
# ---------------------------------------------------------------------------


def _raw_report(**rule_status: bool) -> RawReport:
    return RawReport(rule_status=rule_status, report="Prover report output")


def _summarized_report(todo: str, **rule_status: bool) -> SummarizedReport:
    return SummarizedReport(rule_status=rule_status, report="(truncated)", todo_list=todo)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


result_tool = result_tool_generator(
    "result",
    (str, "Commentary"),
    "Signal completion",
    validator=(StateWithSkips, lambda st, *_: check_completion(st)),
)


def _scenario(
    certora_prover: ProverMock,
    *responses: ProverToolResponse,
    curr_spec: str | None = "rule foo { assert true; }",
    rule_skips: dict[str, str] | None = None,
    required: list[str] | None = None,
):
    prover_tool = certora_prover(responses)
    tools = [
        prover_tool,
        ExpectRuleFailure.as_tool(_SKIP),
        ExpectRulePassage.as_tool(_UNSKIP),
        result_tool,
    ]
    return Scenario(StateWithSkips, *tools).init(
        curr_spec=curr_spec,
        skipped=[],
        validations={},
        required_validations=required if required is not None else [VALIDATION_KEY],
        rule_skips=rule_skips or {},
        config={"files": ["src/Foo.sol"]},
    )


# ---------------------------------------------------------------------------
# Extractors for map_run
# ---------------------------------------------------------------------------


def _rule_skips(st: StateWithSkips) -> dict[str, str]:
    return st["rule_skips"]


def _result_accepted(st: StateWithSkips) -> str:
    assert "result" in st
    return st["result"]


def _is_result_rejection(st: StateWithSkips) -> bool:
    return "result" not in st and Scenario.last_single_tool(
        _RESULT, st
    ).startswith("Completion REJECTED:")


# =========================================================================
# Prover report handling
# =========================================================================


class TestProverReportHandling:
    async def test_no_spec_returns_error(self, certora_prover: ProverMock):
        msg = await _scenario(
            certora_prover, curr_spec=None,
        ).turn(
            _verify()
        ).run_last_single_tool(_PROVER)
        assert "not yet" in msg.lower()

    async def test_string_error_passthrough(self, certora_prover: ProverMock):
        msg = await _scenario(
            certora_prover, "Internal prover error: out of memory",
        ).turn(
            _verify()
        ).run_last_single_tool(_PROVER)
        assert "out of memory" in msg

    async def test_summarized_report_returns_todo(self, certora_prover: ProverMock):
        msg = await _scenario(
            certora_prover,
            _summarized_report("1. Fix rule foo\n2. Fix rule bar", foo=False, bar=False),
        ).turn(
            _verify()
        ).run_last_single_tool(_PROVER)
        assert "Fix rule foo" in msg

    async def test_raw_report_failures_no_stamp(self, certora_prover: ProverMock):
        assert await _scenario(
            certora_prover,
            _raw_report(foo=True, bar=False),
        ).turns(
            _verify(),
            _result("done"),
        ).map_run(_is_result_rejection)

    async def test_raw_report_all_verified_stamps(self, certora_prover: ProverMock):
        assert await _scenario(
            certora_prover,
            _raw_report(foo=True, bar=True),
        ).turns(
            _verify(),
            _result("done"),
        ).map_run(_result_accepted) == "done"

    async def test_filtered_rules_dont_stamp(self, certora_prover: ProverMock):
        assert await _scenario(
            certora_prover,
            _raw_report(foo=True),
        ).turns(
            _verify_rules("foo"),
            _result("done"),
        ).map_run(_is_result_rejection)


# =========================================================================
# Rule skip interactions with prover
# =========================================================================


class TestRuleSkipProverInteraction:
    async def test_skipped_failure_counts_as_verified(self, certora_prover: ProverMock):
        """ruleA fails but is skipped, ruleB passes → all verified."""
        assert await _scenario(
            certora_prover,
            _raw_report(ruleA=False, ruleB=True),
        ).turn(
            _skip("ruleA", "known issue"),
        ).turns(
            _verify(),
            _result("done"),
        ).map_run(_result_accepted) == "done"

    async def test_unskipped_failure_blocks_verification(self, certora_prover: ProverMock):
        """Skip ruleA, then unskip it. Prover returns ruleA=fail → not verified."""
        assert await _scenario(
            certora_prover,
            _raw_report(ruleA=False, ruleB=True),
        ).turn(
            _skip("ruleA", "temp"),
        ).turn(
            _unskip("ruleA"),
        ).turns(
            _verify(),
            _result("done"),
        ).map_run(_is_result_rejection)

    async def test_non_skipped_failure_blocks_despite_other_skips(self, certora_prover: ProverMock):
        """ruleA is skipped and fails, ruleB is NOT skipped and also fails → not verified."""
        assert await _scenario(
            certora_prover,
            _raw_report(ruleA=False, ruleB=False),
        ).turn(
            _skip("ruleA", "known"),
        ).turns(
            _verify(),
            _result("done"),
        ).map_run(_is_result_rejection)


# =========================================================================
# Rule skip reducer integration
# =========================================================================


class TestRuleSkipReducer:
    async def test_multiple_skips_merge(self, certora_prover: ProverMock):
        skips = await _scenario(certora_prover).turn(
            _skip("ruleA", "reason A"),
        ).turn(
            _skip("ruleB", "reason B"),
        ).map_run(_rule_skips)
        assert skips == {"ruleA": "reason A", "ruleB": "reason B"}

    async def test_skip_preserves_existing(self, certora_prover: ProverMock):
        skips = await _scenario(
            certora_prover,
            rule_skips={"ruleA": "existing"},
        ).turn(
            _skip("ruleB", "new"),
        ).map_run(_rule_skips)
        assert skips == {"ruleA": "existing", "ruleB": "new"}

    async def test_skip_overwrites_reason(self, certora_prover: ProverMock):
        skips = await _scenario(certora_prover).turn(
            _skip("ruleA", "old"),
        ).turn(
            _skip("ruleA", "new"),
        ).map_run(_rule_skips)
        assert skips["ruleA"] == "new"

    async def test_unskip_removes(self, certora_prover: ProverMock):
        skips = await _scenario(certora_prover).turn(
            _skip("ruleA", "temp"),
        ).turn(
            _unskip("ruleA"),
        ).map_run(_rule_skips)
        assert "ruleA" not in skips
