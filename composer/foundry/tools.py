"""Static tools for the foundry test author.

Foundry-state-bound analogs of the CVL ``static_tools()``:

* ``put_test_raw`` — writes the agent's draft into ``curr_test``. No
  put-time compile check (per design); ``forge_test`` is the gate.
* ``get_test`` — reads ``curr_test`` back. Optional ``set_did_read`` flag
  to flip a state field if the caller's state carries one (it doesn't in
  the default ``FoundryGenerationState``, so the variant exists only for
  parity with the CVL tool).
* ``record_skip`` / ``unskip_property`` — same property-title-keyed skip
  mechanism as CVL, but injected against ``FoundryGenerationState``
  rather than ``CVLGenerationState`` so the schema type-checks against
  the foundry workflow.
* ``expect_test_failure`` / ``expect_test_passage`` — mark/unmark a
  specific ``test_<name>`` as expected-to-fail. The ``forge_test`` runner
  consults ``expected_failures`` and excludes these from the all-green check.

Property-title validation reads from a ``FoundryGenerationContext`` placed
on the runtime by ``batch_foundry_test_generation``.
"""

from dataclasses import dataclass
from typing import override

from langchain_core.tools import BaseTool
from langgraph.runtime import get_runtime
from langgraph.types import Command
from pydantic import Field

from graphcore.graph import tool_state_update
from graphcore.tools.schemas import (
    WithAsyncImplementation, WithImplementation, WithInjectedId, WithInjectedState,
)

from composer.spec.cvl_generation import SkippedProperty
from composer.ui.tool_display import suppress_ack, tool_display

from composer.foundry.state import (
    DELETE_EXPECTED_FAILURE, FoundryGenerationState,
)


@dataclass
class FoundryGenerationContext:
    """Per-batch runtime context.

    ``titles`` are the property titles in the batch — used by
    ``record_skip`` / ``unskip_property`` to validate that the agent is
    talking about a property it was actually asked to formalize."""
    titles: list[str]


# ---------------------------------------------------------------------------
# put_test_raw / get_test
# ---------------------------------------------------------------------------


@tool_display(
    label=lambda p: f"Putting test draft ({len(p.get('test_source', ''))} chars)",
    result=suppress_ack("Put test result", ("Accepted",)),
)
class PutTestRaw(WithImplementation[Command], WithInjectedId):
    """
    Put a foundry test file into the working buffer.

    The provided source replaces the entire ``curr_test`` buffer. There is no
    put-time compile check — call ``forge_test`` to verify the draft actually
    builds and passes. ``forge_test``'s green stamp is invalidated by any
    subsequent ``put_test_raw``, so call ``forge_test`` *after* you're done
    iterating.
    """
    test_source: str = Field(
        description=(
            "The full source of the foundry test file (a single ``.t.sol`` "
            "file's contents). Must declare a contract that extends "
            "``forge-std/Test.sol``'s ``Test`` and contain ``test_*`` "
            "functions for the properties being verified."
        )
    )

    @override
    def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Accepted",
            curr_test=self.test_source,
        )


@tool_display("Reading current test draft", None)
class GetTest(
    WithImplementation[str],
    WithInjectedState[FoundryGenerationState],
):
    """Retrieve the textual representation of the current test draft."""

    @override
    def run(self) -> str:
        if self.state["curr_test"] is None:
            return "No test draft written yet."
        return self.state["curr_test"]


# ---------------------------------------------------------------------------
# Property skip / unskip
# ---------------------------------------------------------------------------


@tool_display(
    lambda p: f"Skipping property `{p.get('property_title', '?')}`",
    suppress_ack("Skip result", ("Recorded skip",)),
)
class _RecordSkipSchema(
    WithInjectedState[FoundryGenerationState],
    WithInjectedId,
    WithImplementation[Command],
):
    """
    Declare that you are skipping a property from the batch.

    You must provide the property's title and a justification. Skipping
    excludes the property from the publish-time property→test mapping
    check; only use after a genuine attempt to formalize.
    """
    property_title: str = Field(
        description="The snake_case title of the property from the batch listing"
    )
    reason: str = Field(
        description="Justification for why this property cannot be formalized as a foundry test"
    )

    @override
    def run(self) -> Command:
        titles = get_runtime(FoundryGenerationContext).context.titles
        if self.property_title not in titles:
            return tool_state_update(
                self.tool_call_id,
                f"Unknown property title {self.property_title!r}. Must be one "
                f"of: {', '.join(titles)}.",
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
    Remove a previously declared skip for a property. Use this if you later
    find a way to formalize a property you previously skipped.
    """
    property_title: str = Field(
        description="The snake_case title of the property to un-skip"
    )

    @override
    def run(self) -> Command:
        titles = get_runtime(FoundryGenerationContext).context.titles
        if self.property_title not in titles:
            return tool_state_update(
                self.tool_call_id,
                f"Unknown property title {self.property_title!r}. Must be one "
                f"of: {', '.join(titles)}.",
            )
        # Sentinel reason "" — _merge_skips drops empty-reason entries.
        skip = SkippedProperty(property_title=self.property_title, reason="")
        return tool_state_update(
            self.tool_call_id,
            f"Removed skip for property {self.property_title}.",
            skipped=[skip],
        )


# ---------------------------------------------------------------------------
# Expected-failure marking (analog of ExpectRuleFailure / ExpectRulePassage)
# ---------------------------------------------------------------------------


@tool_display(lambda p: f"Expecting test `{p['test_name']}` to fail", None)
class ExpectTestFailure(WithAsyncImplementation[Command], WithInjectedId):
    """
    Mark a foundry test as expected to fail.

    The ``forge_test`` runner excludes expected-fail tests from the
    all-green check, so a failing test marked here will not block the
    publish gate. Use only when the failure is the *demonstration* of
    a property (e.g., a regression test that proves a negation).
    """
    test_name: str = Field(
        description="The name of the test function (e.g., `test_RevertWhen_Unauthorized`)"
    )
    reason: str = Field(description="Why this test is expected to fail")

    @override
    async def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Success",
            expected_failures={self.test_name: self.reason},
        )


@tool_display(lambda p: f"Expecting test `{p['test_name']}` to pass", None)
class ExpectTestPassage(WithAsyncImplementation[Command], WithInjectedId):
    """
    Unmark a test previously marked expected-to-fail.

    By default every test is expected to pass; only call this to revert a
    prior ``expect_test_failure``.
    """
    test_name: str = Field(
        description="The name of the test function previously marked expected-to-fail"
    )

    @override
    async def run(self) -> Command:
        return tool_state_update(
            tool_call_id=self.tool_call_id,
            content="Success",
            expected_failures={self.test_name: DELETE_EXPECTED_FAILURE},
        )


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def foundry_static_tools() -> list[BaseTool]:
    """Return the static-tool set the foundry author always installs.

    Excludes ``feedback_tool`` (no judge in this workflow), the CVL guidance
    tools, and the prover/ConfigEdit machinery — those are CVL-specific.
    """
    return [
        PutTestRaw.as_tool("put_test_raw"),
        GetTest.as_tool("get_test"),
        _RecordSkipSchema.as_tool("record_skip"),
        _UnskipSchema.as_tool("unskip_property"),
        ExpectTestFailure.as_tool("expect_test_failure"),
        ExpectTestPassage.as_tool("expect_test_passage"),
    ]
