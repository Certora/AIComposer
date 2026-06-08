"""State types + completion gate for the foundry test author.

Mirrors ``composer/spec/cvl_generation.py`` for the foundry workflow:

* ``curr_test: str | None`` — the buffered ``.t.sol`` source. Single
  file, single buffer (per the design decision).
* ``expected_failures: dict[str, str]`` — test-name → reason map for tests
  intentionally expected to fail. Populated by ``expect_test_failure``,
  cleared per-key by ``expect_test_passage``. The ``forge_test`` runner
  excludes these from the all-green check.
* ``skipped`` / ``property_rules`` / ``validations`` / ``required_validations``
  — same shape as the CVL counterpart, just keyed against ``curr_test``
  for the digest. ``property_rules`` carries the property→test-function
  mapping enforced at publish time (PropertyRuleMapping is reused
  verbatim — its ``rules`` list holds test_ function names here).
"""

import hashlib
from typing import Annotated, Callable, NotRequired
from typing_extensions import TypedDict

from langgraph.graph import MessagesState

from graphcore.graph import FlowInput

from composer.core.state import merge_validation
from composer.spec.cvl_generation import (
    PropertyRuleMapping, SkippedProperty, _merge_skips,
)


# Sentinel reused for expected_failures: an entry whose value equals this key
# removes the skip from the merged dict (analog of ProverStateExtra.DELETE_SKIP).
DELETE_EXPECTED_FAILURE = "__delete_expected_failure"

FORGE_TEST_VALIDATION_KEY = "forge_test"


def _merge_expected_failures(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    to_ret = left.copy()
    for k, v in right.items():
        if v == DELETE_EXPECTED_FAILURE:
            to_ret.pop(k, None)
            continue
        to_ret[k] = v
    return to_ret


class FoundryGenerationExtra(TypedDict):
    curr_test: str | None
    skipped: Annotated[list[SkippedProperty], _merge_skips]
    property_rules: list[PropertyRuleMapping]
    validations: Annotated[dict[str, str], merge_validation]
    required_validations: list[str]
    expected_failures: Annotated[dict[str, str], _merge_expected_failures]
    failed: bool | None


class FoundryGenerationInput(FoundryGenerationExtra, FlowInput):
    pass


class FoundryGenerationState(FoundryGenerationExtra, MessagesState):
    result: NotRequired[str]


def _foundry_digest(curr_test: str, skipped: list[SkippedProperty]) -> str:
    """Stable digest of the publish surface — the buffered test source plus
    the skip declarations. Stamps from ``forge_test`` use this; a subsequent
    ``put_test_raw`` invalidates them by changing ``curr_test``."""
    h = hashlib.md5()
    h.update(curr_test.encode())
    for s in skipped:
        h.update(f"{s.property_title}:{s.reason}".encode())
    return h.hexdigest()


def make_foundry_validation_stamper(
    key: str,
) -> Callable[[FoundryGenerationExtra], dict[str, str]]:
    def stamp(state: FoundryGenerationExtra) -> dict[str, str]:
        return {
            key: _foundry_digest(state["curr_test"] or "", state["skipped"])
        }
    return stamp


def check_foundry_completion(state: FoundryGenerationExtra) -> str | None:
    """Return None if the publish gate is satisfied, otherwise the reason.

    Required validations must have stamps whose digest matches the current
    ``curr_test + skipped`` digest. A stamp that doesn't match is treated as
    stale (the agent edited the test after the stamp was issued)."""
    test = state["curr_test"]
    if test is None:
        return "Completion REJECTED: no test written yet."
    digest = _foundry_digest(test, state["skipped"])
    validations = state["validations"]
    for key in state["required_validations"]:
        if validations.get(key) != digest:
            return (
                f"Completion REJECTED: {key} validation not satisfied or stale."
            )
    return None
