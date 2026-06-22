"""State types + completion gate for the foundry test author.

Mirrors ``composer/spec/cvl_generation.py`` for the foundry workflow:

* ``curr_test: str | None`` — the buffered ``.t.sol`` source. Single
  file, single buffer (per the design decision).
* ``expected_failures: dict[str, str]`` — test-name → reason map for tests
  intentionally expected to fail. Populated by ``expect_test_failure``,
  cleared per-key by ``expect_test_passage``. The ``forge_test`` runner
  excludes these from the all-green check.
* ``last_test_names`` — the test-function names reported by the most
  recent ``forge_test`` run (parsed from forge's JSON output). The runner
  records this unconditionally on every run that produced parseable
  results; the publish gate uses it to check the declared property→test
  mapping against the tests that *actually ran*, rather than trusting the
  agent's transcription.
* ``skipped`` / ``property_tests`` / ``validations`` / ``required_validations``
  — same shape as the CVL counterpart, just keyed against ``curr_test``
  for the digest. ``property_tests`` carries the property→test-function
  mapping enforced at publish time via ``validate_property_tests``.
"""

import hashlib
from typing import Annotated, Callable, NotRequired
from typing_extensions import TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from graphcore.graph import FlowInput

from composer.core.state import merge_validation
from composer.spec.context import CacheKey, CVLGeneration, CVLJudge
from composer.spec.cvl_generation import SkippedProperty, _merge_skips


FORGE_TEST_VALIDATION_KEY = "forge_test"

FEEDBACK = "feedback"

# WorkflowContext child key for the feedback judge (derives its memory
# namespace and thread ids). The phantom markers are the CVL-named ones
# because that's the vocabulary WorkflowContext speaks today; the key is
# foundry's own.
FOUNDRY_JUDGE_KEY = CacheKey[CVLGeneration, CVLJudge]("judge")

class FoundryTestExtra(TypedDict):
    curr_test: str | None


class PropertyTestMapping(BaseModel):
    """Maps one property from the batch to the foundry test function(s)
    that demonstrate it."""
    property_title: str = Field(
        description="The unique snake_case title of the property (from the "
        "batch listing) that these tests demonstrate"
    )
    tests: list[str] = Field(
        description="The names of the test functions (``test_*`` / "
        "``testFuzz_*`` / ``invariant_*``) in the test file that demonstrate "
        "this property"
    )


def _merge_expected_failures(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    """An empty reason removes the marking — ``expect_test_failure`` rejects
    empty reasons at the tool boundary, so an empty value can only mean
    ``expect_test_passage``'s delete."""
    to_ret = left.copy()
    for k, v in right.items():
        if not v:
            to_ret.pop(k, None)
            continue
        to_ret[k] = v
    return to_ret


class FoundryGenerationExtra(FoundryTestExtra):
    skipped: Annotated[list[SkippedProperty], _merge_skips]
    property_tests: list[PropertyTestMapping]
    validations: Annotated[dict[str, str], merge_validation]
    required_validations: list[str]
    expected_failures: Annotated[dict[str, str], _merge_expected_failures]
    last_test_names: list[str] | None
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


def validate_property_tests(
    property_tests: list[PropertyTestMapping],
    skipped: list[SkippedProperty],
    titles: list[str],
    ran_test_names: list[str],
) -> str | None:
    """Validate the property→tests mapping declared at completion time.

    Unlike the CVL counterpart (which has to trust the agent's transcription
    of rule names), forge reports the name of every test it ran, so the
    mapping is checked against ground truth in both directions: every test
    name in the mapping must have actually run, and every test that ran must
    be tied back to some property. Plus the usual coverage checks: every
    non-skipped property (referenced by its unique title) maps to at least
    one test, no skipped property is mapped, every referenced title exists,
    and no title is mapped twice.
    """
    valid_titles = set(titles)
    skipped_titles = {s.property_title for s in skipped}
    ran = set(ran_test_names)
    errors: list[str] = []
    mapped: set[str] = set()
    claimed_tests: set[str] = set()
    for m in property_tests:
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
        names = [t.strip() for t in m.tests if t.strip()]
        if not names:
            errors.append(f"Property {m.property_title!r} must map to at least one non-empty test name.")
            continue
        for t in names:
            claimed_tests.add(t)
            if t not in ran:
                errors.append(
                    f"Property {m.property_title!r} claims test {t!r}, but no test by that "
                    "name ran in the stamping forge_test invocation."
                )
    for t in titles:
        if t in skipped_titles or t in mapped:
            continue
        errors.append(f"Property {t!r} is neither skipped nor mapped to any tests.")
    for t in sorted(ran - claimed_tests):
        errors.append(
            f"Test {t!r} ran but is not tied back to any property in the mapping. "
            "Every test in the file must demonstrate one of the batch's properties."
        )
    if errors:
        return (
            "Completion REJECTED: the property_tests mapping is invalid. Fix all of the "
            "following and resubmit:\n- " + "\n- ".join(errors)
        )
    return None
