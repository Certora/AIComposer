"""Datatypes for the autoprove report.

`AutoProverReport` is the top-level document written to
``certora/ap_report/report.json``. Rule verdicts reuse ProverOutputUtility's
`NodeStatus`; the inferred properties reuse (subclass) composer's
`PropertyFormulation` so the report speaks the same property vocabulary as the
analysis phase. Bump `schema_version` on a breaking change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from prover_output_utility.models import NodeStatus

from composer.spec.prop import PropertyFormulation


@dataclass(frozen=True)
class RuleRef:
    """A reference to one inferred property: the component that owns it and its
    1-based index within that component's property batch (the index the
    ``<stem>.property_rules.json`` mapping resolves to via title order)."""
    component: str
    index: int


class GroupStatus(str, Enum):
    """Aggregated verdict for a high-level property, rolled up from the POU
    `NodeStatus` of its member rules (see :func:`coverage.aggregate_status`):

      - VERIFIED     — every member rule VERIFIED
      - VIOLATED     — any member rule VIOLATED
      - PARTIAL      — some VERIFIED, some not-yet-VERIFIED (but none VIOLATED)
      - INCONCLUSIVE — none VERIFIED, none VIOLATED (all TIMEOUT/ERROR/…)
    """
    VERIFIED = "VERIFIED"
    VIOLATED = "VIOLATED"
    PARTIAL = "PARTIAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class InferredProperty(PropertyFormulation):
    """A `PropertyFormulation` tagged with the component that owns it and its
    1-based index within that component's batch. Inherits ``title``,
    ``methods``, ``sort``, ``description`` from PropertyFormulation."""
    component: str
    index: int = Field(..., ge=1)


class CVLRule(BaseModel):
    """One CVL rule/invariant the agent authored, joined to its prover verdict
    and the inferred properties it implements.

    ``status``, ``line`` and ``duration_seconds`` come from
    ProverOutputUtility's per-rule ``CheckResult`` (verdict + source location),
    not from parsing the spec text. ``property_refs`` come from the component's
    ``property_rules.json`` mapping."""
    name: str
    component: str
    property_refs: list[RuleRef] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.UNKNOWN
    line: int | None = None
    duration_seconds: float | None = None
    prover_link: str | None = None


class HighLevelProperty(BaseModel):
    """A human-readable property covering one or more `CVLRule`s.

    NOT an AIComposer *component* (a structural unit of the contract produced by
    system analysis). A component can own several high-level properties; each
    high-level property groups the CVL rules establishing one auditable claim,
    regardless of how many methods or components they span.

    Identified by a stable ``slug`` (kebab-case, reused across runs via the
    canonical map) and a ``P-NN`` ``id`` assigned during canonical
    reconciliation."""
    id: str = Field(..., pattern=r"^P-\d{2,}$")
    slug: str = Field(..., min_length=1, max_length=64)
    title: str
    description: str
    status: GroupStatus
    rule_names: list[str]


class CoverageReport(BaseModel):
    """Validation outcomes after grouping (see :func:`coverage.validate`)."""
    total_inferred_properties: int
    total_rules: int
    total_groups: int
    rules_per_group_min: int
    rules_per_group_max: int
    rule_coverage_complete: bool
    rules_in_multiple_groups: list[str] = Field(default_factory=list)
    rules_in_no_group: list[str] = Field(default_factory=list)
    status_aggregation_consistent: bool = True
    warnings: list[str] = Field(default_factory=list)


class AutoProverReport(BaseModel):
    """Top-level report document — written to ``certora/ap_report/report.json``.

    Named to disambiguate from `composer.diagnostics.timing.RunSummary` (the
    wall-clock timing aggregator)."""
    schema_version: Literal["1.0"] = "1.0"
    contract_name: str
    run_timestamp_utc: str | None = None
    #: component slug (or "invariants") -> prover run link/path
    prover_links: dict[str, str] = Field(default_factory=dict)
    inferred_properties: list[InferredProperty]
    rules: list[CVLRule]
    high_level_properties: list[HighLevelProperty]
    coverage: CoverageReport


# ---------------------------------------------------------------------------
# Canonical map (stable slug <-> P-NN id across runs)
# ---------------------------------------------------------------------------

class CanonicalEntry(BaseModel):
    """A slug-anchored P-NN entry kept stable across runs. ``anchor_rules`` is
    the rule set the slug was first defined against; future runs whose group
    overlaps it (Jaccard >= reuse threshold) inherit this id/slug/title."""
    id: str = Field(..., pattern=r"^P-\d{2,}$")
    slug: str = Field(..., min_length=1, max_length=64)
    title: str
    anchor_rules: list[str]


class CanonicalMap(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    entries: list[CanonicalEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM grouping I/O (structured-output shapes; a subset of the public types)
# ---------------------------------------------------------------------------

class RuleForGrouping(BaseModel):
    """One rule's context handed to the grouping LLM."""
    name: str
    component: str
    status: NodeStatus
    sorts: list[Literal["invariant", "safety_property", "attack_vector"]]
    property_descriptions: list[str]


class HighLevelPropertyDraft(BaseModel):
    """One high-level property as proposed by the grouping LLM. The final
    ``P-NN`` id is assigned afterward by canonical reconciliation; the LLM
    proposes only slug, title, description, and member rule names."""
    slug: str = Field(..., min_length=1, max_length=64)
    title: str
    description: str
    rule_names: list[str]


class GroupingResult(BaseModel):
    """Structured response from the grouping LLM call."""
    groups: list[HighLevelPropertyDraft]
