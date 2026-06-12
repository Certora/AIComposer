"""Datatypes for the autoprove report.

`AutoProverReport` is the top-level document written to
``certora/ap_report/report.json``. Rule verdicts reuse ProverOutputUtility's
`NodeStatus`; the inferred properties reuse (subclass) composer's
`PropertyFormulation` so the report speaks the same property vocabulary as the
analysis phase. Bump `schema_version` on a breaking change.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from prover_output_utility.models import NodeStatus

from composer.spec.prop import PropertyFormulation, PropertyType

type RuleName = str
"""A CVL rule/invariant identifier as it appears in the prover report and in
``<stem>.property_rules.json``. Used so rule<->property and rule<->group
references read as the foreign keys they are."""


@dataclass(frozen=True)
class PropertyRef:
    """A reference to one inferred property: the component that owns it and the
    property's ``title`` (titles are unique within a component's batch)."""
    component: str
    title: str


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
    """A `PropertyFormulation` tagged with the AIComposer component that owns it.
    Inherits ``title`` (unique within the component), ``methods``, ``sort`` and
    ``description`` from PropertyFormulation."""
    component: str = Field(description="Name of the AIComposer component that owns this property.")


class CVLRule(BaseModel):
    """One CVL rule/invariant the agent authored, joined to its prover verdict
    and the inferred properties it implements.

    ``status``, ``line`` and ``duration_seconds`` come from
    ProverOutputUtility's per-rule ``CheckResult`` (verdict + source location),
    not from parsing the spec text. ``property_refs`` come from the component's
    ``property_rules.json`` mapping."""
    name: RuleName
    component: str = Field(description="Name of the component whose spec declares this rule.")
    spec_file: str | None = Field(
        default=None,
        description="Basename of the spec file that defines this rule; together with `name` it is the rule's identity, so a rule re-stated under the same name in a different spec stays distinct.",
    )
    property_refs: list[PropertyRef] = Field(
        default_factory=list, description="The inferred properties this rule implements."
    )
    status: NodeStatus = NodeStatus.UNKNOWN
    line: int | None = None
    duration_seconds: float | None = None
    prover_link: str | None = None


class HighLevelProperty(BaseModel):
    """A human-readable property covering one or more `CVLRule`s.

    NOT an AIComposer *component* (a structural unit of the contract produced by
    system analysis). A component can own several high-level properties; each
    high-level property groups the CVL rules establishing one auditable claim,
    regardless of how many methods or components they span. Identified by its
    kebab-case ``slug``."""
    slug: str = Field(..., min_length=1, max_length=64)
    title: str
    description: str
    status: GroupStatus
    rule_names: list[RuleName]


class CoverageReport(BaseModel):
    """Validation outcomes after grouping (see :func:`coverage.validate`)."""
    total_inferred_properties: int
    total_rules: int
    total_groups: int
    rules_per_group_min: int
    rules_per_group_max: int
    rule_coverage_complete: bool
    rules_in_multiple_groups: list[RuleName] = Field(default_factory=list)
    rules_in_no_group: list[RuleName] = Field(default_factory=list)
    status_aggregation_consistent: bool = True
    warnings: list[str] = Field(default_factory=list)


class AutoProverReport(BaseModel):
    """Top-level report document — written to ``certora/ap_report/report.json``."""
    schema_version: Literal["1.0"] = "1.0"
    contract_name: str
    run_timestamp_utc: str | None = None
    #: component name (or "Structural Invariants") -> prover run link/path
    prover_links: dict[str, str] = Field(default_factory=dict)
    inferred_properties: list[InferredProperty]
    rules: list[CVLRule]
    high_level_properties: list[HighLevelProperty]
    coverage: CoverageReport


# ---------------------------------------------------------------------------
# LLM grouping I/O (structured-output shapes; a subset of the public types)
# ---------------------------------------------------------------------------

class RuleForGrouping(BaseModel):
    """One rule's context handed to the grouping LLM."""
    name: RuleName = Field(description="The CVL rule identifier.")
    component: str = Field(description="The component whose spec declares the rule.")
    status: NodeStatus = Field(description="The rule's prover verdict.")
    sorts: list[PropertyType] = Field(
        description="The distinct property kinds (invariant/safety_property/attack_vector) the rule's properties carry."
    )
    property_descriptions: list[str] = Field(
        description="English descriptions of the inferred properties this rule implements."
    )


class HighLevelPropertyDraft(BaseModel):
    """One high-level property proposed by the grouping LLM."""
    slug: str = Field(
        ..., min_length=1, max_length=64,
        description="kebab-case ASCII lower-case identifier for the grouping; deterministic for the same conceptual grouping.",
    )
    title: str = Field(description="A 5-12 word human-readable headline for the property.")
    description: str = Field(
        description="1-3 plain-English sentences summarising what the group establishes; do not name the CVL rules."
    )
    rule_names: list[RuleName] = Field(
        description="The CVL rule names in this group; every input rule must appear in exactly one group."
    )


class GroupingResult(BaseModel):
    """The high-level property groups covering every input rule exactly once."""
    groups: list[HighLevelPropertyDraft] = Field(
        description="The high-level property groups; collectively they cover every input rule exactly once."
    )
