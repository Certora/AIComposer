"""Datatypes for the autoprove report.

`AutoProverReport` is the top-level document written to
``certora/ap_report/report.json``. Rule verdicts reuse ProverOutputUtility's
`NodeStatus`; the property formulations reuse (subclass) composer's
`PropertyFormulation` so the report speaks the same property vocabulary as the
analysis phase. Bump `schema_version` on a breaking change.

The report is a **per-run snapshot**: slugs and statuses describe only the
current run.

Two distinct property granularities (see the types): a
`PropertyFormulationWithComponent` is one granular per-component formulation
(~1:1 with a CVL rule); an `ImplementedProperty` is the audit-level grouping of
the rules that establish one claim.
"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from prover_output_utility.models import NodeStatus

from composer.spec.prop import PropertyFormulation, PropertyType

type RuleName = str
"""A CVL rule/invariant identifier as it appears in the prover report and in
``<stem>.property_rules.json``. Used so rule<->property and rule<->group
references read as the foreign keys they are."""

type ComponentName = str
"""Human name of an AIComposer component (e.g. "Increment")."""

type StemName = str
"""A property-dump / spec-file stem (e.g. "autospec_Increment", "invariants")."""

type PropertyTitle = str
"""A property's unique snake_case title — the key in ``<stem>.property_rules.json``."""


class GroupStatus(str, Enum):
    """Aggregated verdict for an implemented property, rolled up from the POU
    `NodeStatus` of its member rules (see :func:`grouping.aggregate_status`):

      - VERIFIED     — every member rule VERIFIED
      - VIOLATED     — any member rule VIOLATED
      - PARTIAL      — some VERIFIED, some not-yet-VERIFIED (but none VIOLATED)
      - NO_RESULTS   — none VERIFIED, none VIOLATED (all TIMEOUT/ERROR/…)
    """
    VERIFIED = "VERIFIED"
    VIOLATED = "VIOLATED"
    PARTIAL = "PARTIAL"
    NO_RESULTS = "NO_RESULTS"


class PropertyFormulationWithComponent(PropertyFormulation):
    """The granular unit: a `PropertyFormulation` (title, methods, sort,
    description) tagged with the AIComposer component that owns it. One per
    component property, ~1:1 with a CVL rule. Distinct from an
    `ImplementedProperty`, which is the audit-level grouping of several rules."""
    component: str = Field(description="Name of the AIComposer component that owns this property.")


class CVLRule(BaseModel):
    """One CVL rule/invariant the agent authored, joined to its prover verdict
    and the property formulations it implements.

    ``status``, ``line`` and ``duration_seconds`` come from
    ProverOutputUtility's per-rule ``CheckResult`` (verdict + source location),
    not from parsing the spec text. ``properties`` are the formulations this rule
    implements (resolved from the component's ``property_rules.json`` mapping),
    embedded so the grouping/render layers need no second join."""
    name: RuleName
    component: str = Field(description="Name of the component whose spec declares this rule.")
    spec_file: str | None = Field(
        default=None,
        description="Basename of the spec file that defines this rule; together with `name` it is the rule's identity, so a rule re-stated under the same name in a different spec stays distinct.",
    )
    properties: list[PropertyFormulationWithComponent] = Field(
        default_factory=list, description="The property formulations this rule implements."
    )
    status: NodeStatus = NodeStatus.UNKNOWN
    line: int | None = None
    duration_seconds: float | None = None
    prover_link: str | None = None


class ImplementedProperty(BaseModel):
    """The audit-level unit: a human-readable property implemented by one or more
    `CVLRule`s. Distinct from a `PropertyFormulationWithComponent` (the granular,
    per-component formulation) — an ImplementedProperty groups the CVL rules that
    together establish one auditable claim, regardless of how many methods or
    components they span. NOT an AIComposer *component*. Identified by its
    kebab-case ``slug``."""
    slug: str = Field(..., min_length=1, max_length=64)
    title: str
    description: str
    status: GroupStatus
    rule_names: list[RuleName]


class CoverageReport(BaseModel):
    """Validation outcomes after grouping (see :func:`coverage.validate`)."""
    total_property_formulations: int
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
    property_formulations: list[PropertyFormulationWithComponent]
    rules: list[CVLRule]
    implemented_properties: list[ImplementedProperty]
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


class ImplementedPropertyDraft(BaseModel):
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
    groups: list[ImplementedPropertyDraft] = Field(
        description="The high-level property groups; collectively they cover every input rule exactly once."
    )
