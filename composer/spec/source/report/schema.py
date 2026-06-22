"""Datatypes for the autoprove report.

`AutoProverReport` is the top-level document written to ``certora/ap_report/report.json``.
The report is **property-keyed**: a high-level `PropertyGroup` (a "P-NN" heading) groups the
inferred `FormalizedProperty`s it covers, and a `RuleVerdict` may surface under several groups
(rules repeat; properties partition). Rule verdicts reuse ProverOutputUtility's `NodeStatus` so the
report speaks the same vocabulary as the analysis phase.

The report is a **per-run snapshot** — no guarantee that property/group slugs stay stable across
runs. Bump `schema_version` on a breaking change.
"""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from prover_output_utility.models import NodeStatus

from composer.spec.prop import PropertyFormulation

type RuleName = str
"""A CVL rule/invariant identifier as it appears in the prover report and in a component's
``property_rules`` mapping."""

type ComponentName = str
"""Human name of an AIComposer component (e.g. "Increment"), or "Structural Invariants"."""

type PropertyTitle = str
"""A property's unique snake_case title — the key in a component's ``property_rules`` mapping."""

type RuleRef = tuple[str, RuleName]
"""A rule's identity: ``(spec_file, name)``. A name is only unique within a spec, so the defining
spec file disambiguates a rule re-stated under the same name in another spec (and collapses a single
definition — e.g. an imported structural invariant — seen through several component runs)."""

type PropertyKey = tuple[ComponentName, PropertyTitle]
"""A property's identity: ``(component, title)`` — the cross-reference key groups use for members."""


class GroupStatus(str, Enum):
    """Aggregated verdict for a `PropertyGroup`, rolled up from the `NodeStatus` of the rules its
    member properties are formalized by (see :func:`grouping.aggregate_status`):

      - VERIFIED   — every contributing rule VERIFIED
      - VIOLATED   — any contributing rule VIOLATED
      - PARTIAL    — some VERIFIED, some not-yet-VERIFIED (but none VIOLATED)
      - NO_RESULTS — none VERIFIED, none VIOLATED (all TIMEOUT/ERROR/…)
    """
    VERIFIED = "VERIFIED"
    VIOLATED = "VIOLATED"
    PARTIAL = "PARTIAL"
    NO_RESULTS = "NO_RESULTS"


class RuleVerdict(BaseModel):
    """One CVL rule/invariant and its prover outcome — the verdict table the report references by
    `RuleRef`. Stored once even when properties across several groups are formalized by it, so a
    shared rule carries a single consistent verdict/link. ``status``/``line``/``duration_seconds``
    come from ProverOutputUtility's per-rule ``CheckResult``."""
    name: RuleName
    spec_file: str = Field(
        description="Basename of the spec defining this rule; with `name` it is the rule's identity.",
    )
    status: NodeStatus = NodeStatus.UNKNOWN
    line: int | None = None
    duration_seconds: float | None = None
    prover_link: str | None = None

    @property
    def ref(self) -> RuleRef:
        """This rule's identity ``(spec_file, name)`` — the key properties reference it by."""
        return (self.spec_file, self.name)


class FormalizedProperty(PropertyFormulation):
    """An inferred property (title, methods, sort, description) that at least one CVL rule
    formalizes, tagged with its owning component. ``rule_refs`` are the property→rule edges; the
    render layer labels each edge with this property's ``description``. Distinct from a
    `PropertyGroup`, which is the audit-level grouping of several such properties."""
    component: ComponentName = Field(description="The AIComposer component that owns this property.")
    rule_refs: list[RuleRef] = Field(
        default_factory=list,
        description="Identities of the rules that (jointly) formalize this property.",
    )

    @property
    def key(self) -> PropertyKey:
        """This property's identity ``(component, title)`` — how groups reference it."""
        return (self.component, self.title)


class SkippedClaim(PropertyFormulation):
    """A formalization gap: an inferred property the author deliberately declined to formalize, with
    the recorded reason. The component's generation otherwise succeeded."""
    component: ComponentName
    reason: str = Field(description="Why the author skipped formalizing this property.")


class GaveUpComponent(BaseModel):
    """A formalization gap at component granularity: the component's CVL generation gave up (or
    crashed), so none of its inferred properties were formalized. No per-property reason."""
    component: ComponentName
    properties: list[PropertyFormulation]


class PropertyGroup(BaseModel):
    """An audit-level "P-NN" heading: a synthesized claim over a set of `FormalizedProperty`s (its
    ``members``, by identity). Members partition — each property belongs to exactly one group —
    while a rule may surface under several groups via those members' ``rule_refs``. Identified by
    its kebab-case ``slug``."""
    slug: str = Field(..., min_length=1, max_length=64)
    title: str
    description: str
    status: GroupStatus
    members: list[PropertyKey]


class CoverageReport(BaseModel):
    """Validation outcomes after grouping (see :func:`coverage.validate`)."""
    total_properties: int
    total_rules: int
    total_groups: int
    properties_per_group_min: int
    properties_per_group_max: int
    property_coverage_complete: bool
    properties_in_no_group: list[PropertyKey] = Field(default_factory=list)
    #: rules whose properties span >1 group — expected (rules repeat), reported as a stat not an error
    rules_spanning_multiple_groups: list[RuleName] = Field(default_factory=list)
    skipped_count: int = 0
    gave_up_component_count: int = 0
    dropped_orphan_rules: int = 0
    warnings: list[str] = Field(default_factory=list)


class AutoProverReport(BaseModel):
    """Top-level report document — written to ``certora/ap_report/report.json``."""
    schema_version: Literal["2.0"] = "2.0"
    contract_name: str
    run_timestamp_utc: str | None = None
    #: component name (or "Structural Invariants") -> prover run link/path
    prover_links: dict[ComponentName, str] = Field(default_factory=dict)
    properties: list[FormalizedProperty]
    rules: list[RuleVerdict]
    groups: list[PropertyGroup]
    #: Formalization gaps — properties that exist but no rule formalizes (see the two gap types).
    skipped: list[SkippedClaim] = Field(default_factory=list)
    gave_up_components: list[GaveUpComponent] = Field(default_factory=list)
    coverage: CoverageReport
