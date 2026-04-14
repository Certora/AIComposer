"""
Data models for the security report corpus ingestion pipeline.

These models define the structured data flowing through each pipeline stage:
preprocess → triage → extraction → download → analysis → final output.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Report-level models (triage + extraction)
# ---------------------------------------------------------------------------

class RepoReference(BaseModel):
    """A reference to a source code repository found in a security report."""
    url: str = Field(description="GitHub URL of the repository")
    commit: str | None = Field(
        default=None,
        description="Commit hash or tag the audit was performed against, if mentioned",
    )


class ReportMetadata(BaseModel):
    """High-level metadata extracted from a security report during triage."""
    protocol_name: str = Field(description="Name of the protocol or project being audited")
    protocol_description: str = Field(
        description="Brief description of what the protocol does"
    )
    repo: RepoReference | None = Field(
        default=None,
        description="Source code repository reference, if found in the report",
    )
    report_type: Literal["formal_verification", "manual_audit", "other", "not_evm"] = Field(
        description=(
            "Classification of the report type. "
            "'formal_verification' indicates Certora Prover / CVL-based verification for an EVM based protocol. "
            "'manual_audit' indicates a traditional code review. "
            "'not_evm' indicates a Formal verification project for a different blockchain (Soroban/Stellar/Solana/etc)"
            "'other' for anything else."
        )
    )


class RuleExtraction(BaseModel):
    """A single verification rule mentioned within a property group."""
    name: str = Field(description="The CVL rule or invariant name as it appears in the report")
    status: str = Field(
        description="Verification status as stated in the report (e.g. 'Verified', 'Violated', 'Timeout')"
    )
    description: str = Field(
        description="Natural language description of what this rule checks"
    )
    cloud_run_url: str | None = Field(
        default=None,
        description="Cloud run report URL (prover.certora.com/output/...) associated with this rule, if found",
    )


class PropertyGroup(BaseModel):
    """A group of related properties from a security report.

    Reports typically organize verified properties into high-level groups
    (e.g. 'Solvency Properties', 'Access Control') each containing
    multiple individual rules or invariants.
    """
    id: str = Field(description="Unique identifier for this group within the report (e.g. '1', '2a')")
    title: str = Field(description="Title of the property group (e.g. 'Solvency Properties')")
    status: str = Field(
        description="Overall status of the group (e.g. 'Verified', 'Partially Verified')"
    )
    assumptions: str | None = Field(
        default=None,
        description="Any stated assumptions or preconditions for this group's properties",
    )
    description: str = Field(
        description="Natural language description of what this group of properties covers"
    )
    rules: list[RuleExtraction] = Field(
        description="Individual verification rules belonging to this group"
    )


class RuleRef(BaseModel):
    """A reference to a single rule within a property group.

    Points into a PropertyGroup's rules list by index, avoiding
    duplication of the group context.
    """
    group: PropertyGroup
    rule_index: int

    @property
    def rule(self) -> RuleExtraction:
        return self.group.rules[self.rule_index]


class PropertyGroupExtraction(BaseModel):
    """LLM output: just the extracted property groups (no metadata)."""
    properties: list[PropertyGroup] = Field(
        description="All property groups extracted from the report"
    )


class ReportExtraction(BaseModel):
    """Full structured extraction from a security report PDF.

    Assembled in code by combining LLM-extracted property groups
    with already-known metadata from triage.
    """
    metadata: ReportMetadata
    properties: list[PropertyGroup] = Field(
        description="All property groups extracted from the report"
    )


# ---------------------------------------------------------------------------
# Analysis output
# ---------------------------------------------------------------------------

class CorpusEntry(BaseModel):
    """A single analyzed CVL rule with its source code and context.

    Produced by the analysis agent after locating the rule in the
    cloned repository and extracting its implementation details.
    """
    rule_name: str = Field(description="Name of the CVL rule or invariant")

    # property group data
    property_id: str = Field(description="ID of the parent property group")
    property_title: str = Field(description="Title of the parent property group")
    property_description: str = Field(description="Description of the parent property group")

    # rule data
    rule_description: str = Field(description="Natural language description of what this rule checks extracted from the report")
    status: str = Field(description="Verification status from the report")
    assumptions: str | None = Field(
        default=None, description="Assumptions inherited from the parent property group"
    )

    # extracted data

    cvl_code: str = Field(description="The CVL source code of the rule")
    spec_file: str = Field(description="Path to the .spec file containing this rule (relative to repo root)")
    mechanism: str = Field(
        description=(
            "How the rule achieves its verification goal — e.g. what ghost variables it uses, "
            "what hooks it relies on, what invariant preservation strategy it employs"
        )
    )
    implementation_notes: str = Field(
        description="Notable implementation details: helper functions used, dispatchers, summarizations, etc."
    )
    commentary: str
    extracted_property_description: str | None = Field(default=None)


class UnmatchedRule(BaseModel):
    """A rule mentioned in the report that could not be located in the repository."""
    rule: RuleExtraction
    property_id: str = Field(description="ID of the parent property group")
    reason: str = Field(description="Why the rule could not be matched (e.g. 'not found in any .spec file')")


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------

class ProcessedReport(BaseModel):
    """Complete output of processing a single security report PDF."""
    source_pdf: str = Field(description="Filename of the source PDF")
    metadata: ReportMetadata
    entries: list[CorpusEntry] = Field(description="Successfully analyzed rules")
    unmatched: list[UnmatchedRule] = Field(description="Rules that could not be located in the repo")
    skipped_reason: str | None = Field(
        default=None,
        description="If set, the report was skipped (e.g. not a formal verification report, clone failed)",
    )


# ---------------------------------------------------------------------------
# Pipeline state (cached between stages)
# ---------------------------------------------------------------------------

class PipelineState(BaseModel):
    """Incremental pipeline state, cached after each stage for resumability.

    Each field corresponds to a pipeline stage. A None value indicates
    the stage has not yet run. On resume, the pipeline skips stages
    whose corresponding field is already populated.
    """
    model_config = ConfigDict(extra="ignore")

    pdf_hash: str = Field(description="SHA-256 hash of the PDF file contents")
    pdf_path: str = Field(description="Original path to the PDF file")
    triage: ReportMetadata | None = Field(
        default=None, description="Result of triage stage"
    )
    extraction: ReportExtraction | None = Field(
        default=None, description="Result of extraction stage"
    )
    source_dirs: dict[str, str] = Field(
        default_factory=dict,
        description="Map from cloud run output URL to local source directory path",
    )
    analyzed_trees: dict[str, list[CorpusEntry]] = Field(
        default_factory=dict,
        description="Map from cloud run output URL to analyzed corpus entries",
    )
    unmatched: list[UnmatchedRule] = Field(
        default_factory=list,
        description="Accumulated unmatched rules across all analyzed groups",
    )
    skipped_reason: str | None = Field(
        default=None,
        description="If set, processing was aborted at this reason",
    )

    report_links: list[str] | None = Field(description="All of the output URLs extracted from the report", default=None)
