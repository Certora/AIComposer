"""Foundry artifact writer: the ``certora/foundry/`` deliverable layout.

A subclass of the shared :class:`composer.spec.artifacts.ArtifactStore`. The
generated ``.t.sol`` tests live in the foundry project's own ``test/`` (so forge
finds them); everything else the AI tool produces — per-component property dumps,
property→test maps, commentary, per-test statuses, and the run report — lands
under ``certora/foundry/`` (diagnostics under ``.certora_internal/foundry/``) so a
co-located autoprove run shares the project without clobbering its outputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from composer.foundry.author import GeneratedFoundryTest
from composer.foundry.runner import infer_test_dir
from composer.spec.artifacts import ArtifactStore
from composer.spec.context import SourceCode
from composer.spec.cvl_generation import SkippedProperty
from composer.spec.gen_types import (
    FOUNDRY_DELIVERABLE_DIR, FOUNDRY_INTERNAL_DIR, under_project,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.util import ensure_dir


@dataclass(frozen=True)
class FoundryTestArtifact:
    """A per-component generated foundry test. ``base`` is the (collision-
    disambiguated) component slug — the same one used for the ``.t.sol`` filename
    — so a component's metadata sits next to a predictably-named test file."""
    base: str

    @property
    def stem(self) -> str:
        return f"composer_{self.base}"

    @property
    def test_filename(self) -> str:
        return f"{self.stem}.t.sol"


# ---------------------------------------------------------------------------
# Report schema (what the store serializes)
# ---------------------------------------------------------------------------


class PassedTest(BaseModel):
    status: Literal["passed"] = "passed"
    name: str


class ExpectedFailureTest(BaseModel):
    status: Literal["expected_failure"] = "expected_failure"
    name: str
    reason: str


#: A test forge ran: passed, or an author-declared expected failure (which alone
#: carries a reason). Discriminated on ``status``.
TestStatus = Annotated[PassedTest | ExpectedFailureTest, Field(discriminator="status")]


class ComponentTestStatus(BaseModel):
    """``{stem}.status.json`` — the forge-ground-truth status of one component's
    generated tests, plus any properties the author declared unformalizable."""
    tests: list[TestStatus]
    skipped: list[SkippedProperty]


class FoundryComponentEntry(BaseModel):
    """One generated component in the run report; ``stem`` keys its per-component
    files under ``certora/foundry/properties/``."""
    component: str
    stem: str


class FoundryRunReport(BaseModel):
    """``certora/foundry/report.json`` — the run-level rollup. Per-component detail
    lives in the ``properties/`` files keyed by each entry's ``stem``."""
    n_components: int
    n_properties: int
    components: list[FoundryComponentEntry]
    failures: list[str]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FoundryArtifactStore(ArtifactStore):
    """Persists the foundry pipeline's metadata under ``certora/foundry/`` (plus
    ``.certora_internal/foundry/`` diagnostics) and materializes the ``.t.sol``
    tests into the foundry project's own test dir."""

    def _deliverable_dir(self) -> Path:
        return under_project(self._project_root, FOUNDRY_DELIVERABLE_DIR)

    def _internal_dir(self) -> Path:
        return under_project(self._project_root, FOUNDRY_INTERNAL_DIR)

    def _test_dir(self) -> Path:
        """The foundry project's own test dir (``foundry.toml``'s
        ``[profile.default] test``, defaulting to ``test``) — where forge expects
        the generated ``.t.sol`` files, so they can't live under ``certora/``."""
        return ensure_dir(Path(self._project_root) / infer_test_dir(self._project_root))

    # -- per-component ------------------------------------------------------

    def write_analysis_properties(
        self, artifact: FoundryTestArtifact, props: list[PropertyFormulation],
    ) -> None:
        """The analysis-phase properties for this component, accompanying its test."""
        self._write_properties(artifact.stem, props)

    def write_generated_test(
        self,
        artifact: FoundryTestArtifact,
        result: GeneratedFoundryTest,
    ) -> Path:
        """Materialize a generated test: write the ``.t.sol`` into the foundry
        project's test dir, plus its metadata bundle under ``certora/foundry/`` —
        ``{stem}.commentary.md``, ``{stem}.property_tests.json`` (the property→test
        map), and ``{stem}.status.json`` (each test's pass / expected-failure
        status and any declared skips). Returns the absolute path of the written
        ``.t.sol``."""
        test_path = self._test_dir() / artifact.test_filename
        test_path.write_text(result.test_source)

        self._write_commentary(artifact.stem, result.commentary)
        self._write_property_map(
            artifact.stem, "property_tests",
            {m.property_title: m.tests for m in result.property_tests},
        )
        tests: list[PassedTest | ExpectedFailureTest] = []
        for name in result.ran_tests:
            reason = result.expected_failures.get(name) or ""
            if reason:
                tests.append(ExpectedFailureTest(name=name, reason=reason))
            else:
                tests.append(PassedTest(name=name))
        status = ComponentTestStatus(tests=tests, skipped=result.skipped)
        (self._properties_dir() / f"{artifact.stem}.status.json").write_text(
            status.model_dump_json(indent=2)
        )
        return test_path

    # -- run-level ----------------------------------------------------------

    def write_report(self, report: FoundryRunReport) -> None:
        """The run-level rollup to ``certora/foundry/report.json``. Per-component
        detail lives in the ``properties/`` files written above."""
        out = ensure_dir(self._deliverable_dir()) / "report.json"
        out.write_text(report.model_dump_json(indent=2) + "\n")


class FoundrySourceCode(SourceCode):
    """``SourceCode`` that exposes the foundry artifact store. Construct this in the
    foundry entry point; analysis / property-inference passes keep taking plain
    ``SourceCode``."""

    @property
    def artifact_store(self) -> FoundryArtifactStore:
        return FoundryArtifactStore(self.project_root)
