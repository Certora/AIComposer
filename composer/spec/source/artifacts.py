"""Prover (autoprove) artifact writer: the ``certora/`` deliverable layout.

A subclass of the shared :class:`composer.spec.artifacts.ArtifactStore`. Adds the
CVL-specific bundle (``specs/``, ``confs/``) and the autoprove report on top of the
base's shared property / commentary / token-usage primitives. The stem / filename /
run-key conventions for a spec (``autospec_{slug}`` vs ``invariants``) are captured by
the :data:`SpecIdentity` sum type, not interpolated at call sites.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from composer.spec.artifacts import ArtifactStore
from composer.spec.context import SourceCode
from composer.spec.cvl_generation import GeneratedCVL
from composer.spec.gen_types import (
    AP_REPORT_DIR, AUTOPROVE_INTERNAL_DIR, CERTORA_DIR, SPECS_DIR, under_project,
)
from composer.spec.prop import PropertyFormulation
from composer.spec.source.prover import prover_config_overlay
from composer.spec.source.report.schema import AutoProverReport
from composer.spec.util import ensure_dir

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentSpec:
    """A per-component generated spec. ``slug`` is the component's slugified name."""
    slug: str

    @property
    def stem(self) -> str:
        return f"autospec_{self.slug}"

    @property
    def spec_filename(self) -> str:
        return f"{self.stem}.spec"

    @property
    def run_key(self) -> str:
        """Key under which this spec's prover run is recorded in the run-link map."""
        return self.slug


@dataclass(frozen=True)
class InvariantSpec:
    """The single structural-invariants spec."""

    @property
    def stem(self) -> str:
        return "invariants"

    @property
    def spec_filename(self) -> str:
        return f"{self.stem}.spec"

    @property
    def run_key(self) -> str:
        return "invariants"


type SpecIdentity = ComponentSpec | InvariantSpec


class ProverArtifactStore(ArtifactStore):
    """Persists the autoprove pipeline's outputs under ``certora/`` (plus
    ``.certora_internal/autoProve/`` diagnostics)."""

    def __init__(self, project_root: str, main_contract: str):
        super().__init__(project_root)
        self._main_contract = main_contract

    def _deliverable_dir(self) -> Path:
        return under_project(self._project_root, CERTORA_DIR)

    def _internal_dir(self) -> Path:
        return under_project(self._project_root, AUTOPROVE_INTERNAL_DIR)

    # -- per-spec -----------------------------------------------------------

    def write_analysis_properties(
        self, spec: SpecIdentity, props: list[PropertyFormulation],
    ) -> None:
        """The analysis-phase properties for ``spec``, accompanying ``{stem}.spec``."""
        self._write_properties(spec.stem, props)

    def write_generated_spec(self, spec: SpecIdentity, result: GeneratedCVL) -> Path:
        """Write a generated spec's whole bundle and return the spec's
        project-root-relative path (e.g. ``certora/specs/invariants.spec``).

        Bundle: ``specs/{stem}.spec``, ``properties/{stem}.commentary.md``,
        ``properties/{stem}.property_rules.json``, and ``confs/{stem}.conf``.
        """
        specs_dir = ensure_dir(self._deliverable_dir() / "specs")
        (specs_dir / spec.spec_filename).write_text(result.cvl)
        self._write_commentary(spec.stem, result.commentary)
        self._write_property_map(
            spec.stem, "property_rules",
            {m.property_title: m.rules for m in result.property_rules},
        )
        spec_path = SPECS_DIR / spec.spec_filename  # project-root-relative
        self._write_conf(spec, result.config, spec_path)
        return spec_path

    def _write_conf(
        self, spec: SpecIdentity, base_config: dict | None, spec_path: Path,
    ) -> None:
        """The prover conf for the run: the generation's final ``state["config"]`` plus
        the fixed run overlay (shared with the live ``verify_spec`` run). No-op if no
        base config."""
        if base_config is None:
            _log.warning("no base config for %s; skipping conf dump", spec.stem)
            return
        conf = prover_config_overlay(
            base_config,
            main_contract=self._main_contract,
            verify_target=f"{self._main_contract}:{spec_path}",
        )
        confs_dir = ensure_dir(self._deliverable_dir() / "confs")
        (confs_dir / f"{spec.stem}.conf").write_text(json.dumps(conf, indent=2))

    # -- run-level ----------------------------------------------------------

    def write_component_runs(self, runs: dict[str, str]) -> None:
        """``{spec run-key: final prover-run link}`` to
        ``.certora_internal/autoProve/components_to_prover_runs.json``."""
        out_dir = ensure_dir(self._internal_dir())
        (out_dir / "components_to_prover_runs.json").write_text(json.dumps(runs, indent=2))

    def write_report(self, report: AutoProverReport) -> None:
        """The autoprove final report to ``certora/ap_report/report.json``."""
        report_dir = ensure_dir(under_project(self._project_root, AP_REPORT_DIR))
        out = report_dir / "report.json"
        out.write_text(report.model_dump_json(indent=2) + "\n")
        _log.info("autoprove report: wrote %s", out)


class ProverSourceCode(SourceCode):
    """``SourceCode`` that exposes the prover artifact store. Construct this in the
    autoprove entry point; analysis / property-inference passes keep taking plain
    ``SourceCode`` since the store is irrelevant to them."""

    @property
    def artifact_store(self) -> ProverArtifactStore:
        return ProverArtifactStore(self.project_root, self.contract_name)
