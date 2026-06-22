"""Shared base for run-scoped artifact writers.

Each workflow's ``ArtifactStore`` owns its own deliverable layout so the path
conventions live in one place rather than smeared across the pipeline. This base
hosts what is *identical* across workflows — the analysis-phase
``properties.json``, the ``{property title: [demonstrating names]}`` map,
``commentary.md``, and the run's ``token_usage.json`` — keyed off a per-component
``stem`` plus two abstract directories. The workflow-specific bundles (CVL
``.spec``/``.conf`` for the prover, ``.t.sol`` metadata for foundry) live in the
subclasses, which translate their domain objects into ``stem``s and call these
primitives.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from composer.diagnostics.timing import RunSummary
from composer.spec.gen_types import PROPERTIES_SUBDIR
from composer.spec.prop import PropertyFormulation
from composer.spec.util import ensure_dir


class ArtifactStore(ABC):
    """Persists a pipeline's outputs under a single project root.

    Holds only the run-constant project root, so it is cheap to construct ad hoc
    wherever a write is needed. Subclasses fix the deliverable / diagnostics
    directories and add their format-specific bundles.
    """

    def __init__(self, project_root: str):
        self._project_root = project_root

    @abstractmethod
    def _deliverable_dir(self) -> Path:
        """Absolute base dir (under the project root) for human-facing deliverables."""

    @abstractmethod
    def _internal_dir(self) -> Path:
        """Absolute base dir (under the project root) for run diagnostics."""

    def _properties_dir(self) -> Path:
        return ensure_dir(self._deliverable_dir() / PROPERTIES_SUBDIR)

    # -- shared per-component primitives ------------------------------------

    def _write_properties(self, stem: str, props: list[PropertyFormulation]) -> None:
        """Analysis-phase properties → ``{deliverable}/properties/{stem}.properties.json``.
        ``title`` is the cross-reference key used by the accompanying property map."""
        (self._properties_dir() / f"{stem}.properties.json").write_text(
            json.dumps([p.model_dump() for p in props], indent=2)
        )

    def _write_commentary(self, stem: str, commentary: str) -> None:
        (self._properties_dir() / f"{stem}.commentary.md").write_text(commentary)

    def _write_property_map(
        self, stem: str, suffix: str, mapping: dict[str, list[str]],
    ) -> None:
        """A ``{property title: [demonstrating names]}`` map → ``{stem}.{suffix}.json``.
        Titles are unique (enforced at extraction). ``suffix`` is the workflow's term
        for the demonstrators (``property_rules`` for CVL, ``property_tests`` for foundry)."""
        (self._properties_dir() / f"{stem}.{suffix}.json").write_text(
            json.dumps(mapping, indent=2)
        )

    # -- shared run-level ---------------------------------------------------

    def write_token_usage(self, summary: RunSummary) -> None:
        """The run's accumulated LLM token usage → ``{internal}/token_usage.json``.

        Raw counts only (``input`` / ``output`` / ``cache_read`` / ``cache_write``),
        broken down ``by_model`` / ``by_phase`` plus run-wide ``totals``. Captures every
        call through the LLM factory (including out-of-graph prover/CEX side-calls) via
        the usage callback attached at model construction."""
        payload = {"run_id": summary.run_id, **summary.token_usage_summary()}
        (ensure_dir(self._internal_dir()) / "token_usage.json").write_text(
            json.dumps(payload, indent=2)
        )
