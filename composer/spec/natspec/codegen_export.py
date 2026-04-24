"""Export natspec pipeline results into codegen-ready input JSON files.

Each ``new`` contract in the pipeline result becomes one codegen input JSON
plus the artifacts it references (generated interface, stub, spec files).
The emitted JSON matches the schema expected by
``composer/input/files.py::_upload_from_json``, so each file can be fed
straight to the codegen CLI::

    python main.py --input-json <exported-json-path>

## Write layout

Two modes, keyed on whether ``source_root`` is supplied.

**From-source** (``source_root`` is set): generated artifacts land INSIDE
the workspace at their agent-chosen paths — this is the natspec output
landing in the user's source tree. Generated specs are placed under
``<source_root>/certora/specs/<contract>/`` (agent-chosen spec filenames
when available, else ``<contract>_<idx>.spec``). The per-contract input
JSON still lives under ``<output_root>/<contract>/input.json`` and
references the workspace via absolute paths.

**Greenfield**: everything — interface, stub, specs, and input JSON — lives
under ``<output_root>/<contract>/`` in a self-contained layout. The JSON
references artifacts by absolute path.

## What gets written

For each contract exported::

    <output_root>/<contract>/input.json          # codegen-ready JSON

    # In from-source (paths relative to <source_root>):
    <source_root>/<interface.path>               # generated interface
    <source_root>/<stub.path>                    # generated stub (reference)
    <source_root>/certora/specs/<contract>/<spec_filename>  # each spec

    # In greenfield (paths relative to <output_root>/<contract>/):
    <output_root>/<contract>/<interface.path>
    <output_root>/<contract>/<stub.path>
    <output_root>/<contract>/specs/<spec_filename>

The returned ``ExportedContract`` list gives the absolute paths of every
file touched, per contract.

## Notes

- The function is tolerant of ContractFormulation's evolving shape: both
  the current ``spec: str`` and the anticipated ``specs: list[...]`` (with
  either strings or objects carrying ``filename``/``name`` + ``content``)
  are handled.
- Contracts whose generation produced no spec content at all are skipped
  silently; the caller can cross-reference against the original pipeline
  result's ``failures`` list to diagnose.
- Existing files at target paths are overwritten. Callers working in a
  shared workspace should snapshot first if safety is needed.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from composer.spec.natspec.pipeline import PipelineResult


@dataclass
class ExportedContract:
    """Manifest of what was written for one contract task.

    ``input_json`` is the path to feed to ``python main.py --input-json``.
    The rest are the artifact paths the JSON references (absolute).
    """
    name: str
    input_json: pathlib.Path
    interface: pathlib.Path
    stub: pathlib.Path | None
    specs: list[pathlib.Path]


def _spec_entries(formulation: Any) -> list[tuple[str, str]]:
    """Return ``(filename, content)`` tuples for every spec attached to a
    ContractFormulation, normalizing across shape evolutions.

    Supported shapes:
    - ``formulation.specs = list[str]``                — contents only, filenames synthesized
    - ``formulation.specs = list[<object with .content and .filename|.name>]``
    - ``formulation.spec = <str>`` (legacy single-spec shape)
    """
    specs_attr = getattr(formulation, "specs", None)
    if isinstance(specs_attr, list):
        out: list[tuple[str, str]] = []
        for i, entry in enumerate(specs_attr):
            if isinstance(entry, str):
                out.append((f"{formulation.name}_{i}.spec", entry))
            else:
                basename = (
                    getattr(entry, "filename", None)
                    or getattr(entry, "name", None)
                    or f"{formulation.name}_{i}.spec"
                )
                if not basename.endswith(".spec"):
                    basename = f"{basename}.spec"
                content = getattr(entry, "content", None)
                if not content:
                    continue
                out.append((basename, content))
        return out

    legacy_spec = getattr(formulation, "spec", None)
    if legacy_spec:
        return [(f"{formulation.name}.spec", legacy_spec)]
    return []


def _write_file(dst: pathlib.Path, content: str) -> pathlib.Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content)
    return dst


def export_to_codegen_inputs(
    result: "PipelineResult",
    *,
    output_root: pathlib.Path,
    system_doc_path: pathlib.Path,
    source_root: pathlib.Path | None = None,
    prover_conf: dict | None = None,
) -> list[ExportedContract]:
    """Write natspec artifacts to disk and produce one codegen input JSON
    per contract.

    Args:
        result: The ``PipelineResult`` returned by ``run_natspec_pipeline``.
        output_root: Directory under which per-contract subdirectories and
            their ``input.json`` files are created. In greenfield mode,
            artifacts also land here.
        system_doc_path: Absolute or relative path to the original system
            document; copied into each emitted JSON by absolute path so the
            codegen CLI can find it.
        source_root: When set, switches to from-source layout: artifacts are
            written INTO the workspace at their agent-chosen paths and the
            JSON references the workspace. When ``None``, artifacts land
            under ``output_root`` alongside the JSON.
        prover_conf: Optional per-task prover config. Carried into each JSON
            verbatim under the ``prover_conf`` key.

    Returns:
        One ``ExportedContract`` per exported contract, in the order they
        appear in ``result.contracts``. Contracts with no spec content are
        skipped (no entry returned).
    """
    output_root = pathlib.Path(output_root).absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    system_doc_abs = pathlib.Path(system_doc_path).absolute()
    workspace = (
        pathlib.Path(source_root).absolute() if source_root is not None else None
    )

    exported: list[ExportedContract] = []
    for formulation in result.contracts:
        contract_name = formulation.name
        specs = _spec_entries(formulation)
        if not specs:
            # Nothing useful to ship for this contract; skip. Callers can
            # diagnose via the pipeline result's .failures list if needed.
            continue

        contract_out = output_root / contract_name
        contract_out.mkdir(parents=True, exist_ok=True)

        # Artifact write root + spec placement depend on mode.
        if workspace is not None:
            artifact_root = workspace
            spec_rel_dir = pathlib.Path("certora/specs") / contract_name
        else:
            artifact_root = contract_out
            spec_rel_dir = pathlib.Path("specs")

        # Interface is always present (natspec generated it).
        interface_abs = _write_file(
            artifact_root / formulation.interface.path,
            formulation.interface.content,
        )

        # Stub is optional but usually present. Included for reviewer
        # inspection; codegen itself doesn't consume the stub file.
        stub = getattr(formulation, "stub", None)
        stub_abs: pathlib.Path | None = None
        stub_path_hint: str | None = None
        if stub is not None and getattr(stub, "path", None) and getattr(stub, "content", None):
            stub_abs = _write_file(
                artifact_root / stub.path,
                stub.content,
            )
            stub_path_hint = stub.path  # tell codegen where its impl should land

        spec_abs_paths: list[pathlib.Path] = []
        for (basename, content) in specs:
            spec_abs_paths.append(
                _write_file(artifact_root / spec_rel_dir / basename, content)
            )

        task: dict[str, Any] = {
            "name": contract_name,
            "interface": str(interface_abs),
            "system_doc": str(system_doc_abs),
            "specs": [str(p) for p in spec_abs_paths],
        }
        if stub_path_hint is not None:
            task["implementation_path"] = stub_path_hint
        if workspace is not None:
            task["source_root"] = str(workspace)
        if prover_conf is not None:
            task["prover_conf"] = prover_conf

        json_path = contract_out / "input.json"
        json_path.write_text(json.dumps(task, indent=2))

        exported.append(
            ExportedContract(
                name=contract_name,
                input_json=json_path,
                interface=interface_abs,
                stub=stub_abs,
                specs=spec_abs_paths,
            )
        )

    return exported
