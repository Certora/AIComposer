from typing import Dict, Protocol
import json
import os
import pathlib
import zlib
import anthropic

from composer.input.types import (
    UploadPaths, InputJSONPath, InputData, SpecInput, UploadedFile,
)


class _CombinedPaths(UploadPaths, InputJSONPath, Protocol):
    pass


def upload_file_if_needed(client: anthropic.Anthropic, file_path: str, uploaded_files: Dict[str, str]) -> UploadedFile:
    """Upload a file if not already uploaded, return UploadedFile."""
    with open(file_path, 'rb') as f_bytes:
        crc_hex = hex(zlib.crc32(f_bytes.read()))
    basename = os.path.basename(file_path)
    crc_basename = f"{crc_hex}_{basename}"
    if crc_basename not in uploaded_files:
        print(f"Uploading {basename}... (canonical name {crc_basename})")
        uploaded_file = client.beta.files.upload(
            file=(crc_basename, open(file_path, "rb"), "text/plain")
        )
        print(f"Uploaded {basename} with ID: {uploaded_file.id}")
        return UploadedFile(file_id=uploaded_file.id, basename=basename, path=file_path)
    else:
        print(f"Found existing {basename} with ID: {uploaded_files[crc_basename]} (canonical name {crc_basename})")
        return UploadedFile(file_id=uploaded_files[crc_basename], basename=basename, path=file_path)


def _spec_vfs_path(spec_local_path: str, source_root: str | None) -> str:
    """Compute the VFS path for a spec file.

    Greenfield (no source_root): ``certora/<basename>``.
    From-source: path relative to ``source_root``. Errors if the spec does not
    resolve to a descendant of ``source_root``.
    """
    abs_spec = pathlib.Path(spec_local_path).resolve()
    if source_root is None:
        return f"certora/{abs_spec.name}"
    abs_root = pathlib.Path(source_root).resolve()
    try:
        rel = abs_spec.relative_to(abs_root)
    except ValueError:
        raise ValueError(
            f"Spec file {spec_local_path!r} does not resolve to a descendant of "
            f"source_root {source_root!r}. In from-source mode every spec must "
            f"live inside the workspace."
        )
    return str(rel)


def _build_specs(
    client: anthropic.Anthropic,
    uploaded: Dict[str, str],
    spec_local_paths: list[str],
    source_root: str | None,
) -> list[SpecInput]:
    """Upload each spec, compute its VFS path, error on duplicate VFS paths."""
    specs: list[SpecInput] = []
    seen: dict[str, str] = {}
    for local in spec_local_paths:
        vfs_path = _spec_vfs_path(local, source_root)
        if vfs_path in seen:
            raise ValueError(
                f"Spec VFS path collision: {vfs_path!r} would be written by both "
                f"{seen[vfs_path]!r} and {local!r}. Rename one of the files or "
                f"use a source_root layout that gives them distinct paths."
            )
        seen[vfs_path] = local
        uploaded_file = upload_file_if_needed(client, local, uploaded)
        specs.append(SpecInput(file=uploaded_file, vfs_path=vfs_path))
    return specs


def _upload_from_json(
    client: anthropic.Anthropic,
    uploaded: Dict[str, str],
    json_path: str,
) -> InputData:
    """Load a JSON task descriptor and upload its referenced files.

    Relative paths (specs, interface, system_doc, implementation_path) resolve
    against the JSON file's directory. ``source_root`` is absolute or relative
    to the JSON file's directory. ``prover_conf`` is passed through as-is.
    """
    json_abs = pathlib.Path(json_path).resolve()
    base = json_abs.parent

    with open(json_abs, "r") as f:
        data = json.load(f)

    def _resolve(p: str) -> str:
        q = pathlib.Path(p)
        if not q.is_absolute():
            q = base / q
        return str(q.resolve())

    spec_paths_raw = data.get("specs")
    if not isinstance(spec_paths_raw, list) or not spec_paths_raw:
        raise ValueError(
            f"{json_path}: `specs` must be a non-empty list of paths."
        )
    spec_paths = [_resolve(p) for p in spec_paths_raw]

    intf_raw = data.get("interface")
    if not isinstance(intf_raw, str):
        raise ValueError(f"{json_path}: `interface` (string path) is required.")
    intf_path = _resolve(intf_raw)

    sysdoc_raw = data.get("system_doc")
    if not isinstance(sysdoc_raw, str):
        raise ValueError(f"{json_path}: `system_doc` (string path) is required.")
    sysdoc_path = _resolve(sysdoc_raw)

    source_root_raw = data.get("source_root")
    source_root: str | None = _resolve(source_root_raw) if source_root_raw else None

    interface_file = upload_file_if_needed(client, intf_path, uploaded)
    system_doc_file = upload_file_if_needed(client, sysdoc_path, uploaded)
    specs = _build_specs(client, uploaded, spec_paths, source_root)

    return InputData(
        specs=specs,
        system_doc=system_doc_file,
        intf=interface_file,
        source_root=source_root,
        contract_name=data.get("name"),
        implementation_path=data.get("implementation_path"),
        prover_conf=data.get("prover_conf"),
    )


def _upload_from_triad(
    client: anthropic.Anthropic,
    uploaded: Dict[str, str],
    i: UploadPaths,
    prover_conf_path: str | None,
) -> InputData:
    """Legacy single-spec path. Loads --prover-conf from its JSON file, if set."""
    assert i.spec_file is not None
    assert i.interface_file is not None
    assert i.system_doc is not None

    interface_file = upload_file_if_needed(client, i.interface_file, uploaded)
    system_doc_file = upload_file_if_needed(client, i.system_doc, uploaded)
    source_root = getattr(i, "source_root", None)
    specs = _build_specs(client, uploaded, [i.spec_file], source_root)

    prover_conf: dict | None = None
    if prover_conf_path is not None:
        prover_conf = json.loads(pathlib.Path(prover_conf_path).read_text())

    return InputData(
        specs=specs,
        system_doc=system_doc_file,
        intf=interface_file,
        source_root=source_root,
        contract_name=getattr(i, "contract_name", None),
        implementation_path=getattr(i, "implementation_path", None),
        prover_conf=prover_conf,
    )


def upload_input(i: _CombinedPaths, prover_conf_path: str | None = None) -> InputData:
    """Resolve CLI args to a normalized ``InputData``.

    If ``i.input_json`` is set, ingest the JSON descriptor (multi-spec path).
    Otherwise fall back to the legacy positional triad (single-spec).
    Validates that exactly one of the two modes is in use.
    """
    input_json = getattr(i, "input_json", None)
    has_triad = all(
        getattr(i, field, None) is not None
        for field in ("spec_file", "interface_file", "system_doc")
    )

    if input_json is not None and any(
        getattr(i, field, None) is not None
        for field in ("spec_file", "interface_file", "system_doc")
    ):
        raise ValueError(
            "--input-json is mutually exclusive with the positional "
            "spec_file/interface_file/system_doc triad."
        )
    if input_json is None and not has_triad:
        raise ValueError(
            "Must supply either the positional spec_file/interface_file/system_doc "
            "triad OR --input-json <path>."
        )

    client = anthropic.Anthropic()
    uploaded: Dict[str, str] = {}
    for f in client.beta.files.list():
        uploaded[f.filename] = f.id

    if input_json is not None:
        return _upload_from_json(client, uploaded, input_json)
    return _upload_from_triad(client, uploaded, i, prover_conf_path)
