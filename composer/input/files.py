import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol
import json
import mimetypes
import os
import pathlib
import zlib
import anthropic

from composer.input.models import CodegenConfiguration, CmdlineCodegenConfiguration

from composer.input.types import (
    UploadPaths, InputJSONPath, InputData, SpecInput,
)


class _CombinedPaths(UploadPaths, InputJSONPath, Protocol):
    pass


# ---------------------------------------------------------------------------
# Concrete Files-API-backed file shapes
# ---------------------------------------------------------------------------


@dataclass
class UploadedFile:
    """A file uploaded to the Files API. The general (potentially-binary)
    shape — satisfies ``InputFileLike``. ``string_contents`` always
    returns ``None`` here; text uploads land as ``UploadedTextFile``
    instead, where the type system records the text guarantee.

    ``to_document_dict`` is always a Files-API reference; the LLM
    handles text and binary content identically from a ``file_id``."""

    file_id: str
    basename: str
    path: str

    def to_document_dict(self) -> dict:
        return {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": self.file_id
            }
        }

    @property
    def string_contents(self) -> Optional[str]:
        try:
            pathlib.Path(self.path).read_text()
        except UnicodeDecodeError:
            return None

    @property
    def bytes_contents(self) -> bytes:
        with open(self.path, 'rb') as f:
            return f.read()


@dataclass
class UploadedTextFile(UploadedFile):
    """A Files-API upload that was classified as text at upload time.
    Satisfies the stronger ``TextInputFile`` refinement —
    ``string_contents`` is non-None."""

    @property
    def string_contents(self) -> str:  # type: ignore[override]
        with open(self.path, 'r') as f:
            return f.read()


# Suffixes that we always treat as binary regardless of byte content;
# saves a sniff pass for the common PDF case. Anything not in this set
# falls through to the byte-scan heuristic.
_KNOWN_BINARY_SUFFIXES = {".pdf"}

# Bytes scanned by the binary heuristic. The git/grep-I trick: if any
# NUL byte appears in the first 8 KiB the file is binary.
_BINARY_SNIFF_BYTES = 8 * 1024


async def _upload_mime(path: str) -> str:
    """Pick a MIME type to send to the Files API for ``path``.

    Why this matters: the Files API stores whatever content-type we
    declare, and the *consumer* side (document content blocks)
    decodes accordingly. Tagging a PDF as ``text/plain`` makes the
    API reject the eventual ``document`` source with
    ``Invalid encoding for plaintext file`` because it dutifully
    tries to UTF-8-decode the PDF bytes.

    Use ``mimetypes.guess_type`` first (catches ``.pdf``,
    ``.json``, ``.png``, etc.); fall back to the binary heuristic to
    distinguish "text we don't have a mime for" (specs, ``.sol``,
    ``.cvl``) from "binary we don't have a mime for" (very rare —
    arbitrary blob)."""
    guessed, _ = mimetypes.guess_type(path)
    if guessed is not None:
        return guessed
    return "application/octet-stream" if await _is_binary_file(path) else "text/plain"


async def _is_binary_file(path: str) -> bool:
    """True if ``path`` should be treated as binary at upload time.

    Short-circuits on known binary suffixes (``.pdf``). Otherwise scans
    the first 8 KiB for a NUL byte, the same heuristic ``git`` and
    ``grep -I`` use for the binary/text classification.

    Disk I/O for the byte-scan runs on a thread so the calling event
    loop isn't held up on a slow filesystem."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in _KNOWN_BINARY_SUFFIXES:
        return True

    def _scan() -> bytes:
        with open(path, "rb") as f:
            return f.read(_BINARY_SNIFF_BYTES)

    chunk = await asyncio.to_thread(_scan)
    return b"\x00" in chunk


@dataclass
class FileUploader:
    """Bundles the Anthropic client with the cache of already-uploaded
    files (indexed by canonical CRC-prefixed filename) so callers pass
    a single handle through the upload pipeline instead of threading
    the ``(client, uploaded_files)`` pair through every helper.

    Construct via :meth:`fresh`; that static getter creates an async
    Anthropic client and seeds the cache from the live Files API
    listing so duplicate uploads are skipped on subsequent calls.

    Two upload entry points (both async — the network I/O can take a
    beat, so callers running inside an event loop don't block the UI):

    - :meth:`upload_text_file_if_needed` — caller asserts the file is
      text. Raises ``ValueError`` if the binary heuristic disagrees.
      Returns ``UploadedTextFile`` whose ``string_contents`` is
      guaranteed non-None.
    - :meth:`upload_file_if_needed` — general path. Detects text vs
      binary at upload time and returns ``UploadedTextFile`` or
      ``UploadedFile`` accordingly. Use for inputs that may be either
      (system documents, etc.).
    """

    client: anthropic.AsyncAnthropic
    uploaded: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    async def fresh() -> "FileUploader":
        """Build a ``FileUploader`` with a new async Anthropic client and
        the cache pre-populated from the account's existing uploaded
        files. Network call to list — kept off the event loop via the
        async client."""
        client = anthropic.AsyncAnthropic()
        uploaded: Dict[str, str] = {}
        async for f in await client.beta.files.list():
            uploaded[f.filename] = f.id
        return FileUploader(client=client, uploaded=uploaded)

    async def _upload_raw(
        self, file_path: str | pathlib.Path
    ) -> tuple[str, str, str]:
        """Upload-or-reuse and return ``(file_id, basename, abs_path)``.
        File I/O (CRC + open-for-upload) runs on a thread; the upload
        itself is awaited on the async client."""
        if isinstance(file_path, pathlib.Path):
            file_path = str(file_path)
        basename = os.path.basename(file_path)

        def _crc() -> str:
            with open(file_path, 'rb') as f_bytes:
                return hex(zlib.crc32(f_bytes.read()))

        crc_hex = await asyncio.to_thread(_crc)
        crc_basename = f"{crc_hex}_{basename}"
        if crc_basename not in self.uploaded:
            print(f"Uploading {basename}... (canonical name {crc_basename})")
            mime = await _upload_mime(file_path)
            # ``open(...)`` here is the file-handle the SDK reads from
            # during upload. It runs sync at the OS level but the async
            # SDK drives the network I/O; the open itself is fast enough
            # to leave on the event loop.
            uploaded_file = await self.client.beta.files.upload(
                file=(crc_basename, open(file_path, "rb"), mime)
            )
            print(f"Uploaded {basename} with ID: {uploaded_file.id}")
            self.uploaded[crc_basename] = uploaded_file.id
            return uploaded_file.id, basename, file_path
        else:
            print(f"Found existing {basename} with ID: {self.uploaded[crc_basename]} (canonical name {crc_basename})")
            return self.uploaded[crc_basename], basename, file_path

    async def upload_file_if_needed(self, file_path: str | pathlib.Path) -> UploadedFile:
        """Upload ``file_path`` (or reuse cached upload). Returns an
        ``UploadedTextFile`` if the file is text per the binary
        heuristic, otherwise a plain ``UploadedFile`` (binary). The
        caller can narrow via ``isinstance`` if it wants the
        ``TextInputFile`` shape."""
        file_id, basename, abs_path = await self._upload_raw(file_path)
        if await _is_binary_file(abs_path):
            return UploadedFile(file_id=file_id, basename=basename, path=abs_path)
        return UploadedTextFile(file_id=file_id, basename=basename, path=abs_path)

    async def upload_text_file_if_needed(self, file_path: str | pathlib.Path) -> UploadedTextFile:
        """Upload ``file_path`` and assert it is text. Raises
        ``ValueError`` if the binary heuristic flags it. Use for inputs
        whose text-ness is part of the contract (specs, interfaces);
        callers that aren't sure should use ``upload_file_if_needed``."""
        result = await self.upload_file_if_needed(file_path)
        if not isinstance(result, UploadedTextFile):
            raise ValueError(
                f"Expected a text file at {file_path!r} but binary "
                f"heuristic flagged it (NUL byte in first "
                f"{_BINARY_SNIFF_BYTES} bytes, or known binary suffix). "
                f"If this file really is text, fix the heuristic; "
                f"otherwise use ``upload_file_if_needed`` for the "
                f"general path."
            )
        return result


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
    if not abs_spec.is_relative_to(abs_root):
        raise ValueError(
            f"Spec file {spec_local_path!r} does not resolve to a descendant of "
            f"source_root {source_root!r}. In from-source mode every spec must "
            f"live inside the workspace."
        )
    return str(abs_spec.relative_to(abs_root))

async def _build_specs(
    uploader: FileUploader,
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
        uploaded_file = await uploader.upload_text_file_if_needed(local)
        specs.append(SpecInput(file=uploaded_file, vfs_path=vfs_path))
    return specs


async def _upload_from_json(
    uploader: FileUploader,
    json_model: CodegenConfiguration,
    root: pathlib.Path
) -> InputData:
    """Load a JSON task descriptor and upload its referenced files.

    Relative paths (specs, interface, system_doc, implementation_path) resolve
    against the JSON file's directory. ``source_root`` is absolute or relative
    to the JSON file's directory. Prover-config overrides are NOT carried
    here — they're an orthogonal runtime concern, supplied via
    ``--prover-conf`` (CLI) or ``CommonCodeGen.prover_conf`` (assistant)
    and passed straight to the executor.
    """

    real_root = root.resolve()

    def _resolve(p: str) -> pathlib.Path:
        q = pathlib.Path(p)
        if not q.is_absolute():
            q = root / q
        return q.resolve()

    interface_file = await uploader.upload_text_file_if_needed(_resolve(json_model.interface_file))
    system_doc_file = await uploader.upload_file_if_needed(_resolve(json_model.system_doc))

    spec_files : list[SpecInput] = []

    for p in json_model.spec_files:
        r_path = _resolve(p)
        if not r_path.is_relative_to(real_root):
            raise ValueError(f"Input spec file not in source root: {real_root}")
        spec_files.append(SpecInput(
            file=await uploader.upload_text_file_if_needed(r_path),
            vfs_path=str(r_path.relative_to(real_root))
        ))

    return InputData(
        specs=spec_files,
        system_doc=system_doc_file,
        intf=interface_file,
        source_root=str(root),
        contract_name=json_model.contract_name,
        implementation_path=json_model.implementation_path,
        kickstart_context=json_model.kickstart_context
    )


async def _upload_from_triad(
    uploader: FileUploader,
    i: UploadPaths,
    *,
    spec_file: str,
    interface_file_path: str,
    system_doc: str
) -> InputData:
    """Legacy single-spec path."""

    interface_file = await uploader.upload_text_file_if_needed(interface_file_path)
    system_doc_file = await uploader.upload_file_if_needed(system_doc)
    source_root = i.source_root
    specs = await _build_specs(uploader, [spec_file], source_root)

    return InputData(
        specs=specs,
        system_doc=system_doc_file,
        intf=interface_file,
        source_root=source_root,
        contract_name=i.contract_name,
        implementation_path=i.implementation_path,
        kickstart_context=None
    )

async def upload_configuration(
    conf: CodegenConfiguration,
    root: pathlib.Path
) -> InputData:
    return await _upload_from_json(
        await FileUploader.fresh(),
        json_model=conf,
        root=root
    )

async def upload_input(i: _CombinedPaths) -> InputData:
    """Resolve CLI args to a normalized ``InputData``.

    If ``i.input_json`` is set, ingest the JSON descriptor (multi-spec path).
    Otherwise fall back to the legacy positional triad (single-spec).
    Validates that exactly one of the two modes is in use.
    Prover-config overrides flow separately via ``i.prover_conf`` (already
    a dict, parsed at argparse time) — they don't ride along on
    ``InputData``.
    """
    if i.input_json is not None:
        assert all([
            x is None for x in [
                i.spec_file, i.system_doc, i.interface_file
            ]
        ])
    else:
        assert i.input_json is None

    uploader = await FileUploader.fresh()

    if i.input_json is not None:
        explicit_root = i.source_root
        read_conf = pathlib.Path(i.input_json).read_text()
        if explicit_root is None:
            mod = CmdlineCodegenConfiguration.model_validate_json(
                read_conf
            )
            source_root = pathlib.Path(mod.source_root)
        else:
            source_root = pathlib.Path(explicit_root)
            mod = CodegenConfiguration.model_validate_json(read_conf)
        return await _upload_from_json(
            root=source_root,
            uploader=uploader,
            json_model=mod
        )

    assert i.spec_file is not None
    assert i.interface_file is not None
    assert i.system_doc is not None

    return await _upload_from_triad(
        uploader,
        i,
        system_doc=i.system_doc,
        interface_file_path=i.interface_file,
        spec_file=i.spec_file
    )
