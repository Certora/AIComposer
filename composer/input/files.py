"""Files-API-backed file shapes + uploader + Document protocols.

Public surface:

- ``Document`` / ``TextDocument`` — what downstream consumers depend on.
- ``FileUploader`` — owns the async Anthropic client + per-account dedup
  cache. Construct via :meth:`FileUploader.fresh`.
- ``InMemoryTextFile``, ``UploadedFile``, ``UploadedTextFile`` —
  concrete shapes. Implementation details: callers should declare
  protocol-typed parameters and not branch on these.

Policy (current): only binary files go through the Files API. Text
files stay inline as ``InMemoryTextFile`` so they remain visible in the
prompt when a conversation is later debugged. Very-large text files
that should still be uploaded can be loaded explicitly via
:meth:`FileUploader.upload_text_file_if_needed`.
"""


import asyncio
import hashlib
import mimetypes
import os
import pathlib
import zlib
from dataclasses import dataclass, field
from typing import Protocol, Any

import anthropic

from composer.audit.types import InputFileLike


# ---------------------------------------------------------------------------
# Protocols (the public surface)
# ---------------------------------------------------------------------------


class Document(Protocol):
    """A piece of content destined for an LLM message."""

    @property
    def basename(self) -> str: ...
    @property
    def string_contents(self) -> str | None: ...
    @property
    def bytes_contents(self) -> bytes: ...
    def to_dict(self, with_cache: bool = False) -> dict: ...
    def to_digest(self) -> str: ...

    def to_file_like(self) -> InputFileLike:
        ...


class TextDocument(Document, Protocol):
    """Refinement of ``Document`` whose body is guaranteed text."""

    @property
    def string_contents(self) -> str: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# Suffixes treated as binary regardless of byte content; short-circuits
# the byte-scan heuristic for common cases.
_KNOWN_BINARY_SUFFIXES = {".pdf"}

_KNOWN_TEXT_SUFFIXES = {".md", ".txt", ".sol", ".spec", ".conf"}

# Bytes scanned by the binary heuristic. Standard git/grep-I trick: a
# NUL byte in the first 8 KiB means binary.
_BINARY_SNIFF_BYTES = 8 * 1024


async def _upload_mime(path: str) -> str:
    """Pick the MIME type sent to the Files API.

    Anthropic stores whatever content-type we declare; the *consumer*
    side (document blocks) decodes by that. Tagging a PDF as
    ``text/plain`` makes the eventual document block return
    ``Invalid encoding for plaintext file`` because the API tries to
    UTF-8-decode the bytes. ``mimetypes.guess_type`` first, then the
    binary heuristic for unknown suffixes."""
    guessed, _ = mimetypes.guess_type(path)
    if guessed is not None:
        if guessed.startswith("text/"):
            return "text/plain"
        return guessed
    return "application/octet-stream" if await _is_binary_file(path) else "text/plain"


async def _is_binary_file(path: str) -> bool:
    """True if ``path`` should be treated as binary at upload time."""
    suffix = pathlib.Path(path).suffix.lower()
    if suffix in _KNOWN_BINARY_SUFFIXES:
        return True
    elif suffix in _KNOWN_TEXT_SUFFIXES:
        return False

    def _scan() -> bytes:
        with open(path, "rb") as f:
            return f.read(_BINARY_SNIFF_BYTES)

    chunk = await asyncio.to_thread(_scan)
    return b"\x00" in chunk


# ---------------------------------------------------------------------------
# Concrete shapes (implementation details — declare protocol types instead)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InMemoryTextFile:
    """Text content carried inline in the request. Produced by the
    default text-file path so the content stays visible in conversation
    transcripts."""

    basename: str
    string_contents: str

    @property
    def bytes_contents(self) -> bytes:
        return self.string_contents.encode("utf-8")

    def to_dict(self, with_cache: bool = False) -> dict:
        to_ret : dict[str, Any] = {"type": "text", "text": self.string_contents}
        if with_cache:
            to_ret["cache_control"] = {
                "type": "ephemeral",
                "ttl": "5m"
            }
        return to_ret

    def to_digest(self) -> str:
        return _bytes_digest(self.bytes_contents)
    
    def to_file_like(self) -> InputFileLike:
        return self

@dataclass(frozen=True)
class _IFWrapper:
    _wrapped: "UploadedFile"

    @property
    def basename(self) -> str:
        return self._wrapped.basename
    
    @property
    def bytes_contents(self) -> bytes:
        return self._wrapped.bytes_contents
    
    @property
    def string_contents(self) -> str:
        r = self._wrapped.string_contents
        assert r is not None
        return r

@dataclass(frozen=True)
class UploadedFile:
    """A (potentially-binary) file uploaded to the Files API. Bytes are
    cached in memory so ``bytes_contents`` / ``string_contents`` don't
    re-read from disk and survive whatever the local filesystem looks
    like later."""

    file_id: str
    basename: str
    contents: bytes
    digest: str

    def to_dict(self, with_cache: bool = False) -> dict:
        to_ret : dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "file",
                "file_id": self.file_id,
            },
        }
        if with_cache:
            to_ret["cache_control"] = {
                "type": "ephemeral",
                "ttl": "5m"
            }
        return to_ret

    def to_digest(self) -> str:
        return self.digest

    @property
    def string_contents(self) -> str | None:
        try:
            return self.contents.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @property
    def bytes_contents(self) -> bytes:
        return self.contents
    
    def to_file_like(self) -> InputFileLike:
        return _IFWrapper(self)


@dataclass(frozen=True)
class UploadedTextFile(UploadedFile):
    """A Files-API upload that was classified as text at upload time.
    ``string_contents`` is guaranteed non-None. Produced by
    :meth:`FileUploader.upload_text_file_if_needed` for the
    very-large-text case where inlining the body would blow the prompt
    budget."""

    @property
    def string_contents(self) -> str:
        return self.contents.decode("utf-8")


# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------


@dataclass
class FileUploader:
    """Bundles the async Anthropic client with the cache of
    already-uploaded files (indexed by canonical CRC-prefixed
    filename) so callers pass a single handle through the upload
    pipeline instead of threading ``(client, uploaded_files)`` pairs.

    Construct via :meth:`fresh`; it creates an async client and seeds
    the cache from the live Files API listing so duplicate uploads are
    skipped on subsequent calls."""

    client: anthropic.AsyncAnthropic
    uploaded: dict[str, str] = field(default_factory=dict)

    @staticmethod
    async def fresh() -> "FileUploader":
        """Build a fresh ``FileUploader`` with the cache seeded from the
        account's existing Files-API uploads."""
        client = anthropic.AsyncAnthropic()
        uploaded: dict[str, str] = {}
        async for f in await client.beta.files.list():
            uploaded[f.filename] = f.id
        return FileUploader(client=client, uploaded=uploaded)

    async def _upload_raw(
        self, file_path: str | pathlib.Path
    ) -> tuple[str, str, bytes, str]:
        """Upload-or-reuse and return ``(file_id, basename, raw_bytes,
        digest)``. File read + CRC happens on a thread; the upload
        itself awaits on the async client."""
        if isinstance(file_path, pathlib.Path):
            file_path = str(file_path)
        basename = os.path.basename(file_path)

        def _read_and_crc() -> tuple[bytes, str]:
            with open(file_path, "rb") as f:
                data = f.read()
            return data, hex(zlib.crc32(data))

        raw, crc_hex = await asyncio.to_thread(_read_and_crc)
        digest = _bytes_digest(raw)
        crc_basename = f"{crc_hex}_{basename}"
        if crc_basename not in self.uploaded:
            mime = await _upload_mime(file_path)
            uploaded_file = await self.client.beta.files.upload(
                file=(crc_basename, open(file_path, "rb"), mime)
            )
            self.uploaded[crc_basename] = uploaded_file.id
            return uploaded_file.id, basename, raw, digest
        return self.uploaded[crc_basename], basename, raw, digest

    async def upload_file_if_needed(
        self, file_path: str | pathlib.Path
    ) -> UploadedFile:
        """Upload ``file_path`` (or reuse cached upload). Intended for
        binary inputs — callers that know they have text should prefer
        :meth:`get_document` (default text-inline) or
        :meth:`upload_text_file_if_needed` (explicit upload of text)."""
        file_id, basename, raw, digest = await self._upload_raw(file_path)
        return UploadedFile(
            file_id=file_id, basename=basename, contents=raw, digest=digest
        )

    async def upload_text_file_if_needed(
        self, file_path: str | pathlib.Path
    ) -> UploadedTextFile:
        """Upload ``file_path`` and tag the result as text. Use for
        very-large text inputs that would otherwise blow the prompt
        budget if inlined; ordinary text should go through
        :meth:`get_document`, which keeps the content in-prompt for
        transcript debuggability."""
        file_id, basename, raw, digest = await self._upload_raw(file_path)
        return UploadedTextFile(
            file_id=file_id, basename=basename, contents=raw, digest=digest
        )

    async def get_document(
        self, path: str | pathlib.Path
    ) -> Document | None:
        """Load a document from disk, picking a representation by the
        binary-vs-text heuristic.

        - Text files (no NUL bytes in the first 8 KiB, and no known
          binary suffix) become ``InMemoryTextFile`` so the content
          stays visible in the prompt for transcript debuggability.
        - Binary files go through the Files API as ``UploadedFile``.

        Returns ``None`` if ``path`` doesn't point at a regular file."""
        p = pathlib.Path(path) if isinstance(path, str) else path
        if not p.is_file():
            return None
        if await _is_binary_file(str(p)):
            return await self.upload_file_if_needed(p)
        text = await asyncio.to_thread(p.read_text)
        return InMemoryTextFile(basename=p.name, string_contents=text)
