"""
Audit archive backed by LangGraph's ``AsyncPostgresStore``.

Replaces the bespoke PostgreSQL audit DB that used file_blobs +
run_info/vfs_* tables. All state is stored as JSONB values under a
small set of namespaces; there is no content-addressed blob store
and no gzip.

Namespaces:

    ("audit_runs",)                           / thread_id           → StoredRunMeta
    ("audit", tid)                            / "run_info"          → StoredRunInfo
    ("audit", tid)                            / "resume_artifact"   → StoredResumeArtifact
    ("audit", tid)                            / "vfs_initial"       → StoredVFS
    ("audit", tid)                            / "vfs_result"        → StoredVFS
    ("audit", tid, "summarization")           / checkpoint_id       → StoredSummary
    ("audit", tid, "prover_results", tool_id) / rule_name           → StoredProverResult
    ("audit", tid, "manual_results")          / tool_id              → StoredManualResults

The ``audit_runs`` namespace is intentionally flat so callers can list
every registered run without enumerating thread ids out-of-band — useful
for description-based lookups after a crash, where the thread id has
been lost but the human-supplied label survives.

Spec / interface / system doc contents are inlined into StoredRunInfo
(they're small, always read together with the filenames, and the plan
explicitly drops the separate blob store).

Every method is async; callers talk to the same ``AsyncPostgresStore``
the main workflow uses, so there is a single store connection for the
whole executor.

Audit-side document handles (``_StoredText``, ``_StoredBinary``,
``ResumeSpecEntry``) satisfy the ``Uploadable`` protocol — basename
+ bytes_contents + optional string_contents — and nothing more. On
resume, the executor feeds them through its ``FileUploader`` (which
knows the active provider) to materialize real ``Document`` /
``TextDocument`` instances. The audit store has no opinion about
which backend is consuming the data.
"""


import base64
import hashlib
import logging
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import AsyncIterator, Iterable, Iterator, Literal, Optional, TypedDict, cast

from langgraph.store.base import BaseStore

from composer.audit.sink import AuditSink
from composer.audit.types import (
    ManualResult,
    RuleResult,
    RunInput,
    SpecRunEntry,
)
from composer.input.files import Document, TextDocument
from composer.prover.ptypes import StatusCodes
from composer.rag.types import ManualRef


# ---------------------------------------------------------------------------
# Stored value shapes
# ---------------------------------------------------------------------------


class StoredSpecFile(TypedDict):
    vfs_path: str
    basename: str
    contents: str


class StoredSystemBinary(TypedDict):
    """On-disk shape for a binary system document. The text-or-binary
    classification happens once at ``register_run`` time (based on
    whether the source's ``string_contents`` is non-None); on read we
    just dispatch on whether ``system`` is a string or a dict, no
    heuristic re-guessing."""
    type: Literal["b64"]
    contents: str


class StoredRunInfo(TypedDict):
    spec: StoredSpecFile
    interface_name: str
    interface_contents: str
    system_name: str
    # Text body if the system doc was UTF-8 text upstream; otherwise
    # the binary variant. Schema is discriminated, not heuristic.
    system: str | StoredSystemBinary
    reqs: list[str] | None


class StoredResumeArtifact(TypedDict):
    interface_path: str
    commentary: str


class StoredVFS(TypedDict):
    files: dict[str, str]


class StoredRunMeta(TypedDict):
    """Run-lifecycle metadata distinct from the run's inputs.

    Lives in its own audit slot so additive fields (parent thread,
    completion time, run kind, etc.) can land here without bumping
    ``StoredRunInfo``'s version. Optional ``description`` is free-form
    user-supplied text — searchable so a run can be located later by
    label after a crash, even when the thread id has been lost."""
    started_at: str  # ISO 8601, UTC.
    description: str | None


class StoredSummary(TypedDict):
    summary: str


class StoredProverResult(TypedDict):
    status: str
    analysis: str | None


class StoredManualResult(TypedDict):
    similarity: float
    content: str
    header: str


class StoredManualResults(TypedDict):
    """All manual-search hits for one ``cvl_manual_search`` call,
    stored as a single record under ``(audit, tid, "manual_results")``
    keyed by ``tool_id``. (The legacy schema streamed one event per
    hit and stored each under a uuid sub-key — a workaround for
    postgres's clunky list support that no longer applies now that
    audit lives in the LangGraph store.)"""
    results: list[StoredManualResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes_digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


# ---------------------------------------------------------------------------
# In-memory file views — audit-restored ``Uploadable`` carriers.
#
# These satisfy the ``Uploadable`` protocol (basename + bytes_contents
# + optional string_contents) and nothing more. Callers feed them to
# the workflow's ``FileUploader`` to get back real ``Document`` /
# ``TextDocument`` instances (with ``to_dict`` etc.) for the active
# provider. The audit store doesn't know or care which provider is
# running — the uploader takes care of that on rehydration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StoredText:
    """Audit-restored text source. Bytes view is the UTF-8 encoding of
    ``string_contents``; satisfies ``Uploadable`` (and structurally
    satisfies the non-None-text refinement that
    ``upload_text_if_needed`` expects)."""

    path: str
    contents: str

    @property
    def basename(self) -> str:
        return pathlib.Path(self.path).name

    @property
    def bytes_contents(self) -> bytes:
        return self.contents.encode("utf-8")

    @property
    def string_contents(self) -> str:
        return self.contents


@dataclass(frozen=True)
class _StoredBinary:
    """Audit-restored binary source. The text-or-binary call was made
    at ``register_run`` time and baked into the storage schema, so
    this type is *honestly* binary — ``string_contents`` always
    returns ``None``."""

    path: str
    contents_b64: str

    @property
    def basename(self) -> str:
        return pathlib.Path(self.path).name

    @property
    def bytes_contents(self) -> bytes:
        return base64.standard_b64decode(self.contents_b64)

    @property
    def string_contents(self) -> None:
        return None


# ---------------------------------------------------------------------------
# VFS retriever — iterates a flat path→content map
# ---------------------------------------------------------------------------


@dataclass
class VFSRetriever:
    _files: dict[str, str]

    def to_dict(self) -> dict[str, bytes]:
        return {p: c.encode("utf-8") for (p, c) in self._files.items()}

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        for p, c in self._files.items():
            yield (p, c.encode("utf-8"))

    def get_file(self, p: str) -> _StoredText | None:
        c = self._files.get(p)
        if c is None:
            return None
        return _StoredText(path=p, contents=c)

    def __getitem__(self, p: str) -> _StoredText | None:
        return self.get_file(p)


# ---------------------------------------------------------------------------
# Resume artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumeSpecEntry:
    """A single spec file as captured on the completed run. ``vfs_path``
    is what the executor needs to re-overlay the spec into the resumed
    state; ``contents`` is what was there at completion time (before
    any resume-time updates).

    Satisfies ``TextUploadable`` — feed to ``FileUploader.upload_text_if_needed``
    to materialize an ``UploadedTextFile`` for the active provider."""
    vfs_path: str
    basename: str
    contents: str

    @property
    def string_contents(self) -> str:
        return self.contents

    @property
    def bytes_contents(self) -> bytes:
        return self.contents.encode("utf-8")


class ResumeArtifact:
    """Bundle of everything needed to resume a prior run: the final
    interface and system views, the spec file that was in play at
    completion, the full final VFS, and the commentary the LLM
    attached on completion.

    All file-shaped fields satisfy ``Uploadable``; the executor passes
    them through its ``FileUploader`` on resume to rehydrate into
    real ``Document`` / ``TextDocument`` instances for the active
    provider."""

    def __init__(
        self,
        final_intf: _StoredText,
        spec_entry: ResumeSpecEntry,
        system_doc: "_StoredText | _StoredBinary",
        commentary: str,
        intf_path: str,
        vfs_cur: VFSRetriever,
    ):
        self.intf_vfs_handle = final_intf
        self.spec = spec_entry
        self.system_vfs_handle: "_StoredText | _StoredBinary" = system_doc
        self.vfs = vfs_cur
        self.commentary = commentary
        self.interface_path = intf_path

    @cached_property
    def interface_file(self) -> str:
        return self.intf_vfs_handle.string_contents

    @cached_property
    def system_doc(self) -> str | None:
        """Text body of the original system doc, or ``None`` if it was
        a binary input (e.g. PDF). Determined by the discriminator
        baked into the stored shape, not a runtime heuristic."""
        return self.system_vfs_handle.string_contents


# ---------------------------------------------------------------------------
# AuditStore
# ---------------------------------------------------------------------------


def _safe_dict(x: StoredProverResult |
               StoredRunInfo |
               StoredResumeArtifact |
               StoredRunMeta |
               StoredVFS |
               StoredSummary |
               StoredManualResults) -> dict:
    return {**x}


def _decode_text_files(items: Iterable[tuple[str, bytes]]) -> dict[str, str]:
    """Decode VFS bytes as UTF-8, dropping anything binary.

    The audit store's ``StoredVFS`` is a flat ``{path: str}`` dict — it
    has no place for binary blobs. Sources routinely contain binary
    files (images, pdf, git blobs); skip them with a log line and keep
    going. Resume from this snapshot won't restore those binaries, but
    they weren't going to round-trip through a JSONB string column
    anyway."""
    logger = logging.getLogger(__name__)
    out: dict[str, str] = {}
    skipped: list[str] = []
    for path, content in items:
        try:
            out[path] = content.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(path)
    if skipped:
        logger.info(
            "Skipped %d binary file(s) when persisting VFS: %s",
            len(skipped),
            ", ".join(skipped[:10]) + (f" (+{len(skipped) - 10} more)" if len(skipped) > 10 else ""),
        )
    return out


_RUN_INFO_KEY = "run_info"
_RUN_META_NS: tuple[str, ...] = ("audit_runs",)
_VFS_INITIAL_KEY = "vfs_initial"
_VFS_RESULTS_KEY = "vfs_result"
_RESUME_ARTIFACT_KEY = "resume_artifact"


def _system_handle(system_name: str, stored_system: str | StoredSystemBinary) -> "_StoredText | _StoredBinary":
    """Dispatch the discriminated ``system`` slot to the right
    ``Uploadable`` carrier. No heuristic — the shape decides."""
    if isinstance(stored_system, str):
        return _StoredText(path=system_name, contents=stored_system)
    return _StoredBinary(
        path=system_name, contents_b64=stored_system["contents"]
    )


class AuditStore:
    """Async accessor for the audit archive.

    Wraps a ``BaseStore`` (typically ``AsyncPostgresStore``). All
    reads and writes use the ``a*`` (async) store methods, so the
    surrounding code must be async too."""

    def __init__(self, store: BaseStore):
        self._store = store

    # -- namespace helpers -------------------------------------------------

    @staticmethod
    def _ns(thread_id: str, *extra: str) -> tuple[str, ...]:
        return ("audit", thread_id, *extra)

    # -- run registration --------------------------------------------------

    async def register_run(
        self,
        thread_id: str,
        spec_vfs_path: str,
        spec_file: TextDocument,
        interface_file: TextDocument,
        system_doc: Document,
        vfs_init: Iterable[tuple[str, bytes]],
        reqs: list[str] | None,
        description: str | None = None,
    ) -> None:
        """``spec_vfs_path`` is where the spec lives in the VFS (codegen's
        historical convention is ``rules.spec``).

        Spec/interface contents persist as plain strings (text-guaranteed
        upstream). The system doc is classified once here: if its
        ``string_contents`` is non-None it lands as a plain string,
        otherwise it lands as a ``{"type": "b64", "contents": ...}``
        binary record. No heuristic on the read side.

        ``description`` is free-form user-supplied text recorded on the
        ``run_meta`` slot so callers can find a run by name later,
        after the thread id has been lost."""
        stored_spec: StoredSpecFile = {
            "vfs_path": spec_vfs_path,
            "basename": spec_file.basename,
            "contents": spec_file.string_contents,
        }
        system_text = system_doc.string_contents
        stored_system: str | StoredSystemBinary
        if system_text is not None:
            stored_system = system_text
        else:
            stored_system = {
                "type": "b64",
                "contents": base64.standard_b64encode(system_doc.bytes_contents).decode("utf-8"),
            }
        run_info: StoredRunInfo = {
            "spec": stored_spec,
            "interface_name": interface_file.basename,
            "interface_contents": interface_file.string_contents,
            "system_name": system_doc.basename,
            "system": stored_system,
            "reqs": reqs,
        }
        vfs_payload: StoredVFS = {
            "files": _decode_text_files(vfs_init),
        }
        run_meta: StoredRunMeta = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
        }

        await self._store.aput(self._ns(thread_id), _RUN_INFO_KEY, _safe_dict(run_info))
        await self._store.aput(_RUN_META_NS, thread_id, _safe_dict(run_meta))
        await self._store.aput(self._ns(thread_id), _VFS_INITIAL_KEY, _safe_dict(vfs_payload))

    async def register_complete(
        self,
        thread_id: str,
        vfs: Iterable[tuple[str, bytes]],
        intf: str,
        commentary: str,
    ) -> None:
        vfs_payload: StoredVFS = {
            "files": _decode_text_files(vfs),
        }
        await self._store.aput(self._ns(thread_id), _VFS_RESULTS_KEY, _safe_dict(vfs_payload))

        resume: StoredResumeArtifact = {
            "interface_path": intf,
            "commentary": commentary,
        }
        await self._store.aput(self._ns(thread_id), _RESUME_ARTIFACT_KEY, _safe_dict(resume))

    # -- reads -------------------------------------------------------------

    async def get_resume_artifact(self, thread_id: str) -> ResumeArtifact:
        ra_item = await self._store.aget(self._ns(thread_id), _RESUME_ARTIFACT_KEY)
        if ra_item is None:
            raise RuntimeError(f"No resume artifact found for thread {thread_id}")
        ra = cast(StoredResumeArtifact, ra_item.value)

        ri_item = await self._store.aget(self._ns(thread_id), _RUN_INFO_KEY)
        if ri_item is None:
            raise RuntimeError(f"No run info found for thread {thread_id}")
        ri = cast(StoredRunInfo, ri_item.value)

        vfs_item = await self._store.aget(self._ns(thread_id), _VFS_RESULTS_KEY)
        if vfs_item is None:
            raise RuntimeError(f"No vfs_result found for thread {thread_id}")
        vfs_files = cast(StoredVFS, vfs_item.value)["files"]

        intf_contents = vfs_files.get(ra["interface_path"])
        if intf_contents is None:
            raise RuntimeError(
                f"Resume artifact references {ra['interface_path']} "
                f"but it's not in vfs_result"
            )

        # Pull the registered spec's *final* contents from the completed
        # VFS. Missing is a hard error — it was present at registration
        # time and we expect it at the completed VFS too.
        stored_spec = ri["spec"]
        spec_vfs_path = stored_spec["vfs_path"]
        final_spec_contents = vfs_files.get(spec_vfs_path)
        if final_spec_contents is None:
            raise RuntimeError(
                f"vfs_result for thread {thread_id} has no file at {spec_vfs_path!r} "
                f"(registered as the spec at run start)"
            )
        spec_entry = ResumeSpecEntry(
            vfs_path=spec_vfs_path,
            basename=stored_spec["basename"],
            contents=final_spec_contents,
        )

        return ResumeArtifact(
            final_intf=_StoredText(
                path=ra["interface_path"],
                contents=intf_contents,
            ),
            spec_entry=spec_entry,
            system_doc=_system_handle(ri["system_name"], ri["system"]),
            commentary=ra["commentary"],
            intf_path=ra["interface_path"],
            vfs_cur=VFSRetriever(_files=vfs_files),
        )

    async def get_run_info(self, thread_id: str) -> tuple[RunInput, VFSRetriever]:
        ri_item = await self._store.aget(self._ns(thread_id), _RUN_INFO_KEY)
        if ri_item is None:
            raise RuntimeError(f"Didn't find run info for {thread_id}")
        ri = cast(StoredRunInfo, ri_item.value)

        vfs_item = await self._store.aget(self._ns(thread_id), _VFS_INITIAL_KEY)
        vfs_files: dict[str, str] = {}
        if vfs_item is not None:
            vfs_files = cast(StoredVFS, vfs_item.value)["files"]
        retriever = VFSRetriever(_files=vfs_files)

        stored_spec = ri["spec"]
        run_spec: SpecRunEntry = {
            "vfs_path": stored_spec["vfs_path"],
            "basename": stored_spec["basename"],
            "contents": stored_spec["contents"],
        }
        run_input: RunInput = {
            "interface": _StoredText(
                path=ri["interface_name"],
                contents=ri["interface_contents"],
            ),
            "spec": run_spec,
            "system": _system_handle(ri["system_name"], ri["system"]),
            "reqs": ri["reqs"],
        }
        return (run_input, retriever)

    async def get_run_meta(self, thread_id: str) -> StoredRunMeta | None:
        """Return run-lifecycle metadata for ``thread_id``, or ``None``
        for runs registered before the meta slot existed."""
        item = await self._store.aget(_RUN_META_NS, thread_id)
        if item is None:
            return None
        return cast(StoredRunMeta, item.value)

    async def list_run_meta(self, limit: int = 1000) -> list[tuple[str, StoredRunMeta]]:
        """Return ``(thread_id, meta)`` for every registered run,
        newest first.

        Cheap cross-run query backed by the flat ``audit_runs``
        namespace — useful for crash-recovery lookups by description.
        Bring the data home and filter in Python; the namespace is
        small."""
        items = await self._store.asearch(_RUN_META_NS, limit=limit)
        out = [(item.key, cast(StoredRunMeta, item.value)) for item in items]
        out.sort(key=lambda pair: pair[1].get("started_at", ""), reverse=True)
        return out

    # -- prover + manual-search stream sinks -------------------------------

    async def add_rule_result(
        self,
        thread_id: str,
        tool_id: str,
        rule_name: str,
        result: str,
        analysis: str | None,
    ) -> None:
        payload: StoredProverResult = {"status": result, "analysis": analysis}
        await self._store.aput(
            self._ns(thread_id, "prover_results", tool_id), rule_name, _safe_dict(payload)
        )

    async def get_rule_results(
        self, thread_id: str, tool_id: str
    ) -> AsyncIterator[RuleResult]:
        # asearch defaults to limit=10; audit queries want everything.
        items = await self._store.asearch(
            self._ns(thread_id, "prover_results", tool_id), limit=10_000
        )
        for item in items:
            payload = cast(StoredProverResult, item.value)
            yield RuleResult(
                rule=item.key,
                status=payload["status"],
                analysis=payload["analysis"],  # type: ignore[typeddict-item]
            )

    async def add_manual_results(
        self, thread_id: str, tool_id: str, refs: list[ManualRef]
    ) -> None:
        """Store *all* hits from a single ``cvl_manual_search`` call as
        one record under ``(audit, tid, "manual_results") / tool_id``."""
        payload: StoredManualResults = {
            "results": [
                {
                    "similarity": ref.similarity,
                    "content": ref.content,
                    "header": " / ".join(ref.headers),
                }
                for ref in refs
            ],
        }
        await self._store.aput(
            self._ns(thread_id, "manual_results"),
            tool_id,
            _safe_dict(payload),
        )

    async def get_manual_results(
        self, thread_id: str, tool_id: str
    ) -> list[ManualResult]:
        item = await self._store.aget(
            self._ns(thread_id, "manual_results"), tool_id
        )
        if item is None:
            return []
        payload = cast(StoredManualResults, item.value)
        return [
            ManualResult(
                content=r["content"],
                header=r["header"],
                similarity=r["similarity"],
            )
            for r in payload["results"]
        ]

    # -- summarization -----------------------------------------------------

    async def register_summary(
        self, thread_id: str, checkpoint_id: str, summary: str
    ) -> None:
        payload: StoredSummary = {"summary": summary}
        await self._store.aput(
            self._ns(thread_id, "summarization"), checkpoint_id, _safe_dict(payload)
        )

    async def get_summary_after_checkpoint(
        self, thread_id: str, checkpoint_id: str
    ) -> str | None:
        item = await self._store.aget(
            self._ns(thread_id, "summarization"), checkpoint_id
        )
        if item is None:
            return None
        return cast(StoredSummary, item.value)["summary"]


# ---------------------------------------------------------------------------
# AuditStoreSink — async AuditSink bound to a fixed thread_id
# ---------------------------------------------------------------------------


class AuditStoreSink:
    """Async sink that records prover / manual-search / summarization
    events to an ``AuditStore`` under a fixed ``thread_id``. Satisfies
    the async version of the ``AuditSink`` protocol — gets wired into
    ``CodeGenEventHandler`` once the executor is modernized."""

    def __init__(self, audit: AuditStore, thread_id: str):
        self._audit = audit
        self._thread_id = thread_id

    async def on_rule_result(
        self,
        rule: str,
        status: StatusCodes,
        analysis: str | None,
        tool_id: str,
    ) -> None:
        await self._audit.add_rule_result(
            thread_id=self._thread_id,
            tool_id=tool_id,
            rule_name=rule,
            result=status,
            analysis=analysis,
        )

    async def on_manual_search(
        self, tool_id: str, refs: list[ManualRef]
    ) -> None:
        await self._audit.add_manual_results(
            thread_id=self._thread_id, tool_id=tool_id, refs=refs,
        )

    async def on_summarization(self, checkpoint_id: str, summary: str) -> None:
        await self._audit.register_summary(
            thread_id=self._thread_id,
            checkpoint_id=checkpoint_id,
            summary=summary,
        )
