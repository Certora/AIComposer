"""
Audit archive backed by LangGraph's ``AsyncPostgresStore``.

Replaces the bespoke PostgreSQL audit DB that used file_blobs +
run_info/vfs_* tables. All state is stored as JSONB values under a
small set of namespaces; there is no content-addressed blob store
and no gzip.

Namespaces (all rooted at ``("audit", thread_id[, ...])``):

    ("audit", tid)                            / "run_info"          → StoredRunInfo
    ("audit", tid)                            / "requirements"      → StoredRequirements
    ("audit", tid)                            / "resume_artifact"   → StoredResumeArtifact
    ("audit", tid)                            / "vfs_initial"       → StoredVFS
    ("audit", tid)                            / "vfs_result"        → StoredVFS
    ("audit", tid, "summarization")           / checkpoint_id       → StoredSummary
    ("audit", tid, "prover_results", tool_id) / rule_name           → StoredProverResult
    ("audit", tid, "manual_results", tool_id) / uuid_hex            → StoredManualResult

Spec / interface / system doc contents are inlined into StoredRunInfo
(they're small, always read together with the filenames, and the plan
explicitly drops the separate blob store).

Every method is async; callers talk to the same ``AsyncPostgresStore``
the main workflow uses, so there is a single store connection for the
whole executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import AsyncIterator, Iterable, Iterator, TypedDict, cast
import pathlib
import uuid

from langgraph.store.base import BaseStore

from composer.audit.types import (
    InputFileLike,
    ManualResult,
    RuleResult,
    RunInput,
)
from composer.audit.sink import AuditSink
from composer.prover.ptypes import StatusCodes
from composer.rag.types import ManualRef


# ---------------------------------------------------------------------------
# Stored value shapes
# ---------------------------------------------------------------------------


class StoredRunInfo(TypedDict):
    spec_name: str
    spec_contents: str
    interface_name: str
    interface_contents: str
    system_name: str
    system_contents: str
    num_reqs: int | None


class StoredRequirements(TypedDict):
    reqs: list[str]


class StoredResumeArtifact(TypedDict):
    interface_path: str
    commentary: str


class StoredVFS(TypedDict):
    files: dict[str, str]


class StoredSummary(TypedDict):
    summary: str


class StoredProverResult(TypedDict):
    status: str
    analysis: str | None


class StoredManualResult(TypedDict):
    similarity: float
    content: str
    header: str


# ---------------------------------------------------------------------------
# In-memory file view — satisfies InputFileLike
# ---------------------------------------------------------------------------


@dataclass
class _StoredFile:
    """``InputFileLike`` backed by a string that has already been read
    out of the store. No lazy loading; the contents live in memory as
    soon as the enclosing record is fetched."""

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

    def get_file(self, p: str) -> _StoredFile | None:
        c = self._files.get(p)
        if c is None:
            return None
        return _StoredFile(path=p, contents=c)

    def __getitem__(self, p: str) -> _StoredFile | None:
        return self.get_file(p)


# ---------------------------------------------------------------------------
# Resume artifact
# ---------------------------------------------------------------------------


class ResumeArtifact:
    """Bundle of everything needed to resume a prior run: the final
    interface/spec/system views, the full final VFS, and the commentary
    the LLM attached on completion."""

    def __init__(
        self,
        final_intf: _StoredFile,
        final_spec: _StoredFile,
        system_doc: _StoredFile,
        commentary: str,
        intf_path: str,
        vfs_cur: VFSRetriever,
    ):
        self.intf_vfs_handle = final_intf
        self.spec_vfs_handle = final_spec
        self.system_vfs_handle = system_doc
        self.vfs = vfs_cur
        self.commentary = commentary
        self.interface_path = intf_path

    @cached_property
    def interface_file(self) -> str:
        return self.intf_vfs_handle.string_contents

    @cached_property
    def spec_file(self) -> str:
        return self.spec_vfs_handle.string_contents

    @cached_property
    def system_doc(self) -> str:
        return self.system_vfs_handle.string_contents


# ---------------------------------------------------------------------------
# AuditStore
# ---------------------------------------------------------------------------

def _safe_dict(x: StoredProverResult | 
               StoredRunInfo | 
               StoredRequirements | 
               StoredResumeArtifact | 
               StoredVFS | 
               StoredSummary |
               StoredManualResult) -> dict:
    return { **x }

class AuditStore:
    """Async accessor for the audit archive.

    Wraps a ``BaseStore`` (typically ``AsyncPostgresStore``). All
    reads and writes use the ``a*`` (async) store methods, so the
    surrounding code must be async too.
    """

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
        spec_file: InputFileLike,
        interface_file: InputFileLike,
        system_doc: InputFileLike,
        vfs_init: Iterable[tuple[str, bytes]],
        reqs: list[str] | None,
    ) -> None:
        run_info: StoredRunInfo = {
            "spec_name": spec_file.basename,
            "spec_contents": spec_file.string_contents,
            "interface_name": interface_file.basename,
            "interface_contents": interface_file.string_contents,
            "system_name": system_doc.basename,
            "system_contents": system_doc.string_contents,
            "num_reqs": len(reqs) if reqs is not None else None,
        }
        vfs_payload: StoredVFS = {
            "files": {p: c.decode("utf-8") for (p, c) in vfs_init}
        }

        await self._store.aput(self._ns(thread_id), "run_info", _safe_dict(run_info))
        await self._store.aput(self._ns(thread_id), "vfs_initial", _safe_dict(vfs_payload))
        if reqs is not None:
            await self._store.aput(
                self._ns(thread_id), "requirements", {"reqs": reqs}
            )

    async def register_complete(
        self,
        thread_id: str,
        vfs: Iterable[tuple[str, bytes]],
        intf: str,
        commentary: str,
    ) -> None:
        vfs_payload: StoredVFS = {
            "files": {p: c.decode("utf-8") for (p, c) in vfs}
        }
        await self._store.aput(self._ns(thread_id), "vfs_result", _safe_dict(vfs_payload))

        resume: StoredResumeArtifact = {
            "interface_path": intf,
            "commentary": commentary,
        }
        await self._store.aput(self._ns(thread_id), "resume_artifact", _safe_dict(resume))

    # -- reads -------------------------------------------------------------

    async def get_resume_artifact(self, thread_id: str) -> ResumeArtifact:
        ra_item = await self._store.aget(self._ns(thread_id), "resume_artifact")
        if ra_item is None:
            raise RuntimeError(f"No resume artifact found for thread {thread_id}")
        ra = cast(StoredResumeArtifact, ra_item.value)

        ri_item = await self._store.aget(self._ns(thread_id), "run_info")
        if ri_item is None:
            raise RuntimeError(f"No run info found for thread {thread_id}")
        ri = cast(StoredRunInfo, ri_item.value)

        vfs_item = await self._store.aget(self._ns(thread_id), "vfs_result")
        if vfs_item is None:
            raise RuntimeError(f"No vfs_result found for thread {thread_id}")
        vfs_files = cast(StoredVFS, vfs_item.value)["files"]

        intf_contents = vfs_files.get(ra["interface_path"])
        if intf_contents is None:
            raise RuntimeError(
                f"Resume artifact references {ra['interface_path']} "
                f"but it's not in vfs_result"
            )
        spec_contents = vfs_files.get("rules.spec")
        if spec_contents is None:
            raise RuntimeError(
                f"vfs_result for thread {thread_id} has no rules.spec"
            )

        return ResumeArtifact(
            final_intf=_StoredFile(
                path=ra["interface_path"], contents=intf_contents
            ),
            final_spec=_StoredFile(path="rules.spec", contents=spec_contents),
            system_doc=_StoredFile(
                path=ri["system_name"], contents=ri["system_contents"]
            ),
            commentary=ra["commentary"],
            intf_path=ra["interface_path"],
            vfs_cur=VFSRetriever(_files=vfs_files),
        )

    async def get_run_info(
        self, thread_id: str
    ) -> tuple[RunInput, VFSRetriever]:
        ri_item = await self._store.aget(self._ns(thread_id), "run_info")
        if ri_item is None:
            raise RuntimeError(f"Didn't find run info for {thread_id}")
        ri = cast(StoredRunInfo, ri_item.value)

        vfs_item = await self._store.aget(self._ns(thread_id), "vfs_initial")
        vfs_files: dict[str, str] = {}
        if vfs_item is not None:
            vfs_files = cast(StoredVFS, vfs_item.value)["files"]
        retriever = VFSRetriever(_files=vfs_files)

        reqs: list[str] | None = None
        if ri["num_reqs"] is not None:
            req_item = await self._store.aget(
                self._ns(thread_id), "requirements"
            )
            if req_item is not None:
                reqs = cast(StoredRequirements, req_item.value)["reqs"]

        run_input: RunInput = {
            "interface": _StoredFile(
                path=ri["interface_name"], contents=ri["interface_contents"]
            ),
            "spec": _StoredFile(
                path=ri["spec_name"], contents=ri["spec_contents"]
            ),
            "system": _StoredFile(
                path=ri["system_name"], contents=ri["system_contents"]
            ),
            "reqs": reqs,
        }
        return (run_input, retriever)

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

    async def add_manual_result(
        self, thread_id: str, tool_id: str, ref: ManualRef
    ) -> None:
        payload: StoredManualResult = {
            "similarity": ref.similarity,
            "content": ref.content,
            "header": " / ".join(ref.headers),
        }
        await self._store.aput(
            self._ns(thread_id, "manual_results", tool_id),
            uuid.uuid4().hex,
            _safe_dict(payload),
        )

    async def get_manual_results(
        self, thread_id: str, tool_id: str
    ) -> AsyncIterator[ManualResult]:
        items = await self._store.asearch(
            self._ns(thread_id, "manual_results", tool_id), limit=10_000
        )
        for item in items:
            payload = cast(StoredManualResult, item.value)
            yield ManualResult(
                content=payload["content"],
                header=payload["header"],
                similarity=payload["similarity"],
            )

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
    """Implements the (async) ``AuditSink`` protocol by delegating to
    an ``AuditStore`` with a fixed ``thread_id``."""

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

    async def on_manual_search(self, tool_id: str, ref: ManualRef) -> None:
        await self._audit.add_manual_result(
            thread_id=self._thread_id, tool_id=tool_id, ref=ref
        )

    async def on_summarization(self, checkpoint_id: str, summary: str) -> None:
        await self._audit.register_summary(
            thread_id=self._thread_id,
            checkpoint_id=checkpoint_id,
            summary=summary,
        )


# Compatibility re-exports (make the protocol available here so callers
# only need one import).
__all__ = [
    "AuditSink",
    "AuditStore",
    "AuditStoreSink",
    "ResumeArtifact",
    "VFSRetriever",
    "StoredRunInfo",
    "StoredRequirements",
    "StoredResumeArtifact",
    "StoredVFS",
    "StoredSummary",
    "StoredProverResult",
    "StoredManualResult",
]
