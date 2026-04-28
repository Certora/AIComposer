"""Phase 0.5 server: project materialization + contract scanning.

The form's project picker uploads / clones / points-at the project as
soon as the user blurs the relevant input — no waiting for form
submit. This module is the server side of that:

  - ``POST /projects`` (in app.py) instantiates a ``ProjectState``,
    fires off a background task to materialize and scan, returns the
    ``project_id`` immediately.
  - ``GET /projects/{id}`` (in app.py) reports ``{status, contracts, error}``
    for client polling. Once ``status == "ready"`` the JS populates the
    main-contract ``<datalist>`` and the form's hidden ``project_id``
    field; the user submits and ``POST /runs`` looks the project up by
    id rather than re-materializing.

Project source modes are encoded as a sum type so this module never
sees the three-nullable-strings shape — see :data:`ProjectSource`.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import pathlib
import re
import zipfile
from dataclasses import dataclass, field

from composer.spec.util import FS_FORBIDDEN_READ


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project source — sum type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalProject:
    """Existing directory on the local filesystem; used as-is, no copy.

    Avoids duplicating large monorepos. Generated specs are copied into
    the run's outputs dir after a run so they're servable regardless of
    where the project itself lives."""
    path: pathlib.Path


@dataclass(frozen=True)
class GitProject:
    """Remote repo to clone via ``git clone --depth 1``."""
    url: str


@dataclass(frozen=True)
class ZipProject:
    """Already-saved zip archive on disk to extract."""
    archive_path: pathlib.Path


type ProjectSource = LocalProject | GitProject | ZipProject


def describe_source(source: ProjectSource) -> str:
    """Human-readable label, used in the run page's inputs panel."""
    match source:
        case LocalProject(path=p):       return f"local: {p}"
        case GitProject(url=url):        return f"git: {url}"
        case ZipProject(archive_path=p): return f"zip: {p.name}"


# ---------------------------------------------------------------------------
# Project state
# ---------------------------------------------------------------------------

class ProjectStatus(enum.Enum):
    MATERIALIZING = "materializing"
    SCANNING      = "scanning"
    READY         = "ready"
    ERROR         = "error"


@dataclass
class ProjectState:
    project_id: str
    source: ProjectSource
    workspace: pathlib.Path           # <projects_root>/<project_id>/
    status: ProjectStatus = ProjectStatus.MATERIALIZING
    project_root: pathlib.Path | None = None      # set on READY
    contracts: list[str] = field(default_factory=list)  # ``"path:Name"`` strings
    error: str | None = None


PROJECTS: dict[str, ProjectState] = {}


# ---------------------------------------------------------------------------
# Materialization (was in real_pipeline.py; lives here now)
# ---------------------------------------------------------------------------

async def _materialize_project(
    workspace: pathlib.Path,
    source: ProjectSource,
) -> pathlib.Path:
    """Land *source* into a directory we can walk for ``.sol`` files.
    Returns the absolute path to the project root."""
    match source:
        case LocalProject(path=p):
            src = p.expanduser().resolve()
            if not src.exists():
                raise ValueError(f"Project path does not exist: {src}")
            if not src.is_dir():
                raise ValueError(f"Project path is not a directory: {src}")
            return src

        case GitProject(url=url):
            project_dir = workspace / "project"
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", url, str(project_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise ValueError(
                    f"git clone failed: {stderr.decode(errors='replace')[:500]}"
                )
            return project_dir.resolve()

        case ZipProject(archive_path=zip_path):
            project_dir = workspace / "project"
            project_dir.mkdir(exist_ok=True)
            # zipfile is sync; offload to a thread so multi-MB archives
            # don't stall the event loop.
            def _extract() -> None:
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(project_dir)
            await asyncio.to_thread(_extract)
            return project_dir.resolve()


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

# ``contract Foo`` and ``abstract contract Foo`` are verification targets;
# ``interface`` and ``library`` are not. Multiline mode so we don't
# accidentally match inside a comment block or string literal — we'd
# need a real parser to be perfectly correct here, but for the
# datalist-suggestion use case the freeform input remains as the
# escape hatch (the user can always type a ``path:Name`` not in the
# list).
_CONTRACT_RE = re.compile(
    r"^\s*(?:abstract\s+)?contract\s+(\w+)", re.MULTILINE,
)

# Use the same forbidden-read regex autoprove itself uses — single
# source of truth for "paths the pipeline doesn't want to see". Catches
# ``lib/``, ``test/``, ``.git``, ``node_modules/`` (with .sol exception),
# ``.certora_internal``, ``emv-*``, ``*.json``.
_FORBIDDEN_RE = re.compile(FS_FORBIDDEN_READ)

# Build-artifact / IDE noise the canonical regex doesn't cover but
# we still want excluded from a project picker view.
_EXTRA_SKIP_DIRS = {
    "out",      # Foundry artifacts
    "cache",    # Foundry / Hardhat
    "build",    # generic
    "dist",     # generic
    ".vscode",  # IDE
    ".idea",    # IDE
}


def _is_skipped(rel: pathlib.PurePath) -> bool:
    """True if *rel* (relative to project root) falls under any
    excluded directory. Combines :data:`FS_FORBIDDEN_READ` (autoprove's
    canonical exclude list) with extras for build/IDE noise *and* a
    catch-all for ``.certora_*`` directories: the canonical regex only
    catches ``.certora_internal``, but ``.certora_sources`` and other
    Certora-tooling-generated siblings need the same treatment."""
    rel_str = str(rel)
    if _FORBIDDEN_RE.search(rel_str):
        return True
    for part in rel.parts:
        if part.startswith(".certora_"):
            return True
        if part in _EXTRA_SKIP_DIRS:
            return True
    return False


def _scan_contracts(project_root: pathlib.Path) -> list[str]:
    """Walk *project_root* for ``.sol`` files, extract contract
    declarations as ``"<rel_path>:<ContractName>"`` strings.

    Skips paths flagged by :func:`_is_skipped` so the suggestion list
    isn't dominated by vendored OpenZeppelin contracts, test fixtures,
    Certora-generated artefacts, or build output."""
    out: list[str] = []
    for sol in project_root.rglob("*.sol"):
        rel = sol.relative_to(project_root)
        # ``rglob`` walks everything; we filter post-hoc since pathlib
        # doesn't expose directory-level pruning natively.
        if _is_skipped(rel):
            continue
        try:
            text = sol.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _logger.warning("scan: skipping %s — %s", sol, exc)
            continue
        for match in _CONTRACT_RE.finditer(text):
            name = match.group(1)
            out.append(f"{rel}:{name}")
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def run_project_setup(state: ProjectState) -> None:
    """Materialize the project, scan it, and update *state* in place.

    Single coroutine to avoid having to thread state references through
    multiple steps. Catches its own exceptions and writes them to
    ``state.error`` so the polling endpoint can surface them — never
    raises out of this task."""
    try:
        state.status = ProjectStatus.MATERIALIZING
        project_root = await _materialize_project(state.workspace, state.source)
        state.project_root = project_root

        state.status = ProjectStatus.SCANNING
        # Scan is synchronous (regex over ``.sol`` files); offload to a
        # thread so a large repo doesn't block the loop. On a typical
        # Solidity project this returns in well under a second.
        contracts = await asyncio.to_thread(_scan_contracts, project_root)
        state.contracts = contracts

        state.status = ProjectStatus.READY
    except Exception as exc:
        _logger.exception("project %s setup failed", state.project_id)
        state.status = ProjectStatus.ERROR
        state.error = f"{type(exc).__name__}: {exc}"
