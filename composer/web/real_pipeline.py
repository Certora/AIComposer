"""Phase 4: real autoprove pipeline driver for the web frontend.

Drives an actual ``run_autoprove`` invocation through
:class:`AutoProveWebHandler`. Mirrors ``run_mock_pipeline``'s entry
shape so :mod:`app.py` can switch between them with one import.

Project materialization happens earlier — at ``POST /projects`` time
in :mod:`composer.web.projects` — so this module is handed an already-
resolved ``project_root`` and just runs the pipeline against it. The
``MainContract`` parsing stays here because it's a run-start concern
(the user might pick a different ``path:Name`` for two runs against
the same materialized project).

The web handler doesn't run interactive HITL (v1 ships hands-off), and
``cache_ns`` is auto-generated from the contract identifier so retry
support (Phase 5) has a stable key without exposing the field in the
form. Memory namespace stays ``None`` and falls through to thread-id
inside ``WorkflowContext``.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
from dataclasses import dataclass

from composer.rag.db import DEFAULT_CONNECTION as RAG_DEFAULT
from composer.spec.source.autoprove_common import AutoProveInputs, run_autoprove
from composer.web.handler import AutoProveWebHandler
from composer.web.runs import RunState


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(s: str) -> str:
    """Sanitize a string into a cache-namespace-safe key. Alphanumerics
    and ``_``/``-`` pass through; everything else collapses to ``_``."""
    out = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s)
    return out.strip("_") or "default"


@dataclass(frozen=True)
class MainContract:
    """The ``path:Name`` pair pre-parsed and resolved against project
    root. Constructed at the app boundary so the pipeline driver only
    sees well-formed values."""
    path: pathlib.Path        # absolute, inside project_root
    name: str


def parse_main_contract(raw: str, project_root: pathlib.Path) -> MainContract:
    """Parse the form's ``path:ContractName`` string and resolve the
    path against *project_root*. Raises ``ValueError`` for malformed
    input or paths outside the project root."""
    if ":" not in raw:
        raise ValueError(
            f"main_contract must be 'path:ContractName', got {raw!r}"
        )
    path_str, name = raw.split(":", 1)
    abs_path = (project_root / path_str).resolve()
    if not abs_path.is_relative_to(project_root):
        raise ValueError(
            f"main_contract path {abs_path} resolves outside project_root {project_root}"
        )
    if not abs_path.is_file():
        raise ValueError(f"main_contract file does not exist: {abs_path}")
    return MainContract(path=abs_path, name=name)


# ---------------------------------------------------------------------------
# Whole-run driver
# ---------------------------------------------------------------------------

async def run_real_pipeline(
    run: RunState,
    *,
    project_root: pathlib.Path,
    main_contract_raw: str,
    model: str,
    max_concurrent: int,
    cloud: bool,
    system_doc_path: pathlib.Path,
    threat_model_path: pathlib.Path | None,
) -> None:
    """Drive a real autoprove run end-to-end.

    *project_root* is expected to be already materialized (by
    :mod:`composer.web.projects`). All other paths reference files
    saved to the run's workspace at request time.

    Catches its own exceptions and routes them through
    :meth:`AutoProveWebHandler.crashed` so a validation / pipeline
    failure renders as a failed run rather than a stuck SSE stream that
    never finishes."""
    handler = AutoProveWebHandler(run)

    try:
        # 1. Validate main_contract shape and resolve against project_root.
        main_contract = parse_main_contract(main_contract_raw, project_root)

        # 2. Build inputs. cache_ns auto-derived from contract name so
        # retries for the same contract reuse cached phases (Phase 5
        # plumbs the actual reuse; for now this just makes the key
        # stable).
        cache_ns = f"web-{_slugify(main_contract.name)}"
        inputs = AutoProveInputs(
            project_root=project_root,
            main_contract_path=main_contract.path,
            contract_name=main_contract.name,
            system_doc_path=system_doc_path,
            threat_model_path=threat_model_path,
            max_concurrent=max_concurrent,
            cloud=cloud,
            interactive=False,
            cache_ns=cache_ns,
            memory_ns=None,
            rag_db=RAG_DEFAULT,
            model=model,
            # tokens / thinking_tokens / memory_tool / interleaved_thinking
            # all stay at AutoProveInputs defaults — Phase 4 doesn't
            # surface them in the form. Phase 6 polish can if we want.
        )

        # 3. Run. ``thread_id=None`` means a fresh UUID; Phase 5's retry
        # path will pass an existing thread id for soft retry.
        await run_autoprove(
            inputs,
            handler_factory=handler.make_handler,
            thread_id=None,
        )

        # 4. Surface generated files. The pipeline writes ``.spec`` files
        # into ``project_root/certora/`` (and possibly elsewhere); copy
        # them into ``<workspace>/outputs/`` so the existing /files
        # route can serve them regardless of whether project_root lives
        # inside or outside the workspace.
        outputs_dir = run.workspace / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        files: list[dict] = []
        certora_dir = project_root / "certora"
        if certora_dir.is_dir():
            for src in certora_dir.rglob("*.spec"):
                rel = src.relative_to(certora_dir)
                dst = outputs_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                files.append({
                    "url":   f"/runs/{run.run_id}/files/outputs/{rel}",
                    "label": str(rel),
                })
        run.output_files = files
        handler.finish()

    except Exception as exc:
        _logger.exception("Run %s failed", run.run_id)
        handler.crashed(exc)
