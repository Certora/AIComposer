"""FastAPI app for the autoprove web frontend.

Phase 0 shipped the form (``GET /`` + ``POST /runs`` returning 501).
Phase 2 wired the in-memory run registry, SSE event stream, and the
mock pipeline driver. Phase 4 wires the real ``run_autoprove_pipeline``
call; the mock driver remains available behind ``AUTOPROVE_WEB_MOCK=1``
for development and demos that don't want to touch real services.

Run locally::

    uvicorn composer.web.app:app --reload --host 127.0.0.1 --port 8000

Set ``AUTOPROVE_WEB_MOCK=1`` to use the canned fake driver instead of
the real pipeline.
"""

import asyncio
import contextlib
import os
import pathlib
import shutil
import uuid
from dataclasses import replace
from datetime import datetime
from typing import AsyncIterator, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from composer.web.projects import (
    PROJECTS,
    GitProject,
    LocalProject,
    ProjectSource,
    ProjectState,
    ProjectStatus,
    ZipProject,
    describe_source,
    run_project_setup,
)
from composer.web.real_pipeline import run_real_pipeline, slugify
from composer.web.runs import RUNS, RunRequest, RunState, serialize_event


_USE_MOCK = os.environ.get("AUTOPROVE_WEB_MOCK") == "1"


# Project workspaces live under a sibling root from run workspaces so
# they have independent lifetimes (a project can be referenced by
# multiple runs over time).
PROJECTS_ROOT = pathlib.Path.home() / ".cache" / "autoprove-web" / "projects"
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)


BASE = pathlib.Path(__file__).parent
WORKSPACE_ROOT = pathlib.Path.home() / ".cache" / "autoprove-web"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Autoprove (web)")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _save_upload(
    workspace: pathlib.Path, name: str, file: UploadFile,
) -> pathlib.Path:
    """Persist *file*'s bytes under ``<workspace>/<name><ext>`` and
    return the saved path.

    Must be called during the request — UploadFile handles get torn
    down once the response returns, so the background pipeline task
    can't read from them. ``shutil.copyfileobj`` is sync so we offload
    to a thread to avoid stalling the loop on multi-MB uploads."""
    suffix = pathlib.Path(file.filename or "").suffix
    dest = workspace / f"{name}{suffix}"

    def _do_copy() -> None:
        with dest.open("wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
    await asyncio.to_thread(_do_copy)
    return dest


def _project_source_from_form(
    project_mode: str,
    project_path: str,
    project_git: str,
    project_zip_path: pathlib.Path | None,
) -> ProjectSource:
    """Translate the form's segmented-control mode + per-mode inputs
    into the right ``ProjectSource`` variant. Raises ``HTTPException``
    on user-error shapes (the values came from form fields, so a 4xx
    is the right surface)."""
    match project_mode:
        case "path":
            stripped = project_path.strip()
            if not stripped:
                raise HTTPException(400, "project_path is required when mode is 'path'")
            return LocalProject(path=pathlib.Path(stripped))
        case "git":
            stripped = project_git.strip()
            if not stripped:
                raise HTTPException(400, "project_git is required when mode is 'git'")
            return GitProject(url=stripped)
        case "zip":
            if project_zip_path is None:
                raise HTTPException(400, "project_zip is required when mode is 'zip'")
            return ZipProject(archive_path=project_zip_path)
        case _:
            raise HTTPException(400, f"unknown project_mode: {project_mode!r}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/projects")
async def create_project(
    request: Request,
    project_mode: str = Form(...),
    project_path: str = Form(""),
    project_git: str = Form(""),
    project_zip: UploadFile | None = File(None),
):
    """Materialize a project in the background and return its id.

    Called out-of-band by the form's main-contract picker JS as soon
    as the user blurs the relevant project-source input. The picker
    polls ``GET /projects/{id}`` until status ``"ready"`` and then
    populates the ``<datalist>`` from the returned contracts list.
    """
    project_id = uuid.uuid4().hex[:12]
    workspace = PROJECTS_ROOT / project_id
    workspace.mkdir()

    project_zip_path: pathlib.Path | None = None
    if project_zip is not None and project_zip.filename:
        project_zip_path = await _save_upload(workspace, "upload", project_zip)

    source = _project_source_from_form(
        project_mode, project_path, project_git, project_zip_path,
    )

    state = ProjectState(
        project_id=project_id,
        source=source,
        workspace=workspace,
    )
    PROJECTS[project_id] = state

    asyncio.create_task(run_project_setup(state))

    return {"project_id": project_id}


@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    """Polling endpoint for the picker. Returns the project's current
    status; once ``status == "ready"`` the ``contracts`` list is
    populated."""
    state = PROJECTS.get(project_id)
    if state is None:
        raise HTTPException(404, "project not found")
    return {
        "status":    state.status.value,
        "contracts": state.contracts,
        "error":     state.error,
    }


@app.post("/runs")
async def create_run(
    request: Request,
    project_id: str = Form(...),
    main_contract: str = Form(...),
    model: str = Form("claude-opus-4-7"),
    max_concurrent: int = Form(4),
    cloud: bool = Form(False),
    system_doc: UploadFile | None = File(None),
    threat_model: UploadFile | None = File(None),
):
    """Create a run against an already-materialized project.

    ``project_id`` references a ``ProjectState`` set up by an earlier
    ``POST /projects`` (see ``run_project_setup`` for the materialize
    + scan flow). System doc and threat model are run-scoped uploads
    that aren't shared across runs even if the project is reused."""
    if system_doc is None or not system_doc.filename:
        raise HTTPException(400, "system_doc is required")

    project = PROJECTS.get(project_id)
    if project is None:
        raise HTTPException(400, f"unknown project_id {project_id!r}")
    if project.status is not ProjectStatus.READY or project.project_root is None:
        raise HTTPException(
            400,
            f"project {project_id} is not ready (status={project.status.value})",
        )

    run_id = uuid.uuid4().hex[:12]
    workspace = WORKSPACE_ROOT / run_id
    workspace.mkdir()

    # Persist all run-scoped uploads while the request is alive — see
    # _save_upload's docstring for why this can't slip into the
    # background task.
    sys_doc_path = await _save_upload(workspace, "system_doc", system_doc)
    threat_path: pathlib.Path | None = None
    if threat_model is not None and threat_model.filename:
        threat_path = await _save_upload(workspace, "threat_model", threat_model)

    inputs = {
        "main_contract":  main_contract,
        "model":          model,
        "max_concurrent": max_concurrent,
        "cloud":          cloud,
        "project":        describe_source(project.source),
        "system_doc":     system_doc.filename,
        "threat_model":   (
            threat_model.filename
            if threat_model is not None and threat_model.filename
            else "(none)"
        ),
    }

    run = RunState(
        run_id=run_id,
        workspace=workspace,
        started_at=datetime.now(),
        inputs=inputs,
    )
    RUNS[run_id] = run

    if _USE_MOCK:
        from composer.web.mock_pipeline import run_mock_pipeline
        asyncio.create_task(run_mock_pipeline(run))
    else:
        # Build the RunRequest at submit time so retry has a stable
        # handle on cache_ns / root_thread_id (the two values needed
        # to differentiate soft vs hard retry semantics).
        request = RunRequest(
            project_id=project_id,
            main_contract_raw=main_contract,
            model=model,
            max_concurrent=max_concurrent,
            cloud=cloud,
            system_doc_path=sys_doc_path,
            threat_model_path=threat_path,
            cache_ns=f"web-{slugify(main_contract)}",
            memory_ns=None,
            root_thread_id=f"autoprove_{uuid.uuid4().hex[:12]}",
        )
        run.request = request
        asyncio.create_task(run_real_pipeline(
            run, project_root=project.project_root, request=request,
        ))

    return RedirectResponse(url=f"/runs/{run_id}", status_code=303)


@app.post("/runs/{old_run_id}/retry")
async def retry_run(
    old_run_id: str,
    mode: Literal["soft", "hard"] = Form(...),
):
    """Re-launch a failed run.

    *soft*: same ``(cache_ns, memory_ns, root_thread_id)`` triple →
    cached phases skip, the failing phase resumes from its langgraph
    checkpoint (cheap, but inherits the agent state at the moment of
    failure).

    *hard*: same ``(cache_ns, memory_ns)``, fresh ``root_thread_id`` →
    cached phases still skip, but the failing phase gets a clean
    conversation. Use when soft retry can't escape a bad agent loop.

    The new run gets its own ``run_id`` and workspace; uploaded
    artefacts (system doc, threat model) are referenced from the old
    run's workspace rather than copied — Phase 6 polish can switch to
    copy if workspace cleanup ever lands.
    """
    old_run = RUNS.get(old_run_id)
    if old_run is None:
        raise HTTPException(404, f"unknown run {old_run_id!r}")
    if old_run.request is None:
        raise HTTPException(400, "this run can't be retried (no recorded request)")

    project = PROJECTS.get(old_run.request.project_id)
    if project is None or project.project_root is None:
        raise HTTPException(
            400,
            f"project {old_run.request.project_id} for run {old_run_id} "
            f"is no longer available",
        )

    new_request: RunRequest
    match mode:
        case "soft":
            new_request = old_run.request
        case "hard":
            new_request = replace(
                old_run.request,
                root_thread_id=f"autoprove_{uuid.uuid4().hex[:12]}",
            )

    new_run_id = uuid.uuid4().hex[:12]
    new_workspace = WORKSPACE_ROOT / new_run_id
    new_workspace.mkdir()

    inputs = {
        **old_run.inputs,
        "retry_of":  old_run_id,
        "retry_mode": mode,
    }
    new_run = RunState(
        run_id=new_run_id,
        workspace=new_workspace,
        started_at=datetime.now(),
        inputs=inputs,
        request=new_request,
    )
    RUNS[new_run_id] = new_run

    asyncio.create_task(run_real_pipeline(
        new_run, project_root=project.project_root, request=new_request,
    ))

    return RedirectResponse(url=f"/runs/{new_run_id}", status_code=303)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def get_run(request: Request, run_id: str) -> HTMLResponse:
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    # Page is mostly an empty shell — SSE replay populates summary aside,
    # task panels, and outputs section. Only the run-status pill is
    # rendered up-front (it's set in RunState directly, never via push).
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run":             run,
            "run_status_html": run.render_status_html(),
        },
    )


def _parse_last_event_id(request: Request) -> int:
    """Read the SSE ``Last-Event-ID`` header. Returns -1 when absent
    (fresh connection) or unparseable."""
    raw = request.headers.get("Last-Event-ID")
    if raw is None:
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


async def _stream_run(request: Request, run: RunState, last_id: int) -> AsyncIterator[str]:
    """Replay missed events then live-stream new ones.

    Race-free subscribe: we subscribe and snapshot the events length
    *synchronously* — no ``await`` between — so no ``push`` can interleave.
    Every event with ``seq < split`` is in ``run.events[:split]`` only;
    every event with ``seq >= split`` lands in ``queue``. No double-delivery,
    no gap.
    """
    # Stale check — only meaningful for *reconnects* (last_id >= 0). A
    # fresh page-load with no Last-Event-ID gets whatever's still in the
    # buffer; that's correct, since the page just rendered an empty
    # shell and replay populates from low_water_mark forward.
    if last_id >= 0 and last_id + 1 < run.low_water_mark:
        yield "event: stale\ndata: \n\n"
        return

    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2048)
    run.subscribers.append(queue)
    split = len(run.events)
    low_water = run.low_water_mark

    try:
        yield ": connected\n\n"
        # Replay the gap. Slice creates a snapshot list, so concurrent
        # ``push``es appending past ``split`` (which land in queue)
        # don't disturb iteration.
        replay_start_idx = max(0, last_id + 1 - low_water)
        for e in run.events[replay_start_idx:split]:
            yield serialize_event(e)
        # Live stream from here onwards.
        while True:
            if await request.is_disconnected():
                return
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield msg
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
            if run.done and queue.empty():
                yield ": done\n\n"
                return
    finally:
        with contextlib.suppress(ValueError):
            run.subscribers.remove(queue)


@app.get("/runs/{run_id}/events")
async def run_events(request: Request, run_id: str):
    """SSE stream for *run_id*. One subscriber queue per connection;
    the renderer broadcasts to every subscriber on each push.

    Honours ``Last-Event-ID`` for resume — the browser auto-tracks the
    latest ``id:`` field it received and sends it back on reconnect, so
    any wifi blip / page reload picks up exactly where it left off
    (or, if the buffer's been trimmed past that point, gets a ``stale``
    event the JS turns into a ``location.reload()``)."""
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    last_id = _parse_last_event_id(request)

    return StreamingResponse(
        _stream_run(request, run, last_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",          # disable nginx buffering if proxied
        },
    )


@app.get("/runs/{run_id}/files/{path:path}")
async def get_file(run_id: str, path: str):
    """Serve a generated file from the run's workspace. Path-traversal
    guarded by ``is_relative_to`` against the resolved workspace root."""
    run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    full = (run.workspace / path).resolve()
    if not full.is_relative_to(run.workspace.resolve()) or not full.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(full, media_type="text/plain")
