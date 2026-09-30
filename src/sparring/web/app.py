"""FastAPI app: upload videos, run analyses, and serve results to the browser UI.

`create_app()` takes its collaborators (data directory, runner) as arguments so tests can run it
with a fake pipeline and no models.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sparring.detection.registry import METHODS
from sparring.disciplines import DISCIPLINES
from sparring.moves import MOVES
from sparring.skeleton import BONES, KEYPOINTS, K
from sparring.web.jobs import Job, JobOptions, JobStore
from sparring.web.worker import Runner, Worker

log = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
CHUNK = 1 << 20

METHOD_HELP = {
    "ensemble": "Rules plus a learned second opinion. Best precision; the default.",
    "rules": "Hand-written geometry only. No trained model needed.",
    "temporal": "Neural network over the motion sequence (needs a trained model).",
    "learned": "Candidate moves scored by a trained classifier (needs a trained model).",
    "hybrid": "Learned classifier that also sees the rules' output (needs a trained model).",
    "rules_tuned": "Rules with thresholds tuned on labelled data (for benchmarking).",
}
UI_METHODS = ("ensemble", "rules", "temporal", "learned", "hybrid")


class Options(BaseModel):
    discipline: str
    method: str


def _validate(discipline: str, method: str) -> None:
    if discipline not in DISCIPLINES:
        raise HTTPException(422, f"unknown discipline {discipline!r}")
    if method not in METHODS:
        raise HTTPException(422, f"unknown method {method!r}")


def _job_view(job: Job, store: JobStore, worker: Worker) -> dict:
    d = job.to_dict()
    d["progress"] = store.progress(job.id) if job.status in ("running", "done") else None
    d["queue_position"] = worker.queue_position(job.id) if job.status == "queued" else None
    d["has_thumbnail"] = (store.dir(job.id) / "thumb.jpg").exists()
    return d


def create_app(
    data_dir: Path,
    runner: Runner | None = None,
    max_upload_mb: int = 2048,
    start_worker: bool = True,
) -> FastAPI:
    store = JobStore(data_dir)
    worker = Worker(store, runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="Sparring analysis", lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.state.store, app.state.worker = store, worker

    def get_job(job_id: str) -> Job:
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "no such analysis") from None

    # -- options -----------------------------------------------------------------------------
    @app.get("/api/options")
    def options() -> dict:
        return {
            "disciplines": [{"id": d.name, "title": d.title} for d in DISCIPLINES.values()],
            "methods": [{"id": m, "help": METHOD_HELP[m]} for m in UI_METHODS],
            "defaults": {"discipline": "boxing", "method": "ensemble"},
            "moves": {
                k: {"kind": m.kind, "label": m.label, "limb": m.limb, "side": m.side}
                for k, m in MOVES.items()
            },
            "max_upload_mb": max_upload_mb,
        }

    # -- jobs --------------------------------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [_job_view(j, store, worker) for j in store.list()]

    @app.post("/api/jobs", status_code=201)
    async def create_job(
        file: Annotated[UploadFile, File()],
        discipline: Annotated[str, Form()] = "boxing",
        method: Annotated[str, Form()] = "ensemble",
        fighters: Annotated[str | None, Form()] = None,
    ) -> dict:
        _validate(discipline, method)
        name = Path(file.filename or "video").name
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(
                415, f"unsupported file type {suffix or '(none)'}; use a video file"
            )
        pair = None
        if fighters:
            try:
                a, b = (int(x) for x in fighters.split(","))
                pair = (a, b)
            except ValueError:
                raise HTTPException(422, "fighters must be two track ids, e.g. 3,7") from None
        job, dest = store.create(name, suffix, JobOptions(discipline, method, pair))
        limit = max_upload_mb * CHUNK
        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"file is larger than {max_upload_mb} MB")
                    out.write(chunk)
        except HTTPException:
            store.delete(job.id)
            raise
        if size == 0:
            store.delete(job.id)
            raise HTTPException(422, "the file is empty")
        worker.submit(job.id)
        return _job_view(store.get(job.id), store, worker)

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict:
        return _job_view(get_job(job_id), store, worker)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        job = get_job(job_id)
        if job.status == "running":
            raise HTTPException(409, "this analysis is running; wait for it to finish")
        store.delete(job_id)
        return Response(status_code=204)

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str, opts: Options | None = None) -> dict:
        job = get_job(job_id)
        if job.status in ("queued", "running"):
            raise HTTPException(409, "already queued or running")
        options = job.options
        if opts:
            _validate(opts.discipline, opts.method)
            options = JobOptions(opts.discipline, opts.method, job.options.fighters)
        store.update(job_id, status="queued", error=None, log_tail=[], options=options)
        worker.submit(job_id)
        return _job_view(store.get(job_id), store, worker)

    @app.post("/api/jobs/{job_id}/reanalyze")
    async def reanalyze(job_id: str, opts: Options) -> dict:
        get_job(job_id)
        _validate(opts.discipline, opts.method)
        options = JobOptions(opts.discipline, opts.method, store.get(job_id).options.fighters)
        try:
            job = await run_in_threadpool(worker.reanalyze, job_id, options)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        except RuntimeError as e:
            raise HTTPException(500, f"re-analysis failed: {e}") from None
        return _job_view(job, store, worker)

    # -- results -----------------------------------------------------------------------------
    def _done(job_id: str) -> Path:
        job = get_job(job_id)
        if job.status != "done":
            raise HTTPException(409, "analysis not finished")
        return store.run_dir(job_id)

    @app.get("/api/jobs/{job_id}/analysis")
    def analysis(job_id: str) -> JSONResponse:
        run = _done(job_id)
        events = [e for e in json.loads((run / "events.json").read_text())]
        payload = {
            "job": _job_view(store.get(job_id), store, worker),
            "video": json.loads((run / "video.json").read_text()),
            "report": json.loads((run / "report.json").read_text()),
            "fighters": json.loads((run / "fighters.json").read_text()),
            "events": events,
        }
        return JSONResponse(payload)

    @app.get("/api/jobs/{job_id}/poses")
    def poses(job_id: str) -> Response:
        """Cleaned keypoints as little-endian int16 pixels, shape (frames, 2, K, 2), -1 = missing.
        Shape and skeleton are in the response headers / the /skeleton endpoint."""
        run = _done(job_id)
        kp = np.load(run / "poses_clean.npy")
        q = np.where(np.isfinite(kp), np.round(kp), -1).clip(-1, 32767).astype("<i2")
        return Response(
            q.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": ",".join(str(s) for s in q.shape),
                "Cache-Control": "private, max-age=3600",
            },
        )

    @app.get("/api/skeleton")
    def skeleton() -> dict:
        return {"keypoints": list(KEYPOINTS), "bones": [[K[a], K[b]] for a, b in BONES]}

    @app.get("/api/jobs/{job_id}/video")
    def video(job_id: str) -> FileResponse:
        get_job(job_id)
        path = store.dir(job_id) / "playback.mp4"
        if not path.exists():
            raise HTTPException(404, "video not ready")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/thumbnail")
    def thumbnail(job_id: str) -> FileResponse:
        get_job(job_id)
        path = store.dir(job_id) / "thumb.jpg"
        if not path.exists():
            raise HTTPException(404, "no thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    # -- frontend ----------------------------------------------------------------------------
    static = resources.files("sparring") / "web" / "static"
    if static.is_dir() and (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(static), html=True), name="ui")
    else:

        @app.get("/")
        def no_ui() -> Response:
            return Response(
                "The web UI has not been built. Run: cd web && pnpm install && pnpm build",
                media_type="text/plain",
            )

    return app
