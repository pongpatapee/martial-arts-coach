"""Background processing of analysis jobs.

One worker thread takes jobs off a queue, in order. For each job it transcodes the upload into a
browser-friendly playback file, then runs the engine (`sparring analyze`) in a *subprocess*, so an
inference crash or out-of-memory kills only that job, never the web server. The engine writes
`progress.json` into the run directory, which the API reads for the progress view.

Re-analysis after an option change (discipline / method) re-runs only the cheap stages and does
not wait behind the queue.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

from sparring.web.jobs import Job, JobOptions, JobStore

log = logging.getLogger(__name__)

LOG_TAIL_LINES = 15
REANALYZE_FROM = "events"
LAST_STAGE = "report"  # the web app draws its own overlay; the rendered video isn't needed


class Runner(Protocol):
    """How a job is executed. The default shells out to ffmpeg and the engine; tests use a fake."""

    def prepare(self, job: Job, store: JobStore) -> dict: ...

    def analyze(self, job: Job, store: JobStore, from_stage: str) -> None: ...


def _tail(path: Path, n: int = LOG_TAIL_LINES) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except FileNotFoundError:
        return []


class EngineRunner:
    """ffmpeg for playback/thumbnail, the `sparring` CLI for the analysis."""

    MAX_PLAYBACK_SIDE = 1280  # long side of the browser copy; poses stay in source pixels

    def prepare(self, job: Job, store: JobStore) -> dict:
        d = store.dir(job.id)
        src = d / job.input_file
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate:format=duration",
                    "-of",
                    "json",
                    str(src),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        st = probe["streams"][0]
        num, den = (int(x) for x in st["avg_frame_rate"].split("/"))
        fps = num / den if den else 30.0
        # Keyframe every second so seeking and frame stepping are fast in the browser.
        gop = str(max(1, round(fps)))
        side = self.MAX_PLAYBACK_SIDE
        scale = f"scale='if(gt(iw,ih),min({side},iw),-2)':'if(gt(iw,ih),-2,min({side},ih))'"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                scale,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-g",
                gop,
                "-keyint_min",
                gop,
                "-sc_threshold",
                "0",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(d / "playback.mp4"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                "1",
                "-i",
                str(src),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                str(d / "thumb.jpg"),
            ],
            check=False,
            capture_output=True,
        )
        return {
            "fps": fps,
            "width": int(st["width"]),
            "height": int(st["height"]),
            "duration_s": float(probe["format"].get("duration", 0.0)),
        }

    def analyze(self, job: Job, store: JobStore, from_stage: str) -> None:
        d = store.dir(job.id)
        cmd = [
            sys.executable,
            "-m",
            "sparring.cli",
            "analyze",
            str(d / job.input_file),
            "--out",
            str(store.run_dir(job.id)),
            "--from-stage",
            from_stage,
            "--to-stage",
            LAST_STAGE,
            "--discipline",
            job.options.discipline,
            "--method",
            job.options.method,
        ]
        if job.options.fighters:
            cmd += ["--fighters", ",".join(str(x) for x in job.options.fighters)]
        with (d / "run.log").open("a") as out:
            out.write(f"\n$ {' '.join(cmd)}\n")
            out.flush()
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"analysis exited with code {proc.returncode}")


class Worker:
    def __init__(self, store: JobStore, runner: Runner | None = None):
        self.store = store
        self.runner = runner or EngineRunner()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._reanalyzing: set[str] = set()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> None:
        # Jobs interrupted by a restart go back to the queue, oldest first.
        for job in sorted(self.store.list(), key=lambda j: j.created):
            if job.status in ("queued", "running"):
                self.store.update(job.id, status="queued")
                self._queue.put(job.id)
        self._thread = threading.Thread(target=self._loop, name="analysis-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def queue_position(self, job_id: str) -> int | None:
        with self._queue.mutex:
            items = [j for j in self._queue.queue if j is not None]
        return items.index(job_id) + 1 if job_id in items else None

    def wait_idle(self) -> None:
        """Block until every submitted job has been processed (used by tests)."""
        self._queue.join()

    # -- processing --------------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._process(job_id)
            finally:
                self._queue.task_done()

    def _process(self, job_id: str) -> None:
        try:
            job = self.store.update(job_id, status="running", error=None, log_tail=[])
        except KeyError:
            return  # deleted while queued
        try:
            meta = self.runner.prepare(job, self.store)
            job = self.store.update(job_id, **meta)
            self.runner.analyze(job, self.store, from_stage="ingest")
            self.store.update(job_id, status="done")
        except Exception as e:  # noqa: BLE001 - any failure must be recorded on the job
            log.exception("job %s failed", job_id)
            try:
                self.store.update(
                    job_id,
                    status="failed",
                    error=str(e) or e.__class__.__name__,
                    log_tail=_tail(self.store.dir(job_id) / "run.log"),
                )
            except KeyError:
                pass

    def reanalyze(self, job_id: str, options: JobOptions) -> Job:
        """Re-run detection + report with new options (seconds). Runs in the caller's thread."""
        job = self.store.get(job_id)
        if job.status != "done":
            raise ValueError("only finished analyses can be re-analysed")
        with self._lock:
            if job_id in self._reanalyzing:
                raise ValueError("already re-analysing this job")
            self._reanalyzing.add(job_id)
        previous = job.options
        try:
            job = self.store.update(job_id, options=options)
            self.runner.analyze(job, self.store, from_stage=REANALYZE_FROM)
            return self.store.get(job_id)
        except Exception:
            self.store.update(job_id, options=previous)
            raise
        finally:
            with self._lock:
                self._reanalyzing.discard(job_id)
