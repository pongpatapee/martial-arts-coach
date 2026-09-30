"""Analysis jobs, persisted on disk so they survive a server restart.

Layout per job (under the data directory):

    jobs/<id>/job.json        # this record
    jobs/<id>/input.<ext>     # the upload, untouched
    jobs/<id>/playback.mp4    # H.264/AAC transcode for the browser
    jobs/<id>/thumb.jpg
    jobs/<id>/run/            # the engine's run directory (features, events, report, ...)
    jobs/<id>/run.log         # engine output, for error reporting
"""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal["queued", "running", "done", "failed"]


@dataclass
class JobOptions:
    discipline: str = "boxing"
    method: str = "ensemble"
    fighters: tuple[int, int] | None = None  # tracker ids override


@dataclass
class Job:
    id: str
    name: str
    input_file: str
    status: Status = "queued"
    options: JobOptions = field(default_factory=JobOptions)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)
    duration_s: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        opts = d.pop("options", {}) or {}
        if opts.get("fighters"):
            opts["fighters"] = tuple(opts["fighters"])
        return cls(options=JobOptions(**opts), **d)


class JobStore:
    """Thread-safe store of jobs as JSON files. The lock serialises read-modify-write updates
    from the API and the worker thread within one server process."""

    def __init__(self, root: Path):
        self.root = root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def dir(self, job_id: str) -> Path:
        if not job_id.isalnum():  # ids are hex; never let a path component through
            raise KeyError(job_id)
        return self.root / job_id

    def run_dir(self, job_id: str) -> Path:
        return self.dir(job_id) / "run"

    def create(self, name: str, suffix: str, options: JobOptions) -> tuple[Job, Path]:
        job_id = secrets.token_hex(8)
        d = self.dir(job_id)
        d.mkdir(parents=True)
        job = Job(id=job_id, name=name, input_file=f"input{suffix}", options=options)
        self.save(job)
        return job, d / job.input_file

    def save(self, job: Job) -> None:
        with self._lock:
            job.updated = time.time()
            path = self.dir(job.id) / "job.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(job.to_dict(), indent=1))
            tmp.replace(path)

    def get(self, job_id: str) -> Job:
        with self._lock:
            path = self.dir(job_id) / "job.json"
            if not path.exists():
                raise KeyError(job_id)
            return Job.from_dict(json.loads(path.read_text()))

    def update(self, job_id: str, **changes) -> Job:
        with self._lock:
            job = self.get(job_id)
            for k, v in changes.items():
                setattr(job, k, v)
            self.save(job)
            return job

    def list(self) -> list[Job]:
        with self._lock:
            jobs = []
            for p in self.root.glob("*/job.json"):
                try:
                    jobs.append(Job.from_dict(json.loads(p.read_text())))
                except (json.JSONDecodeError, TypeError):
                    continue
            return sorted(jobs, key=lambda j: -j.created)

    def delete(self, job_id: str) -> None:
        with self._lock:
            d = self.dir(job_id)
            if not d.exists():
                raise KeyError(job_id)
            shutil.rmtree(d)

    def progress(self, job_id: str) -> dict | None:
        path = self.run_dir(job_id) / "progress.json"
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
