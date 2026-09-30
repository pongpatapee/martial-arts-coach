"""Web API tests with a fake pipeline (no models, no video)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sparring.web.app import create_app
from sparring.web.jobs import Job, JobStore


class FakeRunner:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple[str, str, str]] = []

    def prepare(self, job: Job, store: JobStore) -> dict:
        d = store.dir(job.id)
        (d / "playback.mp4").write_bytes(bytes(range(256)) * 8)
        (d / "thumb.jpg").write_bytes(b"\xff\xd8thumb")
        return {"fps": 30.0, "width": 1080, "height": 1920, "duration_s": 2.0}

    def analyze(self, job: Job, store: JobStore, from_stage: str) -> None:
        self.calls.append((job.id, from_stage, job.options.discipline))
        if self.fail:
            (store.dir(job.id) / "run.log").write_text("loading models\nTraceback: boom\n")
            raise RuntimeError("analysis exited with code 1")
        run = store.run_dir(job.id)
        run.mkdir(exist_ok=True)
        (run / "video.json").write_text(
            json.dumps({"fps": 30.0, "n_frames": 60, "width": 1080, "height": 1920, "path": "x"})
        )
        (run / "events.json").write_text(
            json.dumps(
                [
                    {
                        "fighter": "A",
                        "type": "jab",
                        "start_t": 0.9,
                        "peak_t": 1.0,
                        "end_t": 1.1,
                        "confidence": 0.9,
                        "evidence": {},
                    },
                ]
            )
        )
        (run / "report.json").write_text(
            json.dumps({"discipline": job.options.discipline, "fighters": {}})
        )
        (run / "fighters.json").write_text(json.dumps({"seed_tracks": {"A": 1, "B": 2}}))
        kp = np.full((60, 2, 26, 2), np.nan, np.float32)
        kp[:, 0, 0] = [100.0, 200.0]
        np.save(run / "poses_clean.npy", kp)
        (run / "progress.json").write_text(json.dumps({"stage": "done", "fraction": 1.0}))


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(runner=None, **kw):
        app = create_app(tmp_path / "data", runner=runner or FakeRunner(), **kw)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c, app

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def _upload(client, name="spar.mp4", data=b"fake video bytes", **form):
    return client.post("/api/jobs", files={"file": (name, data, "video/mp4")}, data=form)


def _finish(app):
    app.state.worker.wait_idle()


def test_options_come_from_the_engine(make_client):
    client, _ = make_client()
    o = client.get("/api/options").json()
    assert {d["id"] for d in o["disciplines"]} == {"boxing", "kickboxing", "muay_thai"}
    assert o["methods"][0]["id"] == "ensemble" and o["methods"][0]["help"]
    assert o["moves"]["jab"]["kind"] == "strike"


def test_upload_process_and_fetch_results(make_client):
    client, app = make_client()
    r = _upload(client, discipline="muay_thai", method="rules")
    assert r.status_code == 201
    job_id = r.json()["id"]
    _finish(app)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert (
        job["status"] == "done"
        and job["fps"] == 30.0
        and job["options"]["discipline"] == "muay_thai"
    )
    a = client.get(f"/api/jobs/{job_id}/analysis").json()
    assert a["events"][0]["type"] == "jab" and a["report"]["discipline"] == "muay_thai"
    p = client.get(f"/api/jobs/{job_id}/poses")
    assert p.headers["X-Shape"] == "60,2,26,2"
    q = np.frombuffer(p.content, "<i2").reshape(60, 2, 26, 2)
    assert q[0, 0, 0].tolist() == [100, 200] and q[0, 1, 0, 0] == -1
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job_id]
    assert client.get(f"/api/jobs/{job_id}/thumbnail").status_code == 200


def test_video_supports_range_requests(make_client):
    client, app = make_client()
    job_id = _upload(client).json()["id"]
    _finish(app)
    r = client.get(f"/api/jobs/{job_id}/video", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))


def test_rejects_bad_uploads(make_client):
    client, _ = make_client(max_upload_mb=1)
    assert _upload(client, name="notes.txt").status_code == 415
    assert _upload(client, data=b"").status_code == 422
    assert _upload(client, discipline="jousting").status_code == 422
    assert _upload(client, fighters="three").status_code == 422
    assert _upload(client, data=b"x" * (1024 * 1024 + 1)).status_code == 413
    assert client.get("/api/jobs").json() == []  # nothing left behind


def test_failure_is_recorded_and_retry_works(make_client):
    runner = FakeRunner(fail=True)
    client, app = make_client(runner)
    job_id = _upload(client).json()["id"]
    _finish(app)
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and "code 1" in job["error"]
    assert job["log_tail"][-1] == "Traceback: boom"
    assert client.get(f"/api/jobs/{job_id}/analysis").status_code == 409
    runner.fail = False
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 200
    _finish(app)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "done"


def test_reanalyze_reruns_only_the_cheap_stages(make_client):
    runner = FakeRunner()
    client, app = make_client(runner)
    job_id = _upload(client).json()["id"]
    _finish(app)
    r = client.post(
        f"/api/jobs/{job_id}/reanalyze", json={"discipline": "kickboxing", "method": "rules"}
    )
    assert r.status_code == 200 and r.json()["options"]["discipline"] == "kickboxing"
    assert runner.calls[-1] == (job_id, "events", "kickboxing")
    assert client.get(f"/api/jobs/{job_id}/analysis").json()["report"]["discipline"] == "kickboxing"


def test_reanalyze_failure_keeps_previous_options(make_client):
    runner = FakeRunner()
    client, app = make_client(runner)
    job_id = _upload(client).json()["id"]
    _finish(app)
    runner.fail = True
    r = client.post(
        f"/api/jobs/{job_id}/reanalyze", json={"discipline": "kickboxing", "method": "rules"}
    )
    assert r.status_code == 500
    assert client.get(f"/api/jobs/{job_id}").json()["options"]["discipline"] == "boxing"


def test_delete_and_unknown_ids(make_client):
    client, app = make_client()
    job_id = _upload(client).json()["id"]
    _finish(app)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get("/api/jobs/../../etc").status_code == 404


def test_interrupted_jobs_resume_after_restart(tmp_path):
    store = JobStore(tmp_path / "data")
    from sparring.web.jobs import JobOptions

    job, dest = store.create("a.mp4", ".mp4", JobOptions())
    dest.write_bytes(b"video")
    store.update(job.id, status="running")  # the server died mid-job
    app = create_app(tmp_path / "data", runner=FakeRunner())
    with TestClient(app) as client:
        app.state.worker.wait_idle()
        assert client.get(f"/api/jobs/{job.id}").json()["status"] == "done"


def test_progress_file_is_written_by_the_pipeline(tmp_path):
    from sparring.pipeline import Progress

    p = Progress(tmp_path / "progress.json", ["track", "pose"])
    p.start("pose")
    p.update(0.5)  # throttled right after start
    d = json.loads((tmp_path / "progress.json").read_text())
    assert d["stage"] == "pose" and d["stage_index"] == 1 and d["stages"] == ["track", "pose"]
    p.finish()
    assert json.loads((tmp_path / "progress.json").read_text())["stage"] == "done"
