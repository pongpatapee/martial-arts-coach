import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sparring.labeling.server import make_handler, save_labels


@pytest.fixture
def server(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(bytes(range(256)) * 4)  # 1024 fake bytes
    labels = tmp_path / "labels" / "clip.json"
    run = tmp_path / "run"
    run.mkdir()
    (run / "video.json").write_text(json.dumps({"fps": 60.0}))
    (run / "events.json").write_text(
        json.dumps(
            [
                {
                    "fighter": "A",
                    "type": "jab",
                    "start_t": 0.9,
                    "peak_t": 1.0,
                    "end_t": 1.1,
                    "confidence": 0.8,
                    "evidence": {},
                },
                {
                    "fighter": "A",
                    "type": "combination",
                    "start_t": 0.9,
                    "peak_t": 1.0,
                    "end_t": 1.5,
                    "confidence": 0.8,
                    "evidence": {},
                },
            ]
        )
    )
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(video, labels, run))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", labels
    srv.shutdown()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
        return r.status, dict(r.headers), r.read()


def test_page_and_state(server):
    base, _ = server
    status, _, body = _get(base + "/")
    assert status == 200 and b"Sparring labeller" in body
    state = json.loads(_get(base + "/api/state")[2])
    assert state["fps"] == 60.0
    assert [p["type"] for p in state["predictions"]] == ["jab"]  # combinations hidden
    assert state["labels"]["events"] == []


def test_video_range_requests(server):
    base, _ = server
    status, headers, body = _get(base + "/video", {"Range": "bytes=10-19"})
    assert (
        status == 206
        and headers["Content-Range"] == "bytes 10-19/1024"
        and body == bytes(range(10, 20))
    )
    status, _, body = _get(base + "/video", {"Range": "bytes=-4"})
    assert status == 206 and body == bytes([252, 253, 254, 255])


def test_save_roundtrip_and_validation(server):
    base, labels = server
    data = {
        "events": [
            {"fighter": "B", "type": "cross", "t": 2.3456},
            {"fighter": "A", "type": "punch", "t": 1.0},
        ]
    }
    req = urllib.request.Request(base + "/api/save", data=json.dumps(data).encode(), method="POST")
    assert json.loads(urllib.request.urlopen(req).read())["ok"]
    saved = json.loads(labels.read_text())
    assert [(e["fighter"], e["t"], e["confidence"]) for e in saved["events"]] == [
        ("A", 1.0, "high"),
        ("B", 2.346, "high"),
    ]
    with pytest.raises(ValueError):
        save_labels(labels, {"events": [{"fighter": "C", "type": "jab", "t": 1}]})
    save_labels(labels, {"events": []})
    assert labels.with_suffix(".json.bak").exists()
