"""Local labelling tool: `sparring label CLIP` (or `sparring label VIDEO --labels x.json [--run RUN]`).

Serves a single page that plays the video with frame stepping and one-key event labelling,
shows the detector's predictions (from the run's events.json) as suggestions to accept or reject,
and saves straight back to the labels file (keeping a .bak of the previous version).

Stdlib only. The video is served with HTTP Range support, which browsers need for seeking.
"""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from sparring.moves import MOVES


def _page() -> bytes:
    return (resources.files("sparring") / "labeling" / "index.html").read_bytes()


def load_labels(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"segment": [0.0, None], "ignore": [], "events": []}


def save_labels(path: Path, data: dict) -> None:
    """Validate and write labels, keeping the previous file as .bak."""
    allowed = set(MOVES) | {"punch", "kick", "strike"}
    for e in data.get("events", []):
        if e.get("fighter") not in ("A", "B") or e.get("type") not in allowed:
            raise ValueError(f"invalid label: {e}")
        e["t"] = round(float(e["t"]), 3)
        e.setdefault("confidence", "high")
    data["events"].sort(key=lambda e: (e["t"], e["fighter"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(json.dumps(data, indent=1))


def make_handler(video: Path, labels: Path, run: Path | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(_page(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                info = {}
                if run and (run / "video.json").exists():
                    info = json.loads((run / "video.json").read_text())
                events = []
                if run and (run / "events.json").exists():
                    events = [
                        e
                        for e in json.loads((run / "events.json").read_text())
                        if e["type"] != "combination"
                    ]
                self._json(
                    {
                        "labels": load_labels(labels),
                        "predictions": events,
                        "fps": info.get("fps", 30.0),
                        "labels_path": str(labels),
                        "moves": {k: {"kind": m.kind, "label": m.label} for k, m in MOVES.items()},
                    }
                )
            elif self.path.startswith("/video"):
                self._video()
            else:
                self._send(b"not found", "text/plain", 404)

        def _video(self) -> None:
            size = video.stat().st_size
            ctype = mimetypes.guess_type(video.name)[0] or "video/mp4"
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and (m := re.match(r"bytes=(\d*)-(\d*)", rng)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:  # suffix range: the last N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with video.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_POST(self) -> None:
            if self.path != "/api/save":
                self._send(b"not found", "text/plain", 404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                save_labels(labels, json.loads(body))
            except (ValueError, KeyError, TypeError) as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            self._json({"ok": True})

    return Handler


def serve(
    video: Path, labels: Path, run: Path | None, port: int = 8765, open_browser: bool = True
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(video, labels, run))
    url = f"http://127.0.0.1:{port}/"
    print(f"labelling {video} -> {labels}\nopen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
