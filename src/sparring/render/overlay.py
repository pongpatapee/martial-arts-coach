"""Annotated video: skeletons, fighter labels, lead/rear hands, event labels and a timeline strip."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sparring.events.base import Event
from sparring.io.video import VideoInfo, frames
from sparring.skeleton import BONES, FIGHTERS, K

log = logging.getLogger(__name__)

FIGHTER_BGR = {"A": (0, 215, 255), "B": (255, 0, 200)}
KIND_BGR = {
    "strike": (60, 220, 60),
    "defense": (255, 200, 0),
    "fault": (40, 40, 240),
    "other": (200, 200, 200),
}
LABEL_HOLD_S = 0.5


def _pt(p: np.ndarray) -> tuple[int, int] | None:
    return (int(p[0]), int(p[1])) if np.isfinite(p).all() else None


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color, thick: int) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _timeline(events: list[Event], width: int, duration: float, row_h: int) -> np.ndarray:
    strip = np.full((row_h * 2, width, 3), 25, np.uint8)
    for e in events:
        if e.type == "combination":
            continue
        row = FIGHTERS.index(e.fighter)
        x1 = int(e.start_t / duration * width)
        x2 = max(x1 + 2, int(e.end_t / duration * width))
        y1 = row * row_h + (4 if e.kind == "strike" else row_h // 2)
        cv2.rectangle(strip, (x1, y1), (x2, y1 + row_h // 2 - 6), KIND_BGR[e.kind], -1)
    for i, f in enumerate(FIGHTERS):
        _text(strip, f, (6, i * row_h + row_h - 10), 0.9, FIGHTER_BGR[f], 2)
    return strip


def render_overlay(
    video: Path,
    out: Path,
    keypoints: np.ndarray,
    features: pd.DataFrame,
    events: list[Event],
    info: VideoInfo,
) -> None:
    w, h, fps = info.width, info.height, info.fps
    scale = h / 1080
    row_h = int(40 * scale)
    strip = _timeline(events, w, info.duration_s, row_h)
    total_h = h + strip.shape[0]
    tmp = out.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, total_h))
    lead = {f: features[features.fighter == f].set_index("frame").lead_side for f in FIGHTERS}
    shown = [e for e in events if e.type != "combination"]

    for i, img in frames(video):
        if i >= len(keypoints):
            break
        t = i / fps
        for fi, f in enumerate(FIGHTERS):
            kp = keypoints[i, fi]
            col = FIGHTER_BGR[f]
            for a, b in BONES:
                pa, pb = _pt(kp[K[a]]), _pt(kp[K[b]])
                if pa and pb:
                    cv2.line(img, pa, pb, col, max(2, int(3 * scale)), cv2.LINE_AA)
            side = lead[f].get(i, 1.0)
            lead_w = kp[K["l_wrist"] if side > 0 else K["r_wrist"]]
            rear_w = kp[K["r_wrist"] if side > 0 else K["l_wrist"]]
            for p, lab in ((lead_w, "L"), (rear_w, "R")):
                q = _pt(p)
                if q:
                    cv2.circle(img, q, int(10 * scale), col, -1)
                    _text(img, lab, (q[0] + int(12 * scale), q[1]), 0.6 * scale, (255, 255, 255), 2)
            head = _pt(kp[K["head"]]) or _pt(kp[K["nose"]])
            if head is None:
                continue
            x, y = head[0] - int(40 * scale), head[1] - int(30 * scale)
            stance = "orthodox" if side > 0 else "southpaw"
            _text(img, f"{f} ({stance})", (x, y), 0.8 * scale, col, 2)
            active = [
                e for e in shown if e.fighter == f and e.start_t <= t <= e.end_t + LABEL_HOLD_S
            ]
            for j, e in enumerate(active[-3:]):
                _text(
                    img,
                    e.type.replace("_", " "),
                    (x, y - int((j + 1) * 32 * scale)),
                    0.9 * scale,
                    KIND_BGR[e.kind],
                    2,
                )
        s = strip.copy()
        px = int(t / info.duration_s * w)
        cv2.line(s, (px, 0), (px, s.shape[0]), (255, 255, 255), 2)
        writer.write(np.vstack([img, s]))
    writer.release()

    if shutil.which("ffmpeg"):
        # Re-encode to H.264 for browser/QuickTime playback and keep the original audio.
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp),
            "-i",
            str(video),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        tmp.unlink()
    else:
        log.warning("ffmpeg not found; overlay left as mp4v (may not play in browsers)")
        tmp.rename(out)
