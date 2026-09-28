"""Video decoding and metadata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> VideoInfo:
        return cls(**json.loads(path.read_text()))


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        # Count by decoding: container frame counts are unreliable for phone footage.
        n = 0
        while cap.grab():
            n += 1
        return VideoInfo(
            path=str(path),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            n_frames=n,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def frames(path: Path | str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR image) for every frame."""
    cap = cv2.VideoCapture(str(path))
    try:
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                return
            yield i, img
            i += 1
    finally:
        cap.release()
