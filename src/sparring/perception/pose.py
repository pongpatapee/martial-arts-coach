"""Top-down 2D pose estimation for the two selected fighters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from sparring.config import PoseConfig
from sparring.skeleton import FIGHTERS, N_KEYPOINTS


class PoseEstimator(Protocol):
    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return keypoints (N, K, 2) in image pixels and scores (N, K) for N xyxy boxes."""
        ...


class RTMPoseEstimator:
    def __init__(self, cfg: PoseConfig):
        from rtmlib import RTMPose

        self._model = RTMPose(
            cfg.model_url,
            model_input_size=tuple(cfg.input_size),
            backend=cfg.backend,
            device=cfg.device,
        )

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kpts, scores = self._model(image, bboxes=boxes.tolist())
        return kpts.astype(np.float32), scores.astype(np.float32)


@dataclass
class PoseTrack:
    """Dense per-frame poses for both fighters; NaN / 0 where a fighter was not detected."""

    keypoints: np.ndarray  # (T, 2, K, 2) pixels
    scores: np.ndarray  # (T, 2, K)
    boxes: np.ndarray  # (T, 2, 4)

    @classmethod
    def empty(cls, n_frames: int) -> PoseTrack:
        return cls(
            keypoints=np.full((n_frames, len(FIGHTERS), N_KEYPOINTS, 2), np.nan, np.float32),
            scores=np.zeros((n_frames, len(FIGHTERS), N_KEYPOINTS), np.float32),
            boxes=np.full((n_frames, len(FIGHTERS), 4), np.nan, np.float32),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(path, keypoints=self.keypoints, scores=self.scores, boxes=self.boxes)

    @classmethod
    def load(cls, path: Path) -> PoseTrack:
        d = np.load(path)
        return cls(keypoints=d["keypoints"], scores=d["scores"], boxes=d["boxes"])
