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


def auto_device() -> str:
    """ONNX Runtime device for rtmlib: CUDA when the GPU build is installed and a GPU is present."""
    import onnxruntime as ort
    import torch

    if "CUDAExecutionProvider" in ort.get_available_providers() and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_device(cfg: PoseConfig) -> str:
    device = cfg.device or auto_device()
    if cfg.backend == "onnxruntime" and device.startswith("cuda"):
        import onnxruntime as ort

        # The CUDA provider needs the CUDA and cuDNN libraries that ship as pip packages
        # alongside torch; ONNX Runtime does not search for them on its own.
        ort.preload_dlls()
    return device


class RTMPoseEstimator:
    def __init__(self, cfg: PoseConfig):
        from rtmlib import RTMPose

        self._model = RTMPose(
            cfg.model_url,
            model_input_size=tuple(cfg.input_size),
            backend=cfg.backend,
            device=_resolve_device(cfg),
        )

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kpts, scores = self._model(image, bboxes=boxes.tolist())
        return kpts.astype(np.float32), scores.astype(np.float32)


class RTMPose3DEstimator:
    """RTMW3D whole-body 3D pose. Returns the 17 COCO body keypoints (same order and names as the
    first 17 Halpe-26 keypoints) with x, y in image pixels and z as root-relative depth converted
    to the same pixel scale, so 3D angles and lengths are meaningful."""

    N_BODY = 17

    def __init__(self, cfg: PoseConfig):
        from rtmlib import RTMPose3d

        self._input = tuple(cfg.input_size_3d)
        self._model = RTMPose3d(
            cfg.model_url_3d,
            model_input_size=self._input,
            backend=cfg.backend,
            device=_resolve_device(cfg),
        )

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kpts3d, scores, _, kpts2d = self._model(image, bboxes=boxes.tolist())
        out = np.zeros((len(boxes), self.N_BODY, 3), np.float32)
        out[..., :2] = kpts2d[:, : self.N_BODY]
        # z is in [-z_range, z_range] for a crop spanning the model input height; the crop spans
        # 1.25x the box height in the image (rtmlib's padding).
        crop_h = (boxes[:, 3] - boxes[:, 1]) * 1.25
        z = kpts3d[:, : self.N_BODY, 2] / self._model.z_range
        out[..., 2] = z * (crop_h[:, None] / 2)
        return out, scores[:, : self.N_BODY].astype(np.float32)


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
