"""Person detection + multi-object tracking.

`PersonTracker` is the seam: the rest of the pipeline only sees `FrameDetections`. The default
implementation uses Ultralytics YOLO with BoT-SORT (camera-motion compensated, which matters for
hand-held phone footage). Note that Ultralytics is AGPL-3.0; an Apache-2.0 alternative would be
RTMDet/YOLOX from `rtmlib` plus a standalone tracker behind this same protocol.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from sparring.config import DetectConfig


@dataclass
class FrameDetections:
    frame: int
    image: np.ndarray  # BGR, only valid during iteration
    boxes: np.ndarray  # (N, 4) xyxy in original pixels
    track_ids: np.ndarray  # (N,) int, -1 when the tracker has not confirmed a track
    confs: np.ndarray  # (N,)


class PersonTracker(Protocol):
    def track(self, video: Path) -> Iterator[FrameDetections]: ...


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class UltralyticsTracker:
    def __init__(self, cfg: DetectConfig, work_dir: Path):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.device = cfg.device or auto_device()
        self.tracker_yaml = work_dir / "botsort.yaml"
        self.tracker_yaml.write_text(
            yaml.safe_dump(
                {
                    "tracker_type": "botsort",
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.1,
                    "new_track_thresh": 0.25,
                    "track_buffer": cfg.track_buffer,
                    "match_thresh": 0.8,
                    "fuse_score": True,
                    "gmc_method": "sparseOptFlow",
                    "proximity_thresh": 0.5,
                    "appearance_thresh": 0.8,
                    "with_reid": False,
                    "model": "auto",
                }
            )
        )

    def track(self, video: Path) -> Iterator[FrameDetections]:
        results = self.model.track(
            source=str(video),
            stream=True,
            persist=True,
            classes=[0],
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            device=self.device,
            tracker=str(self.tracker_yaml),
            verbose=False,
        )
        for i, r in enumerate(results):
            b = r.boxes
            n = len(b)
            ids = b.id.cpu().numpy().astype(int) if b.id is not None else np.full(n, -1)
            yield FrameDetections(
                frame=i,
                image=r.orig_img,
                boxes=b.xyxy.cpu().numpy().astype(np.float32),
                track_ids=ids,
                confs=b.conf.cpu().numpy().astype(np.float32),
            )
