"""Optional 3D arm features from the RTMW3D pass (stage `pose3d`).

2D elbow angle and reach are foreshortened when a punch travels toward the camera; 3D angle and
reach are not (to the extent the model's depth estimate is right). These columns are added next
to their 2D counterparts; lead/rear mapping reuses the 2D stance estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparring.config import CleanConfig
from sparring.kinematics.clean import clean_poses
from sparring.kinematics.features import angle_deg
from sparring.skeleton import FIGHTERS, K


def add_3d_features(
    feats: pd.DataFrame, kp3d: np.ndarray, scores3d: np.ndarray, fps: float, cfg: CleanConfig
) -> pd.DataFrame:
    """kp3d: (T, 2, 17, 3) with x, y pixels and z in the same pixel scale."""
    t, nf, nk, _ = kp3d.shape
    # Clean x/y and z with the same machinery as 2D (it works per coordinate).
    xy = clean_poses(kp3d[..., :2], scores3d, fps, cfg)
    z = clean_poses(np.stack([kp3d[..., 2], np.zeros_like(kp3d[..., 2])], -1), scores3d, fps, cfg)[
        ..., 0
    ]
    p = np.concatenate([xy, z[..., None]], -1).astype(float)

    out = feats.copy()
    for fi, f in enumerate(FIGHTERS):
        rows = out.fighter == f
        lead_left = out.loc[rows, "lead_side"].to_numpy() > 0
        per_side = {}
        for side in ("l", "r"):
            s, e, w = (
                p[:, fi, K[f"{side}_shoulder"]],
                p[:, fi, K[f"{side}_elbow"]],
                p[:, fi, K[f"{side}_wrist"]],
            )
            arm = np.linalg.norm(e - s, axis=-1) + np.linalg.norm(w - e, axis=-1)
            arm_len = np.nanpercentile(arm, 95) if np.isfinite(arm).any() else np.nan
            per_side[side] = (angle_deg(s, e, w), np.linalg.norm(w - s, axis=-1) / arm_len)
        for hand, lead in (("lead", True), ("rear", False)):
            use_left = lead_left if lead else ~lead_left
            elbow = np.where(use_left, per_side["l"][0], per_side["r"][0])
            ext = np.where(use_left, per_side["l"][1], per_side["r"][1])
            out.loc[rows, f"{hand}_elbow3d_deg"] = elbow[: rows.sum()]
            out.loc[rows, f"{hand}_ext3d"] = ext[: rows.sum()]
    return out
