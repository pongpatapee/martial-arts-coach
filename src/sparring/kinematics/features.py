"""Per-frame, per-fighter feature table: the single input to all event detectors.

All positions are in the body frame (see normalize.py): origin at the hip midpoint, +x toward
the opponent, +y up, units of torso length (TL). Hands are named lead/rear (derived from stance),
not left/right, so a jab is always a lead-hand straight regardless of stance. Legs likewise: the
lead leg is the one nearer the opponent, so a "lead teep" means the same thing in either stance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparring.config import KinematicsConfig
from sparring.kinematics.normalize import BodyFrame, body_frames, hysteresis_sign, rolling
from sparring.skeleton import FIGHTERS, K

HANDS = ("lead", "rear")
LEGS = ("lead", "rear")


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees, (T, 2) inputs."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def derivative(x: np.ndarray, fps: float) -> np.ndarray:
    if np.isfinite(x).sum() < 2:
        return np.full_like(x, np.nan)
    return np.gradient(x) * fps


def _pick(side: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per-frame choose between left and right series; side is +1 for left, -1 for right."""
    mask = side > 0
    if left.ndim == 2:
        mask = mask[:, None]
    return np.where(mask, left, right)


def arm_length_tl(bp: dict[str, np.ndarray], side: str) -> float:
    """Arm length (shoulder-elbow-wrist) in TL. The 95th percentile over the clip approximates the
    true length, since 2D projection only ever shortens limbs."""
    upper = np.linalg.norm(bp[f"{side}_elbow"] - bp[f"{side}_shoulder"], axis=-1)
    fore = np.linalg.norm(bp[f"{side}_wrist"] - bp[f"{side}_elbow"], axis=-1)
    total = upper + fore
    v = np.nanpercentile(total, 95) if np.isfinite(total).any() else np.nan
    return float(v) if np.isfinite(v) and v > 0.3 else 1.4


def fighter_features(
    kp: np.ndarray,
    frame: BodyFrame,
    opp_frame: BodyFrame,
    fps: float,
    cfg: KinematicsConfig,
    opp_kp: np.ndarray | None = None,
) -> pd.DataFrame:
    """kp / opp_kp: (T, K, 2) cleaned pixels for this fighter and the opponent."""
    t = len(kp)
    bp = {name: frame.to_body(kp[:, i]) for name, i in K.items()}
    out: dict[str, np.ndarray] = {}

    visible = np.isfinite(bp["l_shoulder"][:, 0] + bp["r_shoulder"][:, 0] + frame.origin[:, 0])
    out["visible"] = visible
    out["torso_px"] = frame.torso_px
    out["facing"] = frame.facing
    out["separation_tl"] = frame.separation_tl
    out["axis_ok"] = frame.separation_tl >= cfg.min_axis_separation_tl

    # Inter-fighter distance (hip to hip), in the mean of both torso lengths.
    d_px = np.linalg.norm(frame.origin - opp_frame.origin, axis=-1)
    dist = d_px / np.nanmean(np.stack([frame.torso_px, opp_frame.torso_px]), axis=0)
    out["distance_tl"] = dist
    out["in_range"] = dist <= cfg.in_range_tl

    # Stance: which ankle is further forward. Positive = left foot leads = orthodox.
    ankle_diff = bp["l_ankle"][:, 0] - bp["r_ankle"][:, 0]
    ankle_diff = np.where(out["axis_ok"], ankle_diff, np.nan)
    ankle_diff = rolling(ankle_diff, int(cfg.stance_window_s * fps), "median")
    lead_side = hysteresis_sign(ankle_diff, threshold=0.12, initial=1.0)
    out["lead_side"] = lead_side  # +1 = left (orthodox), -1 = right (southpaw)

    # Head: mean of available face keypoints.
    face = np.stack([bp[n] for n in ("nose", "l_eye", "r_eye", "l_ear", "r_ear")])
    head = np.nanmean(face, axis=0) if np.isfinite(face).any() else face[0]
    out["head_x"], out["head_y"] = head[:, 0], head[:, 1]
    out["head_vx"] = derivative(head[:, 0], fps)
    out["head_vy"] = derivative(head[:, 1], fps)
    ears_y = np.nanmean(np.stack([bp["l_ear"][:, 1], bp["r_ear"][:, 1]]), axis=0)
    out["chin_up"] = bp["nose"][:, 1] - ears_y  # > 0: nose above ears, head tilted back
    nose_y = bp["nose"][:, 1]

    # Hands.
    for hand in HANDS:
        side = lead_side if hand == "lead" else -lead_side
        shoulder = _pick(side, bp["l_shoulder"], bp["r_shoulder"])
        elbow = _pick(side, bp["l_elbow"], bp["r_elbow"])
        wrist = _pick(side, bp["l_wrist"], bp["r_wrist"])
        arm = _pick(side, np.full(t, arm_length_tl(bp, "l")), np.full(t, arm_length_tl(bp, "r")))
        ext = np.linalg.norm(wrist - shoulder, axis=-1) / arm
        out[f"{hand}_wrist_x"], out[f"{hand}_wrist_y"] = wrist[:, 0], wrist[:, 1]
        out[f"{hand}_wrist_vx"] = derivative(wrist[:, 0], fps)
        out[f"{hand}_wrist_vy"] = derivative(wrist[:, 1], fps)
        out[f"{hand}_ext"] = ext
        out[f"{hand}_ext_vel"] = derivative(ext, fps)
        out[f"{hand}_elbow_deg"] = angle_deg(shoulder, elbow, wrist)
        out[f"{hand}_elbow_above_shoulder"] = elbow[:, 1] - shoulder[:, 1]
        out[f"{hand}_wrist_below_nose"] = nose_y - wrist[:, 1]
        out[f"{hand}_wrist_below_shoulder"] = shoulder[:, 1] - wrist[:, 1]
        out[f"{hand}_wrist_head_dist"] = np.linalg.norm(wrist - head, axis=-1)

    lead_ankle = _pick(lead_side, bp["l_ankle"], bp["r_ankle"])
    rear_ankle = _pick(lead_side, bp["r_ankle"], bp["l_ankle"])
    out["stance_width_tl"] = lead_ankle[:, 0] - rear_ankle[:, 0]
    out["head_past_lead_foot"] = head[:, 0] - lead_ankle[:, 0]
    out["shoulder_span_tl"] = np.abs(bp["l_shoulder"][:, 0] - bp["r_shoulder"][:, 0])
    knees = np.stack(
        [
            angle_deg(bp["l_hip"], bp["l_knee"], bp["l_ankle"]),
            angle_deg(bp["r_hip"], bp["r_knee"], bp["r_ankle"]),
        ]
    )
    out["knee_deg"] = np.nanmean(knees, axis=0) if np.isfinite(knees).any() else knees[0]
    # Hip height above the feet: drops when the fighter sits down into a duck.
    out["hip_height_tl"] = -np.nanmean(np.stack([lead_ankle[:, 1], rear_ankle[:, 1]]), axis=0)

    # Legs (kicks, knees, checks). Heights are relative to the hip midpoint (0) in TL.
    for leg in LEGS:
        side = lead_side if leg == "lead" else -lead_side
        hip = _pick(side, bp["l_hip"], bp["r_hip"])
        knee = _pick(side, bp["l_knee"], bp["r_knee"])
        ankle = _pick(side, bp["l_ankle"], bp["r_ankle"])
        out[f"{leg}_ankle_x"], out[f"{leg}_ankle_y"] = ankle[:, 0], ankle[:, 1]
        out[f"{leg}_ankle_vx"] = derivative(ankle[:, 0], fps)
        out[f"{leg}_ankle_vy"] = derivative(ankle[:, 1], fps)
        out[f"{leg}_knee_x"], out[f"{leg}_knee_y"] = knee[:, 0], knee[:, 1]
        out[f"{leg}_knee_vx"] = derivative(knee[:, 0], fps)
        out[f"{leg}_knee_vy"] = derivative(knee[:, 1], fps)
        out[f"{leg}_knee_deg"] = angle_deg(hip, knee, ankle)

    # Elbow-strike geometry: elbow position/speed relative to the head, for both arms.
    for hand in HANDS:
        side = lead_side if hand == "lead" else -lead_side
        elbow = _pick(side, bp["l_elbow"], bp["r_elbow"])
        out[f"{hand}_elbow_x"], out[f"{hand}_elbow_y"] = elbow[:, 0], elbow[:, 1]
        out[f"{hand}_elbow_speed"] = np.hypot(
            derivative(elbow[:, 0], fps), derivative(elbow[:, 1], fps)
        )

    # Clinch: how many of this fighter's wrists are on the opponent's head/neck (image space,
    # normalised by this fighter's torso length).
    if opp_kp is not None:
        target = np.nanmean(np.stack([opp_kp[:, K["neck"]], opp_kp[:, K["nose"]]]), axis=0)
        on = [
            np.linalg.norm(kp[:, K[w]] - target, axis=-1) / frame.torso_px <= cfg.clinch_hand_tl
            for w in ("l_wrist", "r_wrist")
        ]
        out["hands_on_opponent"] = on[0].astype(float) + on[1].astype(float)
    else:
        out["hands_on_opponent"] = np.zeros(t)

    df = pd.DataFrame(out)
    df.insert(0, "frame", np.arange(t))
    df.insert(1, "t", np.arange(t) / fps)
    return df


def compute_features(kp: np.ndarray, fps: float, cfg: KinematicsConfig) -> pd.DataFrame:
    """kp: (T, 2, K, 2) cleaned pixels -> long table with one row per (frame, fighter)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frames = body_frames(kp, fps, cfg.torso_window_s)
            parts = []
            for fi, f in enumerate(FIGHTERS):
                df = fighter_features(
                    kp[:, fi], frames[fi], frames[1 - fi], fps, cfg, opp_kp=kp[:, 1 - fi]
                )
                df.insert(2, "fighter", f)
                parts.append(df)
    return pd.concat(parts, ignore_index=True)
