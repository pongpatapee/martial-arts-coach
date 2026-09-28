"""Body-relative coordinate frames.

For every fighter and frame we express keypoints in a frame where:
- the origin is the fighter's hip midpoint,
- +x points horizontally toward the opponent, +y points up,
- the unit is one torso length (TL).

This cancels camera pan (hand-held footage), distance to camera, and which side of the screen
each fighter is on, so thresholds mean the same thing for everyone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sparring.skeleton import K


def midpoint(kp: np.ndarray, a: str, b: str) -> np.ndarray:
    return (kp[..., K[a], :] + kp[..., K[b], :]) / 2


def rolling(x: np.ndarray, window: int, fn: str, q: float | None = None) -> np.ndarray:
    s = pd.Series(x)
    r = s.rolling(max(window, 1), center=True, min_periods=1)
    return (r.quantile(q) if fn == "quantile" else getattr(r, fn)()).to_numpy()


def hysteresis_sign(x: np.ndarray, threshold: float, initial: float = 1.0) -> np.ndarray:
    """+1/-1 series that only flips when |x| exceeds threshold with the opposite sign.
    NaN or small values hold the previous state."""
    out = np.empty(len(x))
    state = initial
    first = np.flatnonzero(np.abs(np.nan_to_num(x)) > threshold)
    if len(first):
        state = float(np.sign(x[first[0]]))
    for i, v in enumerate(x):
        if np.isfinite(v) and abs(v) > threshold:
            state = float(np.sign(v))
        out[i] = state
    return out


@dataclass
class BodyFrame:
    origin: np.ndarray  # (T, 2) hip midpoint, pixels
    torso_px: np.ndarray  # (T,) torso length in pixels (smoothed)
    facing: np.ndarray  # (T,) +1 if opponent is to the right in the image, -1 if left
    separation_tl: np.ndarray  # (T,) horizontal hip-to-hip distance to opponent, in TL

    def to_body(self, p: np.ndarray) -> np.ndarray:
        """(T, 2) pixel points -> (T, 2) body frame (x forward, y up, TL units)."""
        rel = (p - self.origin) / self.torso_px[:, None]
        return np.stack([rel[:, 0] * self.facing, -rel[:, 1]], axis=1)


def torso_length_px(kp: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Robust torso length. Uses a high rolling quantile because bending and turning only ever
    make the projected torso look shorter."""
    raw = np.linalg.norm(
        midpoint(kp, "l_shoulder", "r_shoulder") - midpoint(kp, "l_hip", "r_hip"), axis=-1
    )
    return rolling(raw, int(window_s * fps), "quantile", 0.8)


def body_frames(kp: np.ndarray, fps: float, torso_window_s: float) -> list[BodyFrame]:
    """kp: (T, 2, K, 2) cleaned pixels for both fighters."""
    origins = [midpoint(kp[:, f], "l_hip", "r_hip") for f in range(2)]
    torsos = [torso_length_px(kp[:, f], fps, torso_window_s) for f in range(2)]
    frames = []
    for f in range(2):
        o = 1 - f
        dx = origins[o][:, 0] - origins[f][:, 0]
        mean_tl = np.nanmean(np.stack([torsos[f], torsos[o]]), axis=0)
        sep = dx / mean_tl
        # Default facing: A is seeded as the left fighter, so faces right.
        facing = hysteresis_sign(sep, threshold=0.3, initial=1.0 if f == 0 else -1.0)
        frames.append(BodyFrame(origins[f], torsos[f], facing, np.abs(sep)))
    return frames
