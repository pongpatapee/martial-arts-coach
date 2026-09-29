"""Fixed-length feature vectors around a candidate, for the learned classifiers.

Signals are sampled at fixed *time* offsets (not frame offsets) by interpolation, so a model
trained on 30 fps footage applies unchanged to 60 fps footage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparring.detection.candidates import Candidate

OFFSETS_S = (-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)
# Per-hand signals, sampled for the candidate's hand ("h_") and the other hand ("o_").
HAND_SIGNALS = (
    "elbow_deg",
    "ext",
    "wrist_x",
    "wrist_y",
    "wrist_vx",
    "wrist_vy",
    "elbow_above_shoulder",
    "wrist_below_shoulder",
    "wrist_head_dist",
)
BODY_SIGNALS = ("head_x", "head_y", "head_vx", "head_vy", "distance_tl", "shoulder_span_tl")
# Optional 3D signals (present when the pose3d stage ran).
HAND_SIGNALS_3D = ("elbow3d_deg", "ext3d")
# The opponent's reaction: a strike makes the target flinch, cover or move.
OPP_SIGNALS = ("head_vx", "head_vy", "lead_wrist_head_dist", "rear_wrist_head_dist")
OPP_OFFSETS_S = (0.0, 0.1, 0.2)


def _sample(t: np.ndarray, x: np.ndarray, at: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full(len(at), np.nan)
    return np.interp(at, t[ok], x[ok], left=np.nan, right=np.nan)


def _window_stats(t: np.ndarray, x: np.ndarray, t0: float, a: float, b: float) -> list[float]:
    m = (t >= t0 + a) & (t <= t0 + b) & np.isfinite(x)
    if not m.any():
        return [np.nan] * 4
    seg, ts = x[m], t[m]
    return [
        float(seg.max()),
        float(seg.min()),
        float(seg.max() - seg.min()),
        float(ts[seg.argmax()] - t0),
    ]


def feature_names(use_3d: bool = False, rule_features: bool = False) -> list[str]:
    names = ["is_lead"]
    hand_sig = HAND_SIGNALS + (HAND_SIGNALS_3D if use_3d else ())
    for prefix in ("h", "o"):
        for s in hand_sig:
            names += [f"{prefix}_{s}@{o:+.2f}" for o in OFFSETS_S]
    for s in BODY_SIGNALS:
        names += [f"{s}@{o:+.2f}" for o in OFFSETS_S]
    for s in ("elbow_deg", "ext", "speed") + (("elbow3d_deg", "ext3d") if use_3d else ()):
        names += [f"h_{s}_{k}" for k in ("max", "min", "range", "argmax")]
    names += ["axis_ok", "in_range"]
    names += [f"opp_{s}@{o:+.2f}" for s in OPP_SIGNALS for o in OPP_OFFSETS_S]
    names += ["opp_head_disp_max", "distance_change"]
    if rule_features:
        names += ["rule_fired", "rule_same_hand", "rule_conf"]
    return names


class WindowFeaturizer:
    """Turns (fighter feature table, candidate) into a vector; caches per-fighter arrays."""

    def __init__(
        self,
        fighter_df: pd.DataFrame,
        opponent_df: pd.DataFrame | None = None,
        use_3d: bool = False,
    ):
        self.df = fighter_df.reset_index(drop=True)
        self.opp = None
        if opponent_df is not None and not opponent_df.empty:
            o = opponent_df.reset_index(drop=True)
            self.opp_t = o.t.to_numpy()
            self.opp = {c: o[c].to_numpy(float) for c in (*OPP_SIGNALS, "head_x", "head_y")}
        self.t = self.df.t.to_numpy()
        self.use_3d = use_3d
        self.cols = {
            c: self.df[c].to_numpy(float) for c in self.df.columns if c not in ("fighter",)
        }
        for hand in ("lead", "rear"):
            self.cols[f"{hand}_speed"] = np.hypot(
                self.cols[f"{hand}_wrist_vx"], self.cols[f"{hand}_wrist_vy"]
            )

    def __call__(self, c: Candidate, rule_events=None) -> np.ndarray:
        other = "rear" if c.hand == "lead" else "lead"
        at = c.t + np.array(OFFSETS_S)
        v: list[float] = [1.0 if c.hand == "lead" else 0.0]
        hand_sig = HAND_SIGNALS + (HAND_SIGNALS_3D if self.use_3d else ())
        for hand in (c.hand, other):
            for s in hand_sig:
                v += list(_sample(self.t, self.cols[f"{hand}_{s}"], at))
        for s in BODY_SIGNALS:
            v += list(_sample(self.t, self.cols[s], at))
        for s in ("elbow_deg", "ext", "speed") + (("elbow3d_deg", "ext3d") if self.use_3d else ()):
            v += _window_stats(self.t, self.cols[f"{c.hand}_{s}"], c.t, -0.3, 0.2)
        i = int(np.clip(np.searchsorted(self.t, c.t), 0, len(self.t) - 1))
        v += [self.cols["axis_ok"][i], self.cols["in_range"][i]]
        if self.opp is not None:
            at_o = c.t + np.array(OPP_OFFSETS_S)
            for s in OPP_SIGNALS:
                v += list(_sample(self.opp_t, self.opp[s], at_o))
            m = (self.opp_t >= c.t - 0.1) & (self.opp_t <= c.t + 0.3)
            hx, hy = self.opp["head_x"][m], self.opp["head_y"][m]
            ok = np.isfinite(hx) & np.isfinite(hy)
            v.append(
                float(np.max(np.hypot(hx[ok] - hx[ok][0], hy[ok] - hy[ok][0])))
                if ok.sum() > 1
                else np.nan
            )
        else:
            v += [np.nan] * (len(OPP_SIGNALS) * len(OPP_OFFSETS_S) + 1)
        d = _sample(self.t, self.cols["distance_tl"], np.array([c.t - 0.2, c.t + 0.2]))
        v.append(float(d[1] - d[0]))
        if rule_events is not None:
            near = [e for e in rule_events if abs(e.peak_t - c.t) <= 0.15]
            v += [
                1.0 if near else 0.0,
                1.0 if any(e.evidence.get("hand") == c.hand for e in near) else 0.0,
                max((e.confidence for e in near), default=0.0),
            ]
        return np.asarray(v, np.float32)
