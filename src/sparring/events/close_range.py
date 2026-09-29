"""Close-range weapons (muay thai): knees, elbows and the clinch.

- Knee strike: the knee drives up to around hip height with the leg folded, toward the opponent,
  at close range. A knee that then extends into a push is a teep chamber, not a knee strike.
- Elbow strike: at close range, a fast elbow with the arm folded tight and the elbow at about
  shoulder height (horizontal, uppercut and spinning elbows all share this).
- Clinch: the fighters are chest to chest and this fighter has at least one hand on the
  opponent's head or neck, sustained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sparring.config import CloseRangeConfig
from sparring.events.base import Event
from sparring.events.faults import close_gaps, runs
from sparring.events.punches import _interp
from sparring.kinematics.features import HANDS, LEGS
from sparring.moves import strike_name


def detect_knees(df: pd.DataFrame, fighter: str, fps: float, cfg: CloseRangeConfig) -> list[Event]:
    df = df.reset_index(drop=True)
    t = df.t.to_numpy()
    dist = df.distance_tl.to_numpy()
    out = []
    for leg in LEGS:
        ky = _interp(df[f"{leg}_knee_y"].to_numpy())
        kvx = _interp(df[f"{leg}_knee_vx"].to_numpy())
        kdeg = _interp(df[f"{leg}_knee_deg"].to_numpy())
        valid = np.isfinite(df[f"{leg}_knee_y"].to_numpy()) & df.visible.to_numpy()
        peaks, _ = find_peaks(ky, prominence=cfg.knee_min_rise_tl, distance=max(1, int(0.3 * fps)))
        look = max(1, int(0.3 * fps))
        for j in peaks:
            if not valid[j] or ky[j] < cfg.knee_min_height_tl or kdeg[j] > cfg.knee_max_fold_deg:
                continue
            if not np.isfinite(dist[j]) or dist[j] > cfg.max_distance_tl:
                continue
            a = max(0, j - look)
            if kvx[a : j + 1].max() < cfg.knee_min_forward_speed_tl:
                continue
            # A teep chambers the knee and then extends the leg *while the knee is still up*;
            # that is not a knee strike (straightening on the way back down doesn't count).
            after = slice(j, min(len(t), j + look))
            high = ky[after] >= ky[j] - 0.3
            if high.any() and kdeg[after][high].max() >= cfg.teep_extension_deg:
                continue
            out.append(
                Event(
                    fighter=fighter,
                    type=strike_name("knee", "knee", leg),
                    start_t=float(t[a]),
                    peak_t=float(t[j]),
                    end_t=float(t[min(len(t) - 1, j + look)]),
                    confidence=0.7,
                    evidence={
                        "leg": leg,
                        "knee_height_tl": float(ky[j]),
                        "knee_deg": float(kdeg[j]),
                        "in_range": True,
                    },
                )
            )
    return sorted(out, key=lambda e: e.peak_t)


def detect_elbows(df: pd.DataFrame, fighter: str, fps: float, cfg: CloseRangeConfig) -> list[Event]:
    df = df.reset_index(drop=True)
    t = df.t.to_numpy()
    dist = df.distance_tl.to_numpy()
    out = []
    for hand in HANDS:
        sp = _interp(df[f"{hand}_elbow_speed"].to_numpy())
        fold = _interp(df[f"{hand}_elbow_deg"].to_numpy())
        height = _interp(df[f"{hand}_elbow_above_shoulder"].to_numpy())
        lead = _interp(df[f"{hand}_elbow_x"].to_numpy() - df[f"{hand}_wrist_x"].to_numpy())
        valid = np.isfinite(df[f"{hand}_elbow_deg"].to_numpy()) & df.visible.to_numpy()
        peaks, _ = find_peaks(sp, height=cfg.elbow_min_speed_tl, distance=max(1, int(0.25 * fps)))
        for j in peaks:
            if (
                not valid[j]
                or fold[j] > cfg.elbow_max_fold_deg
                or height[j] < cfg.elbow_min_height_tl
            ):
                continue
            if lead[j] < cfg.elbow_min_lead_tl:
                continue
            if not np.isfinite(dist[j]) or dist[j] > cfg.max_distance_tl:
                continue
            w = max(1, int(0.1 * fps))
            out.append(
                Event(
                    fighter=fighter,
                    type=strike_name("elbow", "elbow", hand),
                    start_t=float(t[max(0, j - w)]),
                    peak_t=float(t[j]),
                    end_t=float(t[min(len(t) - 1, j + w)]),
                    confidence=0.65,
                    evidence={
                        "hand": hand,
                        "elbow_speed_tl_s": float(sp[j]),
                        "elbow_deg": float(fold[j]),
                        "in_range": True,
                    },
                )
            )
    return sorted(out, key=lambda e: e.peak_t)


def detect_clinch(df: pd.DataFrame, fighter: str, fps: float, cfg: CloseRangeConfig) -> list[Event]:
    df = df.reset_index(drop=True)
    t = df.t.to_numpy()
    mask = (df.distance_tl.to_numpy() <= cfg.clinch_max_distance_tl) & (
        df.hands_on_opponent.to_numpy() >= 1
    )
    mask = close_gaps(mask, int(0.3 * fps))
    out = []
    for s, e in runs(mask, max(1, int(cfg.clinch_min_s * fps))):
        out.append(
            Event(
                fighter=fighter,
                type="clinch",
                start_t=float(t[s]),
                peak_t=float(t[(s + e) // 2]),
                end_t=float(t[e]),
                confidence=0.7,
                evidence={
                    "two_hands_share": float(
                        (df.hands_on_opponent.to_numpy()[s : e + 1] >= 2).mean()
                    )
                },
            )
        )
    return out
