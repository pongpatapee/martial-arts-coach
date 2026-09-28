"""Punch detection and classification from one fighter's feature table.

Two candidate routes, because 2D projection hides different punches in different ways:

1. Elbow opening (straights and uppercuts). The elbow angle snaps open from the guard (~60-90 deg)
   toward full extension. This is robust when the punch travels partly toward the camera, where
   wrist speed and reach are heavily foreshortened.
2. Fast bent-arm swing (hooks). A hook keeps the elbow bent, so route 1 misses it; instead we
   look for a fast wrist with the elbow bent and raised near shoulder height.

Each candidate is grown into a (chamber -> peak -> retraction) segment, gated on speed and
direction (toward the opponent or up), then classified:
- straight (jab = lead hand, cross = rear hand): elbow near full extension, or clear forward reach
- uppercut: mostly upward travel
- hook: bent-elbow swing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sparring.config import PunchConfig
from sparring.events.base import Event

_NAMES = {
    ("lead", "straight"): "jab",
    ("rear", "straight"): "cross",
    ("lead", "hook"): "lead_hook",
    ("rear", "hook"): "rear_hook",
    ("lead", "uppercut"): "lead_uppercut",
    ("rear", "uppercut"): "rear_uppercut",
}


def _interp(x: np.ndarray) -> np.ndarray:
    """Fill NaNs by linear interpolation (edges held) so peak finding sees a continuous signal."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def classify_opening(elbow_deg: float, forward: float, rise: float, cfg: PunchConfig) -> str:
    """Style of a punch found by the elbow-opening route."""
    if rise >= cfg.uppercut_min_rise_tl and rise >= cfg.uppercut_min_rise_ratio * max(forward, 0.0):
        return "uppercut"
    if elbow_deg >= cfg.straight_min_elbow_deg or forward >= cfg.straight_min_forward_tl:
        return "straight"
    return "hook"


class _HandSignals:
    def __init__(self, df: pd.DataFrame, hand: str):
        self.t = df.t.to_numpy()
        self.wx = _interp(df[f"{hand}_wrist_x"].to_numpy())
        self.wy = _interp(df[f"{hand}_wrist_y"].to_numpy())
        self.elbow = _interp(df[f"{hand}_elbow_deg"].to_numpy())
        self.elbow_h = _interp(df[f"{hand}_elbow_above_shoulder"].to_numpy())
        self.speed = _interp(
            np.hypot(df[f"{hand}_wrist_vx"].to_numpy(), df[f"{hand}_wrist_vy"].to_numpy())
        )
        self.valid = np.isfinite(df[f"{hand}_elbow_deg"].to_numpy()) & df.visible.to_numpy()
        self.axis_ok = df.axis_ok.to_numpy()
        self.in_range = df.in_range.to_numpy()
        self.n = len(df)


def _end_of(sig: _HandSignals, s: int, j: int, dur: int) -> int:
    """Retraction: first frame after the peak where the elbow has closed halfway back."""
    target = sig.elbow[j] - 0.5 * (sig.elbow[j] - sig.elbow[s])
    e = j
    while e + 1 < min(sig.n, s + dur) and sig.elbow[e] > target:
        e += 1
    return e


def _event(
    sig: _HandSignals,
    fighter: str,
    hand: str,
    style: str,
    route: str,
    s: int,
    j: int,
    e: int,
    cfg: PunchConfig,
) -> Event:
    speed = float(sig.speed[s : j + 1].max())
    opening = float(sig.elbow[j] - sig.elbow[s])
    conf = (
        0.4 + 0.3 * min(speed / (3 * cfg.min_speed_tl), 1.0) + 0.3 * min(max(opening, 0) / 90, 1.0)
    )
    if route == "swing":
        conf = 0.4 + 0.4 * min(speed / (2 * cfg.hook_min_speed_tl), 1.0)
    if not sig.axis_ok[j]:
        conf *= 0.6  # camera looking along the line of attack: direction is ambiguous
    return Event(
        fighter=fighter,
        type=_NAMES[(hand, style)],
        start_t=float(sig.t[s]),
        peak_t=float(sig.t[j]),
        end_t=float(sig.t[e]),
        confidence=round(float(conf), 3),
        evidence={
            "hand": hand,
            "route": route,
            "peak_speed_tl_s": speed,
            "elbow_deg": float(sig.elbow[j]),
            "elbow_opening_deg": opening,
            "forward_tl": float(sig.wx[j] - sig.wx[s]),
            "rise_tl": float(sig.wy[j] - sig.wy[s]),
            "in_range": bool(sig.in_range[j]),
            "axis_ok": bool(sig.axis_ok[j]),
        },
    )


def detect_hand(
    df: pd.DataFrame, fighter: str, hand: str, fps: float, cfg: PunchConfig
) -> list[Event]:
    sig = _HandSignals(df, hand)
    windup = max(1, int(cfg.max_windup_s * fps))
    dur = max(2, int(cfg.max_duration_s * fps))
    refractory = max(1, int(cfg.refractory_s * fps))
    found: list[tuple[int, int, int, str, str]] = []  # (s, j, e, style, route)

    # Route 1: elbow opening.
    peaks, _ = find_peaks(sig.elbow, prominence=cfg.min_elbow_opening_deg, distance=refractory)
    for j in peaks:
        if not sig.valid[j]:
            continue
        # find_peaks puts the peak mid-plateau when the arm is held out; the punch lands when
        # the arm first reaches (near) full extension.
        top = sig.elbow[j] - 3.0
        while j > 0 and sig.elbow[j - 1] >= top:
            j -= 1
        a = max(0, j - windup)
        s = a + int(np.argmin(sig.elbow[a : j + 1]))
        if s >= j:
            continue
        opening = sig.elbow[j] - sig.elbow[s]
        forward, rise = sig.wx[j] - sig.wx[s], sig.wy[j] - sig.wy[s]
        if opening < cfg.min_elbow_opening_deg or sig.elbow[j] < cfg.min_peak_elbow_deg:
            continue
        if sig.speed[s : j + 1].max() < cfg.min_speed_tl:
            continue
        if max(forward, rise) < cfg.min_forward_tl:
            continue
        found.append(
            (
                s,
                j,
                _end_of(sig, s, j, dur),
                classify_opening(sig.elbow[j], forward, rise, cfg),
                "opening",
            )
        )

    # Route 2: fast bent-arm swing, only where route 1 found nothing.
    covered = np.zeros(sig.n, bool)
    for s, _, e, _, _ in found:
        covered[max(0, s - refractory) : e + refractory] = True
    peaks, _ = find_peaks(sig.speed, height=cfg.hook_min_speed_tl, distance=refractory)
    half = max(1, windup // 2)
    for p in peaks:
        if covered[p] or not sig.valid[p]:
            continue
        if not (cfg.hook_min_elbow_deg <= sig.elbow[p] <= cfg.hook_max_elbow_deg):
            continue
        if sig.elbow_h[p] < cfg.hook_min_elbow_height_tl:
            continue
        s, j = max(0, p - half), min(sig.n - 1, p + half)
        forward, rise = sig.wx[j] - sig.wx[s], sig.wy[j] - sig.wy[s]
        if np.hypot(forward, rise) < cfg.hook_min_travel_tl or forward < -0.1:
            continue
        style = (
            "uppercut"
            if rise >= cfg.uppercut_min_rise_tl
            and rise >= cfg.uppercut_min_rise_ratio * max(forward, 0)
            else "hook"
        )
        found.append((s, p, min(sig.n - 1, p + half), style, "swing"))
        covered[max(0, s - refractory) : p + half + refractory] = True

    return [
        _event(sig, fighter, hand, style, route, s, j, e, cfg)
        for s, j, e, style, route in sorted(found)
    ]


def detect_punches(df: pd.DataFrame, fighter: str, fps: float, cfg: PunchConfig) -> list[Event]:
    events = detect_hand(df, fighter, "lead", fps, cfg) + detect_hand(df, fighter, "rear", fps, cfg)
    return sorted(events, key=lambda e: e.peak_t)


def detect_combinations(punches: list[Event], cfg: PunchConfig) -> list[Event]:
    """Group a fighter's punches separated by < combo_gap_s into combinations (2+ punches)."""
    combos: list[Event] = []
    group: list[Event] = []

    def flush() -> None:
        if len(group) >= 2:
            combos.append(
                Event(
                    fighter=group[0].fighter,
                    type="combination",
                    start_t=group[0].start_t,
                    peak_t=group[0].peak_t,
                    end_t=group[-1].end_t,
                    confidence=round(float(np.mean([g.confidence for g in group])), 3),
                    evidence={"punches": [g.type for g in group], "count": len(group)},
                )
            )

    for p in sorted(punches, key=lambda e: e.peak_t):
        if group and p.start_t - group[-1].peak_t > cfg.combo_gap_s:
            flush()
            group = []
        group.append(p)
    flush()
    return combos
