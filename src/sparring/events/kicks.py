"""Kick detection (kickboxing, muay thai).

A kick is a fast rise of one ankle well above its standing height. Per leg (lead/rear, from
stance) we find peaks in ankle height, grow each into a (chamber -> peak -> retraction) segment,
gate on ankle speed, and classify:

- teep (push kick): the foot travels mostly *forward* and the knee extends from a chambered
  position; peak height around the hip.
- head kick / body kick / low kick: by the ankle's peak height relative to the hip
  (shoulder level and above / around the hip and ribs / thigh and below).

Heights are in the body frame (hip = 0, up = +, torso lengths), so they do not depend on camera
distance. Like punches, anything moving along the camera's depth axis is foreshortened; a round
kick seen from directly in front shows mostly as height, which is what we classify on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sparring.config import KickConfig
from sparring.events.base import Event
from sparring.events.punches import _interp
from sparring.kinematics.features import LEGS
from sparring.moves import strike_name


def classify_kick(
    peak_y: float, forward: float, rise: float, knee_open: float, cfg: KickConfig
) -> str:
    if (
        forward >= cfg.teep_min_forward_tl
        and forward >= rise * cfg.teep_forward_ratio
        and knee_open >= cfg.teep_min_knee_opening_deg
    ):
        return "teep"
    if peak_y >= cfg.head_kick_min_y_tl:
        return "head_kick"
    if peak_y >= cfg.body_kick_min_y_tl:
        return "body_kick"
    return "low_kick"


def detect_leg(
    df: pd.DataFrame, fighter: str, leg: str, fps: float, cfg: KickConfig
) -> list[Event]:
    t = df.t.to_numpy()
    ay = _interp(df[f"{leg}_ankle_y"].to_numpy())
    ax = _interp(df[f"{leg}_ankle_x"].to_numpy())
    speed = _interp(np.hypot(df[f"{leg}_ankle_vx"].to_numpy(), df[f"{leg}_ankle_vy"].to_numpy()))
    knee = _interp(df[f"{leg}_knee_deg"].to_numpy())
    valid = np.isfinite(df[f"{leg}_ankle_y"].to_numpy()) & df.visible.to_numpy()
    in_range = df.in_range.to_numpy()
    n = len(df)
    windup = max(1, int(cfg.max_windup_s * fps))
    dur = max(2, int(cfg.max_duration_s * fps))
    peaks, _ = find_peaks(
        ay, prominence=cfg.min_rise_tl, distance=max(1, int(cfg.refractory_s * fps))
    )
    out: list[Event] = []
    last_end = -1
    for j in peaks:
        if not valid[j]:
            continue
        top = ay[j] - 0.05
        while j > 0 and ay[j - 1] >= top:
            j -= 1
        a = max(0, j - windup)
        s = a + int(np.argmin(ay[a : j + 1]))
        if s >= j or s <= last_end:
            continue
        rise = ay[j] - ay[s]
        if rise < cfg.min_rise_tl or speed[s : j + 1].max() < cfg.min_speed_tl:
            continue
        if knee[j] < cfg.min_knee_deg_at_impact:
            continue
        forward = ax[j] - ax[s]
        if forward < cfg.min_forward_tl:
            continue
        knee_open = knee[j] - knee[s : j + 1].min()
        family = classify_kick(ay[j], forward, rise, knee_open, cfg)
        e = j
        target = ay[j] - 0.6 * rise
        while e + 1 < min(n, s + dur) and ay[e] > target:
            e += 1
        conf = (
            0.5
            + 0.25 * min(speed[s : j + 1].max() / (2 * cfg.min_speed_tl), 1.0)
            + 0.25 * min(rise / 1.5, 1.0)
        )
        out.append(
            Event(
                fighter=fighter,
                type=strike_name("leg", family, leg),
                start_t=float(t[s]),
                peak_t=float(t[j]),
                end_t=float(t[e]),
                confidence=round(float(conf), 3),
                evidence={
                    "leg": leg,
                    "peak_height_tl": float(ay[j]),
                    "rise_tl": float(rise),
                    "forward_tl": float(forward),
                    "knee_opening_deg": float(knee_open),
                    "peak_speed_tl_s": float(speed[s : j + 1].max()),
                    "in_range": bool(in_range[j]),
                },
            )
        )
        last_end = e
    return out


def detect_kicks(df: pd.DataFrame, fighter: str, fps: float, cfg: KickConfig) -> list[Event]:
    df = df.reset_index(drop=True)
    events = [e for leg in LEGS for e in detect_leg(df, fighter, leg, fps, cfg)]
    return sorted(events, key=lambda e: e.peak_t)
