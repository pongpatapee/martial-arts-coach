"""Defensive responses. Defence is relational: a slip only counts if a punch was coming, so every
defence event is anchored to an opponent punch thrown in range.

For each incoming punch we look at the defender's head/hands/distance from shortly before the
punch starts to shortly after it peaks, and classify the response in priority order:
duck > slip > block > step back > no defence.

Limitation: with one camera, lateral (side-to-side) head movement is partly along the camera's
depth axis, so slips are measured by 2D head displacement relative to the hips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparring.config import DefenseConfig
from sparring.events.base import Event


def _window(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    return df[(df.t >= a) & (df.t <= b)]


def classify_response(
    defender: pd.DataFrame, punch: Event, cfg: DefenseConfig
) -> tuple[str, dict] | None:
    w = _window(defender, punch.start_t - cfg.window_before_s, punch.peak_t + cfg.window_after_s)
    w = w[w.visible]
    if len(w) < 3:
        return None
    base = w.iloc[: max(1, len(w) // 5)]
    hx0, hy0 = base.head_x.mean(), base.head_y.mean()
    if not (np.isfinite(hx0) and np.isfinite(hy0)):
        return None
    drop = float(hy0 - w.head_y.min())
    lateral = float((w.head_x - hx0).abs().max())
    disp = float(np.nanmax(np.hypot(w.head_x - hx0, w.head_y - hy0)))
    at_peak = w.iloc[(w.t - punch.peak_t).abs().argmin()]
    guard = float(np.nanmin([at_peak.lead_wrist_head_dist, at_peak.rear_wrist_head_dist]))
    retreat = float(w.distance_tl.iloc[-1] - base.distance_tl.mean())
    evidence = {
        "against": punch.type,
        "against_t": punch.peak_t,
        "head_drop_tl": drop,
        "head_lateral_tl": lateral,
        "head_disp_tl": disp,
        "guard_wrist_head_tl": guard,
        "retreat_tl": retreat,
    }
    if drop >= cfg.duck_min_drop_tl and drop >= lateral:
        return "duck", evidence
    if disp >= cfg.slip_min_head_disp_tl:
        return "slip", evidence
    if np.isfinite(guard) and guard <= cfg.block_max_wrist_head_tl:
        return "block", evidence
    if np.isfinite(retreat) and retreat >= cfg.step_back_min_tl:
        return "step_back", evidence
    return "no_defense", evidence


def detect_defense(
    defender_df: pd.DataFrame, defender: str, opponent_punches: list[Event], cfg: DefenseConfig
) -> list[Event]:
    events = []
    for p in opponent_punches:
        if not p.evidence.get("in_range", True):
            continue
        res = classify_response(defender_df, p, cfg)
        if res is None:
            continue
        kind, evidence = res
        events.append(
            Event(
                fighter=defender,
                type=kind,
                start_t=p.start_t - cfg.window_before_s,
                peak_t=p.peak_t,
                end_t=p.peak_t + cfg.window_after_s,
                confidence=round(
                    p.confidence * (0.8 if kind in ("slip", "no_defense") else 0.9), 3
                ),
                evidence=evidence,
            )
        )
    return events
