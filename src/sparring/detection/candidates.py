"""High-recall strike candidates for the learned detectors.

A candidate is a moment where one hand *might* be striking: a peak in elbow opening or a peak in
wrist speed, with deliberately loose thresholds. The rule detector and the learned classifier
then decide which candidates are real strikes. Keeping candidate generation separate means its
recall is an explicit, measurable ceiling for every method built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sparring.kinematics.features import HANDS


@dataclass(frozen=True)
class CandidateConfig:
    min_elbow_prominence_deg: float = 20.0
    min_speed_tl: float = 2.0
    refractory_s: float = 0.12
    nms_s: float = 0.1  # merge a hand's elbow- and speed-peaks closer than this


@dataclass(frozen=True)
class Candidate:
    fighter: str
    hand: str
    frame: int
    t: float
    source: str  # "elbow" | "speed" | "both"


def _interp(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def hand_candidates(
    df: pd.DataFrame, fighter: str, hand: str, fps: float, cfg: CandidateConfig
) -> list[Candidate]:
    t = df.t.to_numpy()
    visible = df.visible.to_numpy()
    elbow = _interp(df[f"{hand}_elbow_deg"].to_numpy())
    speed = _interp(np.hypot(df[f"{hand}_wrist_vx"], df[f"{hand}_wrist_vy"]))
    dist = max(1, int(cfg.refractory_s * fps))
    pe, _ = find_peaks(elbow, prominence=cfg.min_elbow_prominence_deg, distance=dist)
    ps, _ = find_peaks(speed, height=cfg.min_speed_tl, distance=dist)
    tagged = sorted([(int(p), "elbow") for p in pe] + [(int(p), "speed") for p in ps])
    out: list[Candidate] = []
    nms = max(1, int(cfg.nms_s * fps))
    for p, src in tagged:
        if not visible[p]:
            continue
        if out and p - out[-1].frame <= nms:
            prev = out[-1]
            # Prefer the elbow peak's timing (closer to full extension = labelled impact time).
            frame = prev.frame if prev.source in ("elbow", "both") else p
            out[-1] = Candidate(fighter, hand, frame, float(t[frame]), "both")
            continue
        out.append(Candidate(fighter, hand, p, float(t[p]), src))
    return out


def fighter_candidates(
    df: pd.DataFrame, fighter: str, fps: float, cfg: CandidateConfig | None = None
) -> list[Candidate]:
    cfg = cfg or CandidateConfig()
    df = df.reset_index(drop=True)
    out = [c for hand in HANDS for c in hand_candidates(df, fighter, hand, fps, cfg)]
    return sorted(out, key=lambda c: (c.t, c.hand))
