"""Technical faults: sustained bad positions (guard down, chin up, square stance) and
punch-linked faults (slow return to guard, over-leaning)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sparring.config import FaultConfig
from sparring.events.base import Event


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs with length >= min_len."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True) if e - s + 1 >= min_len]


def close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False runs of length <= max_gap that sit between True values."""
    mask = np.asarray(mask, bool).copy()
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= max_gap + 1:
            mask[a:b] = True
    return mask


def _punch_mask(t: np.ndarray, punches: list[Event], hand: str | None = None) -> np.ndarray:
    mask = np.zeros(len(t), bool)
    for p in punches:
        if hand is None or p.evidence.get("hand") == hand:
            mask |= (t >= p.start_t) & (t <= p.end_t + 0.3)
    return mask


def _sustained(
    df: pd.DataFrame,
    fighter: str,
    kind: str,
    mask: np.ndarray,
    min_s: float,
    fps: float,
    value: np.ndarray,
    extra: dict | None = None,
) -> list[Event]:
    t = df.t.to_numpy()
    out = []
    for s, e in runs(mask, max(1, int(min_s * fps))):
        out.append(
            Event(
                fighter=fighter,
                type=kind,
                start_t=float(t[s]),
                peak_t=float(t[(s + e) // 2]),
                end_t=float(t[e]),
                confidence=0.8,
                evidence={"mean_value": float(np.nanmean(value[s : e + 1])), **(extra or {})},
            )
        )
    return out


def guard_home(df: pd.DataFrame, hand: str, cfg: FaultConfig) -> np.ndarray:
    return (df[f"{hand}_wrist_below_shoulder"].to_numpy() <= cfg.guard_drop_below_shoulder_tl) & (
        df[f"{hand}_elbow_deg"].to_numpy() <= cfg.guard_home_max_elbow_deg
    )


def detect_faults(
    df: pd.DataFrame, fighter: str, punches: list[Event], fps: float, cfg: FaultConfig
) -> list[Event]:
    t = df.t.to_numpy()
    active = df.visible.to_numpy() & df.in_range.to_numpy()
    events: list[Event] = []

    for hand in ("lead", "rear"):
        below = df[f"{hand}_wrist_below_shoulder"].to_numpy()
        dropped = (
            active & (below > cfg.guard_drop_below_shoulder_tl) & ~_punch_mask(t, punches, hand)
        )
        dropped = close_gaps(dropped, int(cfg.merge_gap_s * fps))
        events += _sustained(
            df, fighter, "guard_drop", dropped, cfg.guard_drop_min_s, fps, below, {"hand": hand}
        )

    chin = df.chin_up.to_numpy()
    events += _sustained(
        df,
        fighter,
        "chin_up",
        active & (chin > cfg.chin_up_nose_above_ears_tl),
        cfg.chin_up_min_s,
        fps,
        chin,
    )

    width = df.stance_width_tl.to_numpy()
    square = active & df.axis_ok.to_numpy() & (np.abs(width) < cfg.square_stance_max_width_tl)
    events += _sustained(df, fighter, "square_stance", square, cfg.square_stance_min_s, fps, width)

    for p in punches:
        hand = p.evidence["hand"]
        home = guard_home(df, hand, cfg)
        after = np.flatnonzero((t > p.peak_t) & home)
        return_s = float(t[after[0]] - p.peak_t) if len(after) else float("nan")
        p.evidence["return_s"] = return_s
        if not np.isfinite(return_s) or return_s > cfg.return_to_guard_max_s:
            events.append(
                Event(
                    fighter=fighter,
                    type="slow_return",
                    start_t=p.peak_t,
                    peak_t=p.peak_t,
                    end_t=p.peak_t
                    + (return_s if np.isfinite(return_s) else cfg.return_to_guard_max_s),
                    confidence=round(p.confidence * 0.9, 3),
                    evidence={"punch": p.type, "hand": hand, "return_s": return_s},
                )
            )
        i = int(np.argmin(np.abs(t - p.peak_t)))
        lean = float(df.head_past_lead_foot.iloc[i])
        if df.axis_ok.iloc[i] and np.isfinite(lean) and lean > cfg.over_lean_head_past_foot_tl:
            events.append(
                Event(
                    fighter=fighter,
                    type="over_lean",
                    start_t=p.start_t,
                    peak_t=p.peak_t,
                    end_t=p.end_t,
                    confidence=round(p.confidence * 0.8, 3),
                    evidence={"punch": p.type, "head_past_lead_foot_tl": lean},
                )
            )
    return events
