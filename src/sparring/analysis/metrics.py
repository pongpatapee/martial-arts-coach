"""Aggregate events + features into per-fighter metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from sparring.config import Config
from sparring.events.base import DEFENSES, PUNCHES, Event


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def head_move_after_punching(
    df: pd.DataFrame, punches: list[Event], threshold_tl: float, window_s: float = 0.6
) -> float | None:
    """Share of punches after which the head moves off its line within `window_s`."""
    if not punches:
        return None
    moved = 0
    for p in punches:
        w = df[(df.t >= p.peak_t) & (df.t <= p.peak_t + window_s)]
        if len(w) < 2:
            continue
        disp = np.hypot(w.head_x - w.head_x.iloc[0], w.head_y - w.head_y.iloc[0])
        if np.nanmax(disp.to_numpy()) >= threshold_tl:
            moved += 1
    return round(moved / len(punches), 3)


def fighter_metrics(
    fighter: str, events: list[Event], feats: pd.DataFrame, fps: float, cfg: Config
) -> dict:
    df = feats[feats.fighter == fighter].sort_values("frame")
    mine = [e for e in events if e.fighter == fighter]
    punches = [e for e in mine if e.kind == "punch"]
    by_type = Counter(e.type for e in mine)
    visible_min = df.visible.sum() / fps / 60
    active = df.visible & df.in_range
    active_s = float(active.sum() / fps)

    returns = np.array([p.evidence.get("return_s", np.nan) for p in punches], float)
    returned = np.isfinite(returns) & (returns <= cfg.faults.return_to_guard_max_s)
    combos = [e for e in mine if e.type == "combination"]
    in_combo = sum(c.evidence["count"] for c in combos)

    incoming = [e for e in mine if e.type in DEFENSES]
    defended = [e for e in incoming if e.type != "no_defense"]
    drops = [e for e in mine if e.type == "guard_drop"]
    stance = df.lead_side[df.visible]

    def seconds(kind: str) -> float:
        return round(sum(e.duration for e in mine if e.type == kind), 2)

    return {
        "visible_s": round(float(df.visible.sum() / fps), 1),
        "in_range_s": round(active_s, 1),
        "stance": {
            "orthodox_share": _rate(float((stance > 0).sum()), float(len(stance))),
        },
        "offense": {
            "punches": len(punches),
            "punches_per_min": round(len(punches) / visible_min, 1) if visible_min else None,
            "by_type": {t: by_type.get(t, 0) for t in PUNCHES},
            "jab_share": _rate(by_type.get("jab", 0), len(punches)),
            "combinations": len(combos),
            "share_in_combinations": _rate(in_combo, len(punches)),
            "mean_combination_length": round(in_combo / len(combos), 2) if combos else None,
            "return_to_guard_rate": _rate(float(returned.sum()), len(punches)),
            "median_return_s": round(float(np.nanmedian(returns)), 3)
            if np.isfinite(returns).any()
            else None,
            "head_move_after_punching": head_move_after_punching(
                df, punches, cfg.defense.slip_min_head_disp_tl
            ),
            "over_leans": by_type.get("over_lean", 0),
        },
        "defense": {
            "incoming_in_range": len(incoming),
            "responses": {t: by_type.get(t, 0) for t in DEFENSES},
            "defense_rate": _rate(len(defended), len(incoming)),
            "head_movement_rate": _rate(
                by_type.get("slip", 0) + by_type.get("duck", 0), len(incoming)
            ),
        },
        "guard": {
            "drops": len(drops),
            "drops_by_hand": dict(Counter(e.evidence["hand"] for e in drops)),
            "low_share_by_hand": {
                h: _rate(sum(e.duration for e in drops if e.evidence["hand"] == h), active_s)
                for h in ("lead", "rear")
            },
            "drop_seconds": seconds("guard_drop"),
            "drops_per_min_in_range": round(len(drops) / (active_s / 60), 2) if active_s else None,
            "chin_up_share": _rate(seconds("chin_up"), active_s),
            "square_stance_share": _rate(seconds("square_stance"), active_s),
        },
    }
