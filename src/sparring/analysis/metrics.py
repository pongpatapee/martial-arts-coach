"""Aggregate events + features into per-fighter metrics (the input to the coaching layer).

Metrics are discipline-agnostic: strikes are counted per limb, so a boxing session simply has no
kicks. Rates are None when their denominator is zero, so "no data" is never mistaken for 0%.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from sparring.config import Config
from sparring.events.base import Event
from sparring.moves import DEFENSES, MOVES, strikes
from sparring.skeleton import opponent

COUNTER_WINDOW_S = 0.8  # a strike this soon after the opponent's is a counter...
COUNTER_QUIET_S = 0.5  # ...if the fighter wasn't already throwing just before it


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 3) if den else None


def head_move_after_striking(
    df: pd.DataFrame, attacks: list[Event], threshold_tl: float, window_s: float = 0.6
) -> float | None:
    """Share of strikes after which the head moves off its line within `window_s`."""
    if not attacks:
        return None
    moved = 0
    for p in attacks:
        w = df[(df.t >= p.peak_t) & (df.t <= p.peak_t + window_s)]
        if len(w) < 2:
            continue
        disp = np.hypot(w.head_x - w.head_x.iloc[0], w.head_y - w.head_y.iloc[0])
        if np.nanmax(disp.to_numpy()) >= threshold_tl:
            moved += 1
    return round(moved / len(attacks), 3)


def is_counter(incoming: Event, mine: list[Event]) -> bool:
    """A strike within COUNTER_WINDOW_S after the opponent's, when the fighter was not already
    mid-attack (no own strike in the COUNTER_QUIET_S before it): a response, not a trade."""
    busy = any(0 <= incoming.peak_t - m.peak_t < COUNTER_QUIET_S for m in mine)
    return not busy and any(0 < m.peak_t - incoming.peak_t <= COUNTER_WINDOW_S for m in mine)


def counter_rate(mine: list[Event], theirs: list[Event]) -> float | None:
    """Share of the opponent's strikes (in range) that the fighter countered."""
    incoming = [e for e in theirs if e.evidence.get("in_range", True)]
    if not incoming:
        return None
    return round(sum(is_counter(e, mine) for e in incoming) / len(incoming), 3)


def fighter_metrics(
    fighter: str, events: list[Event], feats: pd.DataFrame, fps: float, cfg: Config
) -> dict:
    df = feats[feats.fighter == fighter].sort_values("frame")
    mine = [e for e in events if e.fighter == fighter]
    theirs = [e for e in events if e.fighter == opponent(fighter)]
    all_strikes = [e for e in mine if e.kind == "strike"]
    punches = [e for e in all_strikes if MOVES[e.type].limb == "fist"]
    kicks = [e for e in all_strikes if MOVES[e.type].limb == "leg"]
    by_type = Counter(e.type for e in mine)
    visible_min = df.visible.sum() / fps / 60
    active = df.visible & df.in_range
    active_s = float(active.sum() / fps)

    returns = np.array([p.evidence.get("return_s", np.nan) for p in punches], float)
    returned = np.isfinite(returns) & (returns <= cfg.faults.return_to_guard_max_s)
    combos = [e for e in mine if e.type == "combination"]
    in_combo = sum(c.evidence["count"] for c in combos)

    incoming = [e for e in mine if e.type in DEFENSES]
    vs_kicks = [
        e
        for e in incoming
        if MOVES.get(e.evidence.get("against", ""), None)
        and MOVES[e.evidence["against"]].limb == "leg"
    ]
    vs_hands = [e for e in incoming if e not in vs_kicks]
    defended = [e for e in vs_hands if e.type != "no_defense"]
    drops = [e for e in mine if e.type == "guard_drop"]
    stance = df.lead_side[df.visible]
    rear_punches = [p for p in punches if MOVES[p.type].side == "rear"]

    def seconds(kind: str) -> float:
        return round(sum(e.duration for e in mine if e.type == kind), 2)

    def per_min(n: int) -> float | None:
        return round(n / visible_min, 1) if visible_min else None

    return {
        "visible_s": round(float(df.visible.sum() / fps), 1),
        "in_range_s": round(active_s, 1),
        "stance": {"orthodox_share": _rate(float((stance > 0).sum()), float(len(stance)))},
        "offense": {
            "strikes": len(all_strikes),
            "strikes_per_min": per_min(len(all_strikes)),
            "punches": len(punches),
            "punches_per_min": per_min(len(punches)),
            "kicks": len(kicks),
            "kick_share": _rate(len(kicks), len(all_strikes)),
            "by_type": {t: by_type[t] for t in strikes() if by_type.get(t)},
            "jab_share": _rate(by_type.get("jab", 0), len(punches)),
            "rear_hand_share": _rate(len(rear_punches), len(punches)),
            "variety": len({e.type for e in all_strikes}),
            "combinations": len(combos),
            "share_in_combinations": _rate(in_combo, len(all_strikes)),
            "mean_combination_length": round(in_combo / len(combos), 2) if combos else None,
            "counter_rate": counter_rate(all_strikes, [e for e in theirs if e.kind == "strike"]),
            "return_to_guard_rate": _rate(float(returned.sum()), len(punches)),
            "median_return_s": round(float(np.nanmedian(returns)), 3)
            if np.isfinite(returns).any()
            else None,
            "head_move_after_striking": head_move_after_striking(
                df, all_strikes, cfg.defense.slip_min_head_disp_tl
            ),
            "over_leans": by_type.get("over_lean", 0),
            "kicks_hands_down": by_type.get("kick_no_guard", 0),
        },
        "defense": {
            "incoming_in_range": len(vs_hands),
            "responses": {t: by_type.get(t, 0) for t in DEFENSES if t != "check"},
            "defense_rate": _rate(len(defended), len(vs_hands)),
            "head_movement_rate": _rate(
                by_type.get("slip", 0) + by_type.get("duck", 0), len(vs_hands)
            ),
            "block_rate": _rate(by_type.get("block", 0), len(vs_hands)),
            "kicks_incoming": len(vs_kicks),
            "check_rate": _rate(sum(1 for e in vs_kicks if e.type == "check"), len(vs_kicks)),
            "kick_defense_rate": _rate(
                sum(1 for e in vs_kicks if e.type != "no_defense"), len(vs_kicks)
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
        "clinch": {"seconds": seconds("clinch")},
    }
