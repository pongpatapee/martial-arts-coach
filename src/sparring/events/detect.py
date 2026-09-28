"""Runs every event detector over the feature table. This is the seam where a learned model can
replace or augment any rule detector: it only has to emit `Event`s."""

from __future__ import annotations

import pandas as pd

from sparring.config import Config
from sparring.events.base import Event
from sparring.events.defense import detect_defense
from sparring.events.faults import detect_faults
from sparring.events.punches import detect_combinations, detect_punches
from sparring.skeleton import FIGHTERS, opponent


def detect_all(features: pd.DataFrame, fps: float, cfg: Config) -> list[Event]:
    per_fighter = {
        f: features[features.fighter == f].sort_values("frame").reset_index(drop=True)
        for f in FIGHTERS
    }
    punches = {f: detect_punches(per_fighter[f], f, fps, cfg.punch) for f in FIGHTERS}
    events: list[Event] = []
    for f in FIGHTERS:
        events += punches[f]
        events += detect_combinations(punches[f], cfg.punch)
        events += detect_defense(per_fighter[f], f, punches[opponent(f)], cfg.defense)
        events += detect_faults(per_fighter[f], f, punches[f], fps, cfg.faults)
    return sorted(events, key=lambda e: (e.peak_t, e.fighter))
