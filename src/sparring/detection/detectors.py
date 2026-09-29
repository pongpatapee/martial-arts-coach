"""Detector plugins and the detection pipeline.

A discipline is a list of detectors (see `sparring.disciplines`). Detectors run in stages:

1. strike detectors (punches, kicks, knees, elbows) read the feature table;
2. relational detectors (defence, combinations, faults, clinch) read the feature table *and* the
   strikes found in stage 1 (a slip only means something if a strike was coming).

Every detector only emits `Event`s, so any of them can be swapped for a learned model without
touching metrics, coaching or rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from sparring.config import Config
from sparring.events.base import Event
from sparring.moves import MOVES
from sparring.skeleton import FIGHTERS, opponent

log = logging.getLogger(__name__)


@dataclass
class DetectionContext:
    features: dict[str, pd.DataFrame]  # per fighter, sorted by frame
    fps: float
    config: Config
    events: list[Event] = field(default_factory=list)  # accumulated output of earlier stages

    @classmethod
    def from_table(cls, feats: pd.DataFrame, fps: float, config: Config) -> DetectionContext:
        per = {
            f: feats[feats.fighter == f].sort_values("frame").reset_index(drop=True)
            for f in FIGHTERS
        }
        return cls(per, fps, config)

    def strikes(self, fighter: str | None = None, limb: str | None = None) -> list[Event]:
        return [
            e
            for e in self.events
            if e.kind == "strike"
            and (fighter is None or e.fighter == fighter)
            and (limb is None or MOVES[e.type].limb == limb)
        ]


class Detector(Protocol):
    name: str
    stage: int  # 1 = strikes, 2 = relational

    def detect(self, ctx: DetectionContext) -> list[Event]: ...


# -- stage 1: strikes ------------------------------------------------------------------------
@dataclass
class PunchDetector:
    """Punches via a pluggable method (rules / ensemble / learned / hybrid / temporal)."""

    method: str = "ensemble"
    model_path: Path | None = None
    name: str = "punches"
    stage: int = 1

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.detection.methods import RulesMethod, Session
        from sparring.detection.registry import load_method

        session = Session(ctx.features, ctx.fps)
        method = load_method(self.method, self.model_path, ctx.config)
        if method is None:
            method = RulesMethod(ctx.config.punch)
        return method.detect(session)


@dataclass
class KickDetector:
    name: str = "kicks"
    stage: int = 1

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.kicks import detect_kicks

        return [
            e
            for f, df in ctx.features.items()
            for e in detect_kicks(df, f, ctx.fps, ctx.config.kick)
        ]


@dataclass
class KneeDetector:
    name: str = "knees"
    stage: int = 1

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.close_range import detect_knees

        return [
            e
            for f, df in ctx.features.items()
            for e in detect_knees(df, f, ctx.fps, ctx.config.close_range)
        ]


@dataclass
class ElbowDetector:
    name: str = "elbows"
    stage: int = 1

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.close_range import detect_elbows

        return [
            e
            for f, df in ctx.features.items()
            for e in detect_elbows(df, f, ctx.fps, ctx.config.close_range)
        ]


def resolve_limb_conflicts(events: list[Event]) -> list[Event]:
    """When a fighter kicks or knees, the arms swing for balance and can look like punches; and a
    knee strike is a folded-leg rise that the kick detector may also see. Keep the leg strike and
    drop overlapping, less confident arm/leg duplicates from the same fighter."""
    legs = [e for e in events if e.kind == "strike" and MOVES[e.type].limb in ("leg", "knee")]
    out = []
    for e in events:
        if e.kind != "strike":
            out.append(e)
            continue
        limb = MOVES[e.type].limb
        clash = [
            k
            for k in legs
            if k is not e
            and k.fighter == e.fighter
            and k.start_t - 0.1 <= e.peak_t <= k.end_t + 0.1
        ]
        if limb == "fist" and any(k.confidence >= e.confidence - 0.1 for k in clash):
            continue
        if limb == "leg" and any(
            MOVES[k.type].limb == "knee" and k.confidence > e.confidence for k in clash
        ):
            continue
        out.append(e)
    return out


# -- stage 2: relational ---------------------------------------------------------------------
@dataclass
class CombinationDetector:
    name: str = "combinations"
    stage: int = 2

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.punches import detect_combinations

        return [e for f in FIGHTERS for e in detect_combinations(ctx.strikes(f), ctx.config.punch)]


@dataclass
class DefenseDetector:
    name: str = "defense"
    stage: int = 2

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.defense import detect_defense

        return [
            e
            for f, df in ctx.features.items()
            for e in detect_defense(df, f, ctx.strikes(opponent(f)), ctx.config.defense)
        ]


@dataclass
class FaultDetector:
    kicks: bool = False
    name: str = "faults"
    stage: int = 2

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.faults import detect_faults, detect_kick_faults

        out = []
        for f, df in ctx.features.items():
            out += detect_faults(df, f, ctx.strikes(f, "fist"), ctx.fps, ctx.config.faults)
            if self.kicks:
                out += detect_kick_faults(df, f, ctx.strikes(f, "leg"), ctx.config.faults)
        return out


@dataclass
class ClinchDetector:
    name: str = "clinch"
    stage: int = 2

    def detect(self, ctx: DetectionContext) -> list[Event]:
        from sparring.events.close_range import detect_clinch

        return [
            e
            for f, df in ctx.features.items()
            for e in detect_clinch(df, f, ctx.fps, ctx.config.close_range)
        ]


def run_detectors(detectors: list[Detector], ctx: DetectionContext) -> list[Event]:
    for stage in sorted({d.stage for d in detectors}):
        for d in (d for d in detectors if d.stage == stage):
            found = d.detect(ctx)
            log.debug("%s: %d events", d.name, len(found))
            ctx.events += found
        if stage == 1:
            ctx.events = resolve_limb_conflicts(ctx.events)
    return sorted(ctx.events, key=lambda e: (e.peak_t, e.fighter))
