"""Martial-art disciplines: which moves exist, which detectors find them, which coaching applies.

Adding a discipline (e.g. MMA striking, karate) means declaring its move set and detector list
here and its coaching rules in `sparring.coaching.knowledge`; the perception, kinematics, event
model, evaluation, metrics and report are shared.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sparring.config import Config
from sparring.detection.detectors import (
    ClinchDetector,
    CombinationDetector,
    DefenseDetector,
    Detector,
    ElbowDetector,
    FaultDetector,
    KickDetector,
    KneeDetector,
    PunchDetector,
)
from sparring.moves import DEFENSES, FAULTS, KICKS, PUNCHES, strikes


@dataclass(frozen=True)
class Discipline:
    name: str
    title: str
    moves: tuple[str, ...]
    build_detectors: Callable[[Config], list[Detector]]

    def detectors(self, config: Config) -> list[Detector]:
        return self.build_detectors(config)

    def allows(self, move: str) -> bool:
        return move in self.moves


def _punch(cfg: Config) -> PunchDetector:
    path = Path(cfg.detection.model_path) if cfg.detection.model_path else None
    return PunchDetector(method=cfg.detection.method, model_path=path)


def _boxing(cfg: Config) -> list[Detector]:
    return [_punch(cfg), CombinationDetector(), DefenseDetector(), FaultDetector(kicks=False)]


def _kickboxing(cfg: Config) -> list[Detector]:
    return [
        _punch(cfg),
        KickDetector(),
        CombinationDetector(),
        DefenseDetector(),
        FaultDetector(kicks=True),
    ]


def _muay_thai(cfg: Config) -> list[Detector]:
    return [
        _punch(cfg),
        KickDetector(),
        KneeDetector(),
        ElbowDetector(),
        CombinationDetector(),
        DefenseDetector(),
        FaultDetector(kicks=True),
        ClinchDetector(),
    ]


_COMMON = (*DEFENSES, *FAULTS, "combination")
DISCIPLINES: dict[str, Discipline] = {
    d.name: d
    for d in (
        Discipline(
            "boxing",
            "Boxing",
            (*PUNCHES, *(x for x in _COMMON if x not in ("check", "kick_no_guard"))),
            _boxing,
        ),
        Discipline("kickboxing", "Kickboxing", (*PUNCHES, *KICKS, *_COMMON), _kickboxing),
        Discipline("muay_thai", "Muay Thai", (*strikes(), *_COMMON, "clinch"), _muay_thai),
    )
}


def get_discipline(name: str) -> Discipline:
    try:
        return DISCIPLINES[name]
    except KeyError:
        raise ValueError(
            f"unknown discipline {name!r}; choose from {sorted(DISCIPLINES)}"
        ) from None
