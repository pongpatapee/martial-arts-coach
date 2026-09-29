"""Event model shared by every detector, rule-based or learned."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sparring.moves import DEFENSES, FAULTS, KICKS, MOVES, PUNCHES, kind_of

__all__ = ["DEFENSES", "FAULTS", "KICKS", "PUNCHES", "Event", "load_events", "save_events"]


@dataclass
class Event:
    fighter: str
    type: str
    start_t: float
    peak_t: float
    end_t: float
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in MOVES:
            raise ValueError(f"unknown event type {self.type!r}")

    @property
    def kind(self) -> str:
        return kind_of(self.type)

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v) for v in obj]
    return obj


def save_events(events: list[Event], path: Path) -> None:
    events = sorted(events, key=lambda e: (e.peak_t, e.fighter))
    path.write_text(json.dumps([_round(asdict(e)) for e in events], indent=1))


def load_events(path: Path) -> list[Event]:
    return [Event(**d) for d in json.loads(path.read_text())]
