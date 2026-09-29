"""Move taxonomy shared by every discipline.

A discipline (boxing, kickboxing, muay thai, ...) is a *subset* of these moves plus the detectors
and coaching rules that go with them (see `sparring.disciplines`). Keeping one global vocabulary
means events, metrics, evaluation and the report never need discipline-specific code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["strike", "defense", "fault", "other"]
Limb = Literal["fist", "leg", "knee", "elbow", "none"]
Side = Literal["lead", "rear"]


@dataclass(frozen=True)
class MoveSpec:
    name: str
    kind: Kind
    label: str
    limb: Limb = "none"
    side: Side | None = None
    family: str = ""  # e.g. "straight", "hook", "roundhouse"; groups lead/rear variants


def _strikes(
    limb: Limb, family: str, label: str, names: tuple[str, str] | None = None
) -> list[MoveSpec]:
    """Lead and rear variants. With explicit names (jab/cross), the names are also the labels."""
    lead, rear = names or (f"lead_{family}", f"rear_{family}")
    lead_label, rear_label = names or (f"lead {label}", f"rear {label}")
    return [
        MoveSpec(lead, "strike", lead_label, limb, "lead", family),
        MoveSpec(rear, "strike", rear_label, limb, "rear", family),
    ]


_ALL: list[MoveSpec] = [
    # Punches (all disciplines). The lead/rear straights have their own names.
    *_strikes("fist", "straight", "straight", ("jab", "cross")),
    *_strikes("fist", "hook", "hook"),
    *_strikes("fist", "uppercut", "uppercut"),
    # Kicks (kickboxing, muay thai).
    *_strikes("leg", "teep", "teep"),
    *_strikes("leg", "low_kick", "low kick"),
    *_strikes("leg", "body_kick", "body kick"),
    *_strikes("leg", "head_kick", "head kick"),
    # Knees and elbows (muay thai).
    *_strikes("knee", "knee", "knee"),
    *_strikes("elbow", "elbow", "elbow"),
    # Defence: responses to an incoming strike.
    MoveSpec("slip", "defense", "slip"),
    MoveSpec("duck", "defense", "duck / roll"),
    MoveSpec("block", "defense", "block / cover"),
    MoveSpec("step_back", "defense", "step back"),
    MoveSpec("check", "defense", "check (shin block)"),
    MoveSpec("no_defense", "defense", "no defence"),
    # Faults.
    MoveSpec("guard_drop", "fault", "hand held low"),
    MoveSpec("slow_return", "fault", "slow return to guard"),
    MoveSpec("chin_up", "fault", "chin up"),
    MoveSpec("square_stance", "fault", "square stance"),
    MoveSpec("over_lean", "fault", "over-reaching"),
    MoveSpec("kick_no_guard", "fault", "hands down while kicking"),
    # Other.
    MoveSpec("combination", "other", "combination"),
    MoveSpec("clinch", "other", "clinch"),
]

MOVES: dict[str, MoveSpec] = {m.name: m for m in _ALL}

# Generic label types used when a labeller can see *that* a strike happened but not which one.
GENERIC_STRIKES = {"punch": "fist", "kick": "leg", "strike": None}


def spec(name: str) -> MoveSpec:
    try:
        return MOVES[name]
    except KeyError:
        raise ValueError(f"unknown move type {name!r}") from None


def kind_of(name: str) -> Kind:
    return spec(name).kind


def strikes(limb: Limb | None = None) -> tuple[str, ...]:
    """Strike names in taxonomy order (stable for reports)."""
    return tuple(m.name for m in _ALL if m.kind == "strike" and (limb is None or m.limb == limb))


def by_kind(kind: Kind) -> tuple[str, ...]:
    return tuple(m.name for m in _ALL if m.kind == kind)


PUNCHES = strikes("fist")
KICKS = strikes("leg")
DEFENSES = by_kind("defense")
FAULTS = by_kind("fault")


def strike_name(limb: Limb, family: str, side: Side) -> str:
    for m in _ALL:
        if m.kind == "strike" and m.limb == limb and m.family == family and m.side == side:
            return m.name
    raise ValueError(f"no strike for {limb}/{family}/{side}")
