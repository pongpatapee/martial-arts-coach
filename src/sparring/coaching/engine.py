"""Turn a fighter's metrics and events into prioritised, actionable coaching."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sparring.coaching.knowledge import RELIABILITY, RULES, Rule
from sparring.events.base import Event


@dataclass
class Feedback:
    id: str
    kind: str
    title: str
    observation: str
    why: str
    fix: str
    drills: list[dict]
    watch: list[float]  # timestamps (s) where it happens
    score: float
    basis: str
    reliability: str


def _fill(text: str, facts: dict) -> str:
    try:
        return text.format(**facts)
    except (KeyError, IndexError):
        return text


def evidence_times(
    events: list[Event], fighter: str, types: tuple[str, ...], match: dict, n: int, spacing: float
) -> list[float]:
    """The n most confident matching events' times, at least `spacing` seconds apart."""
    chosen = [
        e
        for e in events
        if e.fighter == fighter
        and e.type in types
        and all(e.evidence.get(k) == v for k, v in match.items())
    ]
    chosen.sort(key=lambda e: -e.confidence)
    picked: list[float] = []
    for e in chosen:
        if all(abs(e.peak_t - t) >= spacing for t in picked):
            picked.append(round(e.peak_t, 2))
        if len(picked) == n:
            break
    return sorted(picked)


def evaluate_rule(
    rule: Rule, metrics: dict, events: list[Event], fighter: str, n: int, spacing: float
) -> Feedback | None:
    res = rule.check(metrics)
    if res is None:
        return None
    score, facts = res
    match = {k: _fill(v, facts) for k, v in rule.evidence_filter.items()}
    return Feedback(
        id=rule.id,
        kind=rule.kind,
        title=_fill(rule.title, facts),
        observation=_fill(rule.observation, facts),
        why=_fill(rule.why, facts),
        fix=_fill(rule.fix, facts),
        drills=[asdict(d) for d in rule.drills],
        watch=evidence_times(events, fighter, rule.evidence, match, n, spacing),
        score=round(float(score), 3),
        basis=rule.basis,
        reliability=RELIABILITY[rule.basis],
    )


def coach(
    fighter: str,
    metrics: dict,
    events: list[Event],
    discipline: str,
    max_items: int = 3,
    n_evidence: int = 5,
    spacing: float = 1.0,
) -> dict:
    fired = [
        fb
        for rule in RULES
        if discipline in rule.disciplines
        and (fb := evaluate_rule(rule, metrics, events, fighter, n_evidence, spacing)) is not None
    ]
    strengths = sorted((f for f in fired if f.kind == "strength"), key=lambda f: -f.score)
    priorities = sorted((f for f in fired if f.kind == "improve"), key=lambda f: -f.score)
    strengths, priorities = strengths[:max_items], priorities[:max_items]

    # Session plan: the drills for the top priorities, then one to keep a strength sharp.
    plan, seen = [], set()
    for fb in priorities:
        for d in fb.drills[:2]:
            if d["name"] not in seen:
                seen.add(d["name"])
                plan.append({**d, "for": fb.title})
    if strengths and strengths[0].drills:
        d = strengths[0].drills[0]
        if d["name"] not in seen:
            plan.append({**d, "for": f"keep sharp: {strengths[0].title}"})

    if priorities:
        summary = f"Main focus: {priorities[0].title.lower()}."
        if strengths:
            summary += f" Build on: {strengths[0].title.lower()}."
    elif strengths:
        summary = f"No clear weaknesses in this clip. Build on: {strengths[0].title.lower()}."
    else:
        summary = "Not enough action in this clip for confident feedback."
    return {
        "summary": summary,
        "strengths": [asdict(f) for f in strengths],
        "priorities": [asdict(f) for f in priorities],
        "session_plan": plan[:5],
    }
