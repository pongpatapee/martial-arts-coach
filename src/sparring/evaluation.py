"""Score detected events against hand labels.

Label file format (labels/<video>.json):

    {
      "video": "test/video/boxing.mp4",
      "fighters": {"A": "gold headgear, white top", "B": "red headgear, black top"},
      "segment": [0.0, 20.0],          # only this time range was labelled
      "events": [{"fighter": "A", "type": "jab", "t": 1.23}, ...]
    }

`t` is the moment of full extension for punches, the moment of the evasion for defences, and any
time inside the interval for sustained faults such as guard_drop. The type "punch" marks a punch
whose type is unclear; it only counts toward the "punch (any)" score.
Matching is greedy by time within a tolerance, per fighter. We report exact-type scores and a
coarse "punch (any type)" score, which separates detection errors from classification errors.
"""

from __future__ import annotations

import json
from pathlib import Path

from sparring.events.base import PUNCHES, Event, load_events

ANY_PUNCH = "punch (any)"
# Faults that span an interval: a label anywhere inside the predicted interval is a match.
SUSTAINED = {"guard_drop", "chin_up", "square_stance"}


def _gap(interval: tuple[float, float], t: float) -> float:
    a, b = interval
    return max(a - t, t - b, 0.0)


def match(pred: list[tuple[float, float]], truth: list[float], tol: float) -> int:
    """Number of one-to-one matches (greedy, closest pairs first). A prediction is an interval
    (start, end); a label matches when it lies within `tol` of it. Point events use start == end."""
    pairs = sorted(
        (_gap(p, t), i, j)
        for i, p in enumerate(pred)
        for j, t in enumerate(truth)
        if _gap(p, t) <= tol
    )
    used_p, used_t, n = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            n += 1
    return n


def _prf(tp: int, n_pred: int, n_true: int) -> dict:
    p = tp / n_pred if n_pred else None
    r = tp / n_true if n_true else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if n_pred and n_true else None)
    return {
        "tp": tp,
        "pred": n_pred,
        "true": n_true,
        "precision": None if p is None else round(p, 3),
        "recall": None if r is None else round(r, 3),
        "f1": None if f1 is None else round(f1, 3),
    }


def evaluate(events: list[Event], labels: dict, tol: float) -> dict:
    a, b = labels.get("segment", [0.0, float("inf")])
    events = [e for e in events if a <= e.peak_t <= b]
    truth = labels["events"]
    types = sorted({lbl["type"] for lbl in truth} - {"punch"})
    result: dict[str, dict] = {}

    def score(type_set: set[str]) -> dict:
        tp = n_pred = n_true = 0
        for f in ("A", "B"):
            pred = [
                (e.start_t, e.end_t) if e.type in SUSTAINED else (e.peak_t, e.peak_t)
                for e in events
                if e.fighter == f and e.type in type_set
            ]
            true = [lbl["t"] for lbl in truth if lbl["fighter"] == f and lbl["type"] in type_set]
            tp += match(pred, true, tol)
            n_pred += len(pred)
            n_true += len(true)
        return _prf(tp, n_pred, n_true)

    for t in types:
        result[t] = score({t})
    if any(lbl["type"] in (*PUNCHES, "punch") for lbl in truth):
        result[ANY_PUNCH] = score({*PUNCHES, "punch"})
    return {"tolerance_s": tol, "segment": [a, b], "scores": result}


def evaluate_run(run_dir: Path, labels_path: Path, tol: float) -> dict:
    labels = json.loads(labels_path.read_text())
    return evaluate(load_events(run_dir / "events.json"), labels, tol)


def format_table(result: dict) -> str:
    rows = [f"{'type':<16}{'pred':>6}{'true':>6}{'tp':>5}{'prec':>7}{'rec':>7}{'f1':>7}"]
    for t, s in result["scores"].items():
        fmt = lambda v: "   -" if v is None else f"{v:.2f}"  # noqa: E731
        rows.append(
            f"{t:<16}{s['pred']:>6}{s['true']:>6}{s['tp']:>5}{fmt(s['precision']):>7}"
            f"{fmt(s['recall']):>7}{fmt(s['f1']):>7}"
        )
    return "\n".join(rows)
