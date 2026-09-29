"""Score detected events against hand labels.

Label file format (labels/<video>.json):

    {
      "video": "test/video/boxing.mp4",
      "segment": [0.0, 52.6],             # the labelled time range
      "ignore": [{"fighter": "B", "from": 19.0, "to": 22.0, "why": "camera behind B"}],
      "events": [{"fighter": "A", "type": "jab", "t": 1.07, "confidence": "high"}, ...]
    }

- `t` is the moment of impact / full extension for strikes, the moment of the evasion for
  defences, and any time inside the interval for sustained faults such as guard_drop.
- Generic types (`punch`, `kick`, `strike`) mark a strike whose type the labeller could not tell.
  They count toward detection scores but not toward type accuracy.
- `confidence: low` labels are *don't care*: a prediction matching one is neither a true nor a
  false positive, and the label is not a miss if nothing matches.
- `ignore` intervals are stretches where a fighter's actions are not visible (e.g. the camera is
  behind them). Labels and predictions for that fighter inside them are dropped.

Scores:
- `strike (any)`: was a strike detected at the right time, whatever its type? (detection)
- `strike type`: of the detected strikes whose label has a specific type, how many got the
  right type? (classification)
- one row per defence / fault type present in the labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sparring.events.base import Event, load_events
from sparring.moves import GENERIC_STRIKES, MOVES, kind_of, strikes

ANY_STRIKE = "strike (any)"
STRIKE_TYPE = "strike type"
# Faults that span an interval: a label anywhere inside the predicted interval is a match.
SUSTAINED = {"guard_drop", "chin_up", "square_stance", "clinch"}
ALL_STRIKES = set(strikes()) | set(GENERIC_STRIKES)


@dataclass
class Label:
    fighter: str
    type: str
    t: float
    confidence: str = "high"

    @property
    def is_strike(self) -> bool:
        return self.type in ALL_STRIKES

    @property
    def dont_care(self) -> bool:
        return self.confidence == "low"


@dataclass
class Labels:
    events: list[Label]
    segment: tuple[float, float] = (0.0, float("inf"))
    ignore: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Labels:
        keys = ("fighter", "type", "t", "confidence")
        return cls(
            events=[Label(**{k: e[k] for k in keys if k in e}) for e in d["events"]],
            segment=tuple(d.get("segment", (0.0, float("inf")))),
            ignore=d.get("ignore", []),
        )

    @classmethod
    def load(cls, path: Path) -> Labels:
        return cls.from_dict(json.loads(path.read_text()))

    def ignored(self, fighter: str, t: float) -> bool:
        return any(i["fighter"] == fighter and i["from"] <= t <= i["to"] for i in self.ignore)

    def restrict(self, a: float, b: float) -> Labels:
        """Labels within [a, b] (used for cross-validation folds)."""
        return Labels(
            events=[e for e in self.events if a <= e.t <= b],
            segment=(max(a, self.segment[0]), min(b, self.segment[1])),
            ignore=self.ignore,
        )


def _gap(interval: tuple[float, float], t: float) -> float:
    a, b = interval
    return max(a - t, t - b, 0.0)


def match(pred: list[tuple[float, float]], truth: list[float], tol: float) -> list[tuple[int, int]]:
    """One-to-one matches (pred index, truth index), greedy with the closest pairs first. A
    prediction is an interval (start, end); a label matches when it lies within `tol` of it."""
    pairs = sorted(
        (_gap(p, t), i, j)
        for i, p in enumerate(pred)
        for j, t in enumerate(truth)
        if _gap(p, t) <= tol
    )
    used_p, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            out.append((i, j))
    return out


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


def _interval(e: Event) -> tuple[float, float]:
    return (e.start_t, e.end_t) if e.type in SUSTAINED else (e.peak_t, e.peak_t)


def _score_group(
    events: list[Event], labels: Labels, pred_types: set[str], label_types: set[str], tol: float
) -> tuple[dict, list[tuple[Event, Label]]]:
    """P/R/F1 for one group, honouring don't-care labels and ignore intervals. Also returns the
    matched (event, label) pairs for type-accuracy scoring."""
    tp = n_pred = n_true = 0
    pairs: list[tuple[Event, Label]] = []
    for f in ("A", "B"):
        pred = [
            e
            for e in events
            if e.fighter == f and e.type in pred_types and not labels.ignored(f, e.peak_t)
        ]
        truth = [
            lb
            for lb in labels.events
            if lb.fighter == f and lb.type in label_types and not labels.ignored(f, lb.t)
        ]
        matches = match([_interval(e) for e in pred], [lb.t for lb in truth], tol)
        on_dont_care = {i for i, j in matches if truth[j].dont_care}
        for i, j in matches:
            if not truth[j].dont_care:
                tp += 1
                pairs.append((pred[i], truth[j]))
        # A prediction matched to a don't-care label is neither right nor wrong.
        n_pred += len(pred) - len(on_dont_care)
        n_true += sum(1 for lb in truth if not lb.dont_care)
    return _prf(tp, n_pred, n_true), pairs


def evaluate(events: list[Event], labels: Labels, tol: float = 0.15) -> dict:
    a, b = labels.segment
    events = [e for e in events if a <= e.peak_t <= b]
    result: dict[str, dict] = {}

    if any(lb.is_strike for lb in labels.events):
        result[ANY_STRIKE], pairs = _score_group(events, labels, set(strikes()), ALL_STRIKES, tol)
        typed = [(e, lb) for e, lb in pairs if lb.type in MOVES]
        correct = sum(1 for e, lb in typed if e.type == lb.type)
        result[STRIKE_TYPE] = {
            "tp": correct,
            "true": len(typed),
            "accuracy": round(correct / len(typed), 3) if typed else None,
            "confusions": sorted({f"{lb.type}->{e.type}" for e, lb in typed if e.type != lb.type}),
        }
    other = {lb.type for lb in labels.events if lb.type in MOVES and kind_of(lb.type) != "strike"}
    for t in sorted(other):
        result[t], _ = _score_group(events, labels, {t}, {t}, tol)
    return {"tolerance_s": tol, "segment": [a, b], "scores": result}


def evaluate_run(run_dir: Path, labels_path: Path, tol: float) -> dict:
    return evaluate(load_events(run_dir / "events.json"), Labels.load(labels_path), tol)


def format_table(result: dict) -> str:
    def fmt(v):
        return "   -" if v is None else f"{v:.2f}"

    rows = [f"{'type':<16}{'pred':>6}{'true':>6}{'tp':>5}{'prec':>7}{'rec':>7}{'f1':>7}"]
    for t, s in result["scores"].items():
        if t == STRIKE_TYPE:
            rows.append(f"{t:<16}{'':>6}{s['true']:>6}{s['tp']:>5}   accuracy {fmt(s['accuracy'])}")
            continue
        rows.append(
            f"{t:<16}{s['pred']:>6}{s['true']:>6}{s['tp']:>5}{fmt(s['precision']):>7}"
            f"{fmt(s['recall']):>7}{fmt(s['f1']):>7}"
        )
    return "\n".join(rows)
