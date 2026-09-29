"""Time-blocked cross-validation benchmark of strike-detection methods.

The clip is split into K contiguous time blocks. Each method is trained on K-1 blocks (rules
methods ignore training) and scored on the held-out block; true/false positives are summed
across folds (micro-averaged). Contiguous blocks, rather than random candidates, keep
near-duplicate frames of one exchange from leaking between train and test.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from sparring.detection.methods import Session, StrikeMethod
from sparring.evaluation import ANY_STRIKE, STRIKE_TYPE, evaluate


@dataclass
class BenchResult:
    method: str
    tp: int
    pred: int
    true: int
    type_correct: int
    type_total: int
    seconds: float
    fold_f1: list[float] = field(default_factory=list)

    @property
    def f1_spread(self) -> float:
        """Standard deviation of per-fold F1: differences between methods smaller than this are
        not meaningful."""
        return float(np.std(self.fold_f1)) if self.fold_f1 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / self.pred if self.pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.true if self.true else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.tp / (self.pred + self.true) if self.pred + self.true else 0.0

    @property
    def type_accuracy(self) -> float | None:
        return self.type_correct / self.type_total if self.type_total else None

    def row(self) -> dict:
        return {
            "method": self.method,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "type_accuracy": None if self.type_accuracy is None else round(self.type_accuracy, 3),
            "f1_fold_std": round(self.f1_spread, 3),
            "tp": self.tp,
            "pred": self.pred,
            "true": self.true,
            "seconds": round(self.seconds, 1),
        }


def folds(session: Session, k: int) -> list[tuple[float, float]]:
    a, b = session.labels.segment
    b = min(b, max(float(d.t.max()) for d in session.features.values()))
    edges = np.linspace(a, b, k + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(k)]


def cross_validate(method: StrikeMethod, session: Session, k: int = 5) -> BenchResult:
    t0 = time.perf_counter()
    tp = pred = true = tc = tt = 0
    fold_f1 = []
    blocks = folds(session, k)
    for i, (a, b) in enumerate(blocks):
        train = [session.window(*blk) for j, blk in enumerate(blocks) if j != i]
        test = session.window(a, b)
        m = copy.deepcopy(method).fit(train)
        scores = evaluate(m.detect(test), test.labels)["scores"]
        s, ty = scores[ANY_STRIKE], scores[STRIKE_TYPE]
        tp, pred, true = tp + s["tp"], pred + s["pred"], true + s["true"]
        tc, tt = tc + ty["tp"], tt + ty["true"]
        fold_f1.append(s["f1"] or 0.0)
    return BenchResult(method.name, tp, pred, true, tc, tt, time.perf_counter() - t0, fold_f1)


def pooled(results: list[BenchResult]) -> BenchResult:
    """Sum per-clip results of one method (micro-average across clips)."""
    r0 = results[0]
    return BenchResult(
        r0.method,
        sum(r.tp for r in results),
        sum(r.pred for r in results),
        sum(r.true for r in results),
        sum(r.type_correct for r in results),
        sum(r.type_total for r in results),
        sum(r.seconds for r in results),
        [f for r in results for f in r.fold_f1],
    )


def format_results(results: list[BenchResult]) -> str:
    head = f"{'method':<14}{'prec':>7}{'rec':>7}{'f1':>7}{'±fold':>7}{'type acc':>10}{'tp/pred/true':>16}{'secs':>7}"
    rows = [head]
    for r in sorted(results, key=lambda r: -r.f1):
        ta = "-" if r.type_accuracy is None else f"{r.type_accuracy:.2f}"
        rows.append(
            f"{r.method:<14}{r.precision:>7.2f}{r.recall:>7.2f}{r.f1:>7.2f}{r.f1_spread:>7.2f}{ta:>10}"
            f"{f'{r.tp}/{r.pred}/{r.true}':>16}{r.seconds:>7.1f}"
        )
    return "\n".join(rows)
