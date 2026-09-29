"""Interchangeable strike-detection methods.

Every method maps a fighter's feature table to strike `Event`s, so they can be benchmarked
against each other (`sparring bench`) and swapped in the pipeline (`detection.method`).

- `rules`: the hand-written geometric detector (`events/punches.py`).
- `learned`: high-recall candidates (`candidates.py`) + a gradient-boosted classifier over
  time-window features (`windows.py`). Type/hand come from the same geometry as the rules.
- `hybrid`: `learned`, with the rule detector's output as extra classifier inputs.
- `temporal`: a small dilated 1D-CNN over the whole feature sequence that outputs a per-frame
  "strike now" heatmap (see `temporal.py`).

Learned methods need training labels; `fit` takes feature tables plus `Labels`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from sparring.config import PunchConfig
from sparring.detection.candidates import Candidate, CandidateConfig, fighter_candidates
from sparring.detection.windows import WindowFeaturizer
from sparring.evaluation import Labels
from sparring.events.base import Event
from sparring.events.punches import detect_punches, event_at

log = logging.getLogger(__name__)

TRAIN_TOL_S = 0.15  # a candidate within this of a labelled strike is a positive
NEGATIVE_GAP_S = 0.3  # ... and one further than this from any strike label is a negative


@dataclass
class Session:
    """One labelled (or unlabelled) clip: per-fighter feature tables + optional labels."""

    features: dict[str, pd.DataFrame]
    fps: float
    labels: Labels | None = None
    name: str = ""

    @classmethod
    def from_table(
        cls, feats: pd.DataFrame, fps: float, labels: Labels | None = None, name: str = ""
    ) -> Session:
        per = {
            f: g.sort_values("frame").reset_index(drop=True) for f, g in feats.groupby("fighter")
        }
        return cls(per, fps, labels, name)

    def window(self, a: float, b: float) -> Session:
        """The sub-session in [a, b] seconds (for time-blocked cross-validation)."""
        per = {
            f: d[(d.t >= a) & (d.t <= b)].reset_index(drop=True) for f, d in self.features.items()
        }
        return Session(
            per, self.fps, self.labels.restrict(a, b) if self.labels else None, self.name
        )


class StrikeMethod(Protocol):
    name: str

    def fit(self, sessions: list[Session]) -> StrikeMethod: ...

    def detect(self, session: Session) -> list[Event]: ...


def nms(events: list[Event], window_s: float) -> list[Event]:
    """Per fighter, keep the most confident event among any closer than window_s."""
    kept: list[Event] = []
    for e in sorted(events, key=lambda e: -e.confidence):
        if all(k.fighter != e.fighter or abs(k.peak_t - e.peak_t) >= window_s for k in kept):
            kept.append(e)
    return sorted(kept, key=lambda e: (e.peak_t, e.fighter))


# --------------------------------------------------------------------------------------------
@dataclass
class RulesMethod:
    cfg: PunchConfig = field(default_factory=PunchConfig)
    name: str = "rules"

    def fit(self, sessions: list[Session]) -> RulesMethod:
        return self

    def detect(self, session: Session) -> list[Event]:
        out = []
        for f, df in session.features.items():
            out += detect_punches(df, f, session.fps, self.cfg)
        return sorted(out, key=lambda e: (e.peak_t, e.fighter))


@dataclass
class TunedRulesMethod:
    """Rules whose key thresholds are grid-searched on the training folds."""

    base: PunchConfig = field(default_factory=PunchConfig)
    name: str = "rules_tuned"
    cfg: PunchConfig | None = None

    GRID = {
        "min_elbow_opening_deg": (35.0, 45.0),
        "min_peak_elbow_deg": (110.0, 125.0),
        "min_speed_tl": (1.2, 2.0),
        "reach_min_gain_tl": (0.45, 0.6, 0.8),
        "reach_min_speed_tl": (3.5, 5.0),
    }

    def fit(self, sessions: list[Session]) -> TunedRulesMethod:
        from itertools import product

        from sparring.evaluation import ANY_STRIKE, evaluate

        best, best_f1 = self.base, -1.0
        keys = list(self.GRID)
        for combo in product(*self.GRID.values()):
            cfg = self.base.model_copy(update=dict(zip(keys, combo, strict=True)))
            tp = pred = true = 0
            for s in sessions:
                r = evaluate(RulesMethod(cfg).detect(s), s.labels)["scores"][ANY_STRIKE]
                tp, pred, true = tp + r["tp"], pred + r["pred"], true + r["true"]
            f1 = 2 * tp / (pred + true) if pred + true else 0.0
            if f1 > best_f1:
                best, best_f1 = cfg, f1
        self.cfg = best
        return self

    def detect(self, session: Session) -> list[Event]:
        return RulesMethod(self.cfg or self.base).detect(session)


# --------------------------------------------------------------------------------------------
def label_candidates(cands: list[Candidate], labels: Labels, fighter: str) -> np.ndarray:
    """1 = strike, 0 = not a strike, -1 = unknown (near a don't-care label or ignored)."""
    strike_labels = [lb for lb in labels.events if lb.fighter == fighter and lb.is_strike]
    y = np.zeros(len(cands), int)
    for i, c in enumerate(cands):
        if labels.ignored(fighter, c.t):
            y[i] = -1
            continue
        near = [lb for lb in strike_labels if abs(lb.t - c.t) <= NEGATIVE_GAP_S]
        close = [lb for lb in near if abs(lb.t - c.t) <= TRAIN_TOL_S]
        if any(not lb.dont_care for lb in close):
            y[i] = 1
        elif near:
            y[i] = -1  # near a strike but not clearly it (don't-care, or the other hand)
    # Each label may justify only one positive per hand pair; the classifier learns which hand.
    return y


class _Classifier:
    """Gradient-boosted trees on the columns that carry information in the training set."""

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> _Classifier:
        from sklearn.ensemble import HistGradientBoostingClassifier

        finite = np.isfinite(X)
        self.cols = np.array(
            [
                finite[:, j].sum() >= 2 and np.unique(X[finite[:, j], j]).size >= 2
                for j in range(X.shape[1])
            ]
        )
        self.clf = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=1.0,
            min_samples_leaf=8,
            random_state=0,
        ).fit(X[:, self.cols], y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X[:, self.cols])


@dataclass
class LearnedMethod:
    """Candidate generation + gradient-boosted window classifier."""

    use_rules: bool = False
    use_3d: bool = False
    use_opponent: bool = True
    threshold: float | None = None  # None = choose on training data (out-of-fold F1)
    punch_cfg: PunchConfig = field(default_factory=PunchConfig)
    cand_cfg: CandidateConfig = field(default_factory=CandidateConfig)
    name: str = "learned"
    model: object | None = None

    def __post_init__(self) -> None:
        if self.name == "learned":
            self.name = "hybrid" if self.use_rules else "learned"
            if self.use_3d:
                self.name += "_3d"
            if not self.use_opponent:
                self.name += "_solo"

    def _xy(self, session: Session, with_labels: bool):
        X, y, meta = [], [], []
        rules = RulesMethod(self.punch_cfg).detect(session) if self.use_rules else None
        for f, df in session.features.items():
            if df.empty:
                continue
            opp = next((d for g, d in session.features.items() if g != f), None)
            fz = WindowFeaturizer(df, opp if self.use_opponent else None, use_3d=self.use_3d)
            cands = fighter_candidates(df, f, session.fps, self.cand_cfg)
            if not cands:
                continue
            fr = [e for e in rules if e.fighter == f] if rules is not None else None
            X += [fz(c, fr) for c in cands]
            meta += [(f, c) for c in cands]
            if with_labels:
                y += list(label_candidates(cands, session.labels, f))
        X = np.asarray(X, np.float32) if X else np.zeros((0, 1), np.float32)
        return X, np.asarray(y, int), meta

    @staticmethod
    def _classifier():
        return _Classifier()

    @staticmethod
    def _weights(y: np.ndarray) -> np.ndarray:
        pos = max(int(y.sum()), 1)
        return np.where(y == 1, (len(y) - pos) / pos, 1.0)

    def _train(self, parts: list) -> _Classifier:
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        keep = y >= 0
        return self._classifier().fit(X[keep], y[keep], sample_weight=self._weights(y[keep]))

    def oof_predictions(self, sessions: list[Session], parts: list | None = None) -> list:
        """Out-of-fold candidate probabilities: for each training session, a model trained on the
        others. Used to choose thresholds without touching the held-out test data."""
        parts = parts or [self._xy(s, True) for s in sessions]
        out = []
        for i, (X, _, meta) in enumerate(parts):
            others = [p for j, p in enumerate(parts) if j != i and len(p[0]) and (p[1] == 1).any()]
            if not len(X) or not others:
                out.append((meta, np.zeros(len(meta))))
                continue
            out.append((meta, self._train(others).predict_proba(X)[:, 1]))
        return out

    def fit(self, sessions: list[Session]) -> LearnedMethod:
        parts = [self._xy(s, True) for s in sessions]
        usable = [p for p in parts if len(p[0])]
        if self.threshold is None:
            self.threshold = 0.5
            if len(sessions) >= 2:
                oof = self.oof_predictions(sessions, parts)
                self.threshold = best_threshold(
                    sessions,
                    lambda s, th, i: self._events(s, *oof[i], th),
                    np.arange(0.2, 0.95, 0.05),
                )
        self.model = self._train(usable)
        log.info("%s: trained, threshold %.2f", self.name, self.threshold)
        return self

    def _events(
        self, session: Session, meta: list, proba: np.ndarray, threshold: float
    ) -> list[Event]:
        out = []
        for (f, c), p in zip(meta, proba, strict=True):
            if p < threshold:
                continue
            e = event_at(
                session.features[f],
                f,
                c.hand,
                c.frame,
                session.fps,
                self.punch_cfg,
                route=self.name,
            )
            e.confidence = round(float(p), 3)
            e.evidence["candidate"] = c.source
            out.append(e)
        return nms(out, 0.15)

    def probabilities(self, session: Session) -> tuple[list, np.ndarray]:
        X, _, meta = self._xy(session, False)
        return meta, (self.model.predict_proba(X)[:, 1] if len(meta) else np.zeros(0))

    def detect(self, session: Session) -> list[Event]:
        meta, proba = self.probabilities(session)
        return self._events(session, meta, proba, self.threshold)

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "method": self.name,
                "use_rules": self.use_rules,
                "use_3d": self.use_3d,
                "use_opponent": self.use_opponent,
                "threshold": self.threshold,
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, punch_cfg: PunchConfig | None = None) -> LearnedMethod:
        d = joblib.load(path)
        m = cls(
            use_rules=d["use_rules"],
            use_3d=d["use_3d"],
            threshold=d["threshold"],
            use_opponent=d.get("use_opponent", True),
            punch_cfg=punch_cfg or PunchConfig(),
        )
        m.model = d["model"]
        return m


def best_threshold(sessions: list[Session], events_for, grid) -> float:
    """The grid value maximising micro-F1 of strike detection summed over sessions.
    `events_for(session, value, session_index)` returns that session's events."""
    from sparring.evaluation import ANY_STRIKE, evaluate

    best, best_f1 = grid[0], -1.0
    for v in grid:
        tp = pred = true = 0
        for i, s in enumerate(sessions):
            r = evaluate(events_for(s, v, i), s.labels)["scores"][ANY_STRIKE]
            tp, pred, true = tp + r["tp"], pred + r["pred"], true + r["true"]
        f1 = 2 * tp / (pred + true) if pred + true else 0.0
        if f1 > best_f1:
            best, best_f1 = v, f1
    return best


@dataclass
class EnsembleMethod:
    """Rules as the backbone, the learned classifier as a second opinion: a rule detection is
    vetoed when the classifier is confident it is not a strike, and a candidate the rules missed is
    added when the classifier is confident it is one. Both thresholds are chosen out-of-fold."""

    learned: LearnedMethod = field(default_factory=LearnedMethod)
    rules: RulesMethod = field(default_factory=RulesMethod)
    veto: float = 0.1
    add: float = 0.9
    name: str = "ensemble"

    def _combine(
        self, session: Session, meta: list, proba: np.ndarray, veto: float, add: float
    ) -> list[Event]:
        rule_events = self.rules.detect(session)
        by_f: dict[str, list[tuple[Candidate, float]]] = {}
        for (f, c), p in zip(meta, proba, strict=True):
            by_f.setdefault(f, []).append((c, float(p)))
        out = []
        for e in rule_events:
            near = [p for c, p in by_f.get(e.fighter, []) if abs(c.t - e.peak_t) <= 0.1]
            p = max(near, default=None)
            if p is not None and p < veto:
                continue
            e.evidence["model_p"] = p
            out.append(e)
        for (f, c), p in zip(meta, proba, strict=True):
            if p >= add and all(o.fighter != f or abs(o.peak_t - c.t) > 0.15 for o in out):
                ev = event_at(
                    session.features[f],
                    f,
                    c.hand,
                    c.frame,
                    session.fps,
                    self.rules.cfg,
                    route="model",
                )
                ev.confidence = round(float(p), 3)
                out.append(ev)
        return nms(out, 0.15)

    def fit(self, sessions: list[Session]) -> EnsembleMethod:
        self.learned.threshold = 0.5  # unused by the ensemble; skip its own threshold search
        if len(sessions) >= 2:
            oof = self.learned.oof_predictions(sessions)
            grid = [(v, a) for v in (0.0, 0.05, 0.1, 0.2, 0.3) for a in (0.7, 0.8, 0.9, 0.95, 1.01)]
            self.veto, self.add = best_threshold(
                sessions, lambda s, va, i: self._combine(s, *oof[i], *va), grid
            )
        self.learned.fit(sessions)
        log.info("ensemble: veto < %.2f, add >= %.2f", self.veto, self.add)
        return self

    def detect(self, session: Session) -> list[Event]:
        meta, proba = self.learned.probabilities(session)
        return self._combine(session, meta, proba, self.veto, self.add)

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "method": self.name,
                "veto": self.veto,
                "add": self.add,
                "learned": {
                    "use_rules": self.learned.use_rules,
                    "use_3d": self.learned.use_3d,
                    "use_opponent": self.learned.use_opponent,
                    "model": self.learned.model,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, punch_cfg: PunchConfig | None = None) -> EnsembleMethod:
        d = joblib.load(path)
        cfg = punch_cfg or PunchConfig()
        lm = d["learned"]
        learned = LearnedMethod(
            use_rules=lm["use_rules"],
            use_3d=lm["use_3d"],
            use_opponent=lm["use_opponent"],
            threshold=0.5,
            punch_cfg=cfg,
        )
        learned.model = lm["model"]
        return cls(learned=learned, rules=RulesMethod(cfg), veto=d["veto"], add=d["add"])
