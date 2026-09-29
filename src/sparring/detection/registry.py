"""Construct, save and load strike-detection methods by name."""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

import joblib

from sparring.config import Config
from sparring.detection.methods import EnsembleMethod, LearnedMethod, RulesMethod, StrikeMethod

log = logging.getLogger(__name__)

METHODS = ("rules", "rules_tuned", "learned", "hybrid", "ensemble", "temporal")
LEARNED = {"learned", "hybrid", "ensemble", "temporal"}


def new_method(name: str, config: Config) -> StrikeMethod:
    """An untrained method (for training and benchmarking)."""
    from sparring.detection.methods import TunedRulesMethod
    from sparring.detection.temporal import TemporalMethod

    punch = config.punch
    return {
        "rules": lambda: RulesMethod(punch),
        "rules_tuned": lambda: TunedRulesMethod(punch),
        "learned": lambda: LearnedMethod(punch_cfg=punch),
        "hybrid": lambda: LearnedMethod(use_rules=True, punch_cfg=punch),
        "ensemble": lambda: EnsembleMethod(
            learned=LearnedMethod(punch_cfg=punch), rules=RulesMethod(punch)
        ),
        "temporal": lambda: TemporalMethod(punch_cfg=punch),
    }[name]()


def bundled_model(method: str, discipline: str) -> Path | None:
    """A punch model shipped with the package (trained on the sample clip), if there is one.
    Punches look the same in every discipline, so others fall back to the boxing model."""
    for d in (discipline, "boxing"):
        ref = resources.files("sparring") / "assets" / f"{d}_punches_{method}.joblib"
        if ref.is_file():
            return Path(str(ref))
    return None


def save_method(method: StrikeMethod, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(method, "save"):
        method.save(path)
    else:
        joblib.dump({"method": method.name, "object": method}, path)


def load_method(name: str, path: Path | None, config: Config) -> StrikeMethod | None:
    """The configured method ready to run, or None (-> rules) if it needs a model we don't have."""
    if name not in METHODS:
        raise ValueError(f"unknown detection method {name!r}; choose from {METHODS}")
    if name not in LEARNED:
        return new_method(name, config)
    path = path or bundled_model(name, config.discipline)
    if path is None or not Path(path).exists():
        log.warning(
            "no trained model for method %r (discipline %s); falling back to rules. "
            "Train one with `sparring train`.",
            name,
            config.discipline,
        )
        return None
    if name == "ensemble":
        return EnsembleMethod.load(path, config.punch)
    if name in ("learned", "hybrid"):
        return LearnedMethod.load(path, config.punch)
    from sparring.detection.temporal import TemporalMethod

    return TemporalMethod.load(path, config.punch)
