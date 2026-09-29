"""Entry point for event detection: run the configured discipline's detectors."""

from __future__ import annotations

import pandas as pd

from sparring.config import Config
from sparring.detection.detectors import DetectionContext, run_detectors
from sparring.disciplines import get_discipline
from sparring.events.base import Event


def detect_all(features: pd.DataFrame, fps: float, cfg: Config) -> list[Event]:
    discipline = get_discipline(cfg.discipline)
    ctx = DetectionContext.from_table(features, fps, cfg)
    events = run_detectors(discipline.detectors(cfg), ctx)
    return [e for e in events if discipline.allows(e.type)]
