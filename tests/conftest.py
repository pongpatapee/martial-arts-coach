from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import FPS  # noqa: E402

from sparring.config import Config  # noqa: E402
from sparring.kinematics.features import compute_features  # noqa: E402


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def features_of(cfg):
    def run(kp) -> pd.DataFrame:
        return compute_features(kp, FPS, cfg.kinematics)

    return run
