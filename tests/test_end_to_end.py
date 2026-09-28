"""End-to-end run on the sample clip. Needs the (git-ignored) video and downloads model weights,
so it is marked slow and skipped by default: run with `uv run pytest -m slow`."""

import json
from pathlib import Path

import pytest

from sparring.config import Config
from sparring.pipeline import RunContext, run

VIDEO = Path("test/video/boxing.mp4")
LABELS = Path("labels/boxing.json")


@pytest.mark.slow
@pytest.mark.skipif(not VIDEO.exists(), reason="sample video not present")
def test_pipeline_on_sample_clip(tmp_path):
    from sparring.evaluation import evaluate_run

    ctx = RunContext(video=VIDEO, run_dir=tmp_path, config=Config())
    run(ctx, to_stage="report")

    fighters = json.loads((tmp_path / "fighters.json").read_text())["fighters"]
    assert fighters["A"]["coverage"] > 0.9 and fighters["B"]["coverage"] > 0.9

    report = json.loads((tmp_path / "report.json").read_text())
    for f in ("A", "B"):
        assert report["fighters"][f]["metrics"]["offense"]["punches"] > 5

    scores = evaluate_run(tmp_path, LABELS, 0.15)["scores"]
    assert scores["punch (any)"]["precision"] >= 0.6
