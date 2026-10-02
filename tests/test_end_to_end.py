"""End-to-end run on the sample clip. Needs the (git-ignored) video and downloads model weights,
so it is marked slow and skipped by default: run with `uv run pytest -m slow`.

The bundled strike model was trained on this clip, so the strike score here is in-sample: it
guards against regressions, not a measure of accuracy (see `sparring bench` for that)."""

import json
from pathlib import Path

import pytest

from sparring.config import Config
from sparring.pipeline import RunContext, run

CLIP = Path("datasets/clips/boxing-sample")
VIDEO = CLIP / "video.mp4"
LABELS = CLIP / "labels.json"


@pytest.mark.slow
@pytest.mark.skipif(not VIDEO.exists(), reason="sample video not present")
def test_pipeline_on_sample_clip(tmp_path):
    from sparring.evaluation import ANY_STRIKE, evaluate_run

    ctx = RunContext(video=VIDEO, run_dir=tmp_path, config=Config())
    run(ctx, to_stage="report")

    fighters = json.loads((tmp_path / "fighters.json").read_text())["fighters"]
    assert fighters["A"]["coverage"] > 0.9 and fighters["B"]["coverage"] > 0.9

    report = json.loads((tmp_path / "report.json").read_text())
    for f in ("A", "B"):
        assert report["fighters"][f]["metrics"]["offense"]["strikes"] > 10
        assert report["fighters"][f]["coaching"]["summary"]

    strikes = evaluate_run(tmp_path, LABELS, 0.15)["scores"][ANY_STRIKE]
    assert strikes["f1"] >= 0.7
