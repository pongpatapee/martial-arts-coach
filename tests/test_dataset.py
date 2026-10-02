import json

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from sparring.cli import _training_data, app
from sparring.dataset import Clip, ClipError, Dataset


@pytest.fixture
def video(tmp_path):
    """A real, decodable 1-second clip at 30 fps."""
    path = tmp_path / "src" / "Sparring Round 1.mp4"
    path.parent.mkdir()
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(30):
        out.write(np.full((48, 64, 3), i * 8, np.uint8))
    out.release()
    return path


@pytest.fixture
def ds(tmp_path):
    return Dataset(tmp_path / "datasets")


def _label(clip: Clip, *types: str) -> None:
    events = [{"fighter": "A", "type": t, "t": 0.1 * i} for i, t in enumerate(types)]
    clip.labels_path.write_text(json.dumps({"events": events}))


def _fake_run(clip: Clip, runs, stamp: bool = True):
    run = clip.run_dir(runs)
    run.mkdir(parents=True)
    (run / "events.json").write_text("[]")
    if stamp:
        clip.stamp_run(run)
    return run


def test_add_writes_metadata_and_copies(ds, video):
    clip = ds.add(video, "boxing-gym-2026-10-04-r1", "boxing", camera="side-on")
    assert video.exists()  # copied, not moved
    assert clip.video.name == "video.mp4" and clip.video.read_bytes() == video.read_bytes()
    reloaded = ds.get("boxing-gym-2026-10-04-r1")
    assert reloaded.meta == clip.meta
    m = reloaded.meta
    assert (m.discipline, m.split, m.camera, m.fps, m.width, m.height) == (
        "boxing",
        "train",
        "side-on",
        30.0,
        64,
        48,
    )
    assert len(m.sha256) == 64
    assert [c.id for c in ds.clips()] == ["boxing-gym-2026-10-04-r1"]


def test_add_move(ds, video):
    clip = ds.add(video, "a", "muay_thai", move=True)
    assert not video.exists() and clip.video.exists()


def test_add_rejects_bad_input_without_leaving_a_folder(ds, video, tmp_path):
    with pytest.raises(ClipError, match="invalid clip id"):
        ds.add(video, "Bad Id", "boxing")
    with pytest.raises(ValueError, match="unknown discipline"):
        ds.add(video, "x", "karate")
    not_video = tmp_path / "notes.mp4"
    not_video.write_text("hello")
    with pytest.raises(FileNotFoundError):
        ds.add(not_video, "y", "boxing")
    assert ds.clips() == []
    ds.add(video, "z", "boxing")
    with pytest.raises(ClipError, match="already exists"):
        ds.add(video, "z", "boxing")


def test_verify_video_catches_missing_and_changed_files(ds, video):
    clip = ds.add(video, "c", "boxing")
    clip.verify_video()
    with clip.video.open("ab") as f:
        f.write(b"\0")
    with pytest.raises(ClipError, match="does not match the sha256"):
        clip.verify_video()
    clip.video.unlink()
    with pytest.raises(ClipError, match="video missing"):
        clip.verify_video()


def test_clip_toml_is_strict(ds, video):
    clip = ds.add(video, "c", "boxing")
    toml = clip.dir / "clip.toml"
    toml.write_text(toml.read_text() + 'camra = "typo"\n')
    with pytest.raises(ClipError, match="camra"):
        ds.get("c")


def test_problems_flags_moves_outside_the_discipline(ds, video):
    clip = ds.add(video, "c", "boxing")
    _label(clip, "jab", "punch", "teep")
    assert any("teep" in p for p in clip.problems())
    _label(clip, "jab", "punch")
    assert clip.problems() == []


def test_run_stamps(ds, video, tmp_path):
    runs = tmp_path / "runs"
    clip = ds.add(video, "c", "boxing")
    assert clip.run_status(clip.run_dir(runs)) == "none"
    with pytest.raises(ClipError, match="sparring analyze c"):
        clip.require_run(clip.run_dir(runs))
    run = _fake_run(clip, runs, stamp=False)
    assert clip.run_status(run) == "unstamped"
    clip.stamp_run(run)
    assert clip.require_run(run) == run
    stale = Clip(clip.dir, clip.meta.model_copy(update={"sha256": "0" * 64}))
    assert stale.run_status(run) == "stale"


def test_resolve(ds, video, tmp_path):
    clip = ds.add(video, "c", "boxing")
    assert ds.resolve("c").id == "c"
    assert ds.resolve(clip.dir).id == "c"
    assert ds.resolve(str(video)) is None  # a plain video file
    with pytest.raises(ClipError, match="no clip 'nope'"):
        ds.resolve("nope")


def test_training_data_selection(ds, video, tmp_path):
    runs = tmp_path / "runs"
    train = ds.add(video, "train-clip", "boxing")
    test = ds.add(video, "test-clip", "boxing", split="test")
    ds.add(video, "unlabelled", "boxing")
    for c in (train, test):
        _label(c, "jab")
        _fake_run(c, runs)
    pick = lambda **kw: [n for n, _, _ in _training_data(dataset=ds.root, runs=runs, **kw)]  # noqa: E731
    assert pick(data=None, clips=None, split=None) == ["test-clip", "train-clip"]
    assert pick(data=None, clips=None, split="train") == ["train-clip"]
    assert pick(data=None, clips=["test-clip"], split="train") == ["test-clip"]  # explicit wins
    assert pick(data=["r:l.json"], clips=None, split=None) == ["r"]
    _label(ds.get("unlabelled"), "jab")
    with pytest.raises(ClipError, match="no analysis"):
        pick(data=None, clips=None, split=None)


def test_cli_clip_commands(ds, video, tmp_path):
    runner = CliRunner()
    env = {"SPARRING_DATASET": str(ds.root), "SPARRING_RUNS": str(tmp_path / "runs")}
    r = runner.invoke(app, ["clip", "list"], env=env)
    assert r.exit_code == 0 and "no clips" in r.output
    r = runner.invoke(app, ["clip", "add", str(video), "--discipline", "boxing"], env=env)
    assert r.exit_code == 0, r.output
    assert "added boxing-sparring-round-1" in r.output and "under 50 fps" in r.output
    r = runner.invoke(app, ["clip", "list"], env=env)
    assert "boxing-sparring-round-1  boxing" in r.output and r.output.rstrip().endswith("none")
    assert runner.invoke(app, ["clip", "check"], env=env).exit_code == 0
    r = runner.invoke(app, ["eval", "boxing-sparring-round-1"], env=env)
    assert r.exit_code == 1 and "no labels yet" in r.output
    r = runner.invoke(app, ["analyze", "nope"], env=env)
    assert r.exit_code == 1 and "no clip 'nope'" in r.output
    r = runner.invoke(app, ["analyze", str(video)], env=env)
    assert r.exit_code == 1 and "--out is required" in r.output
