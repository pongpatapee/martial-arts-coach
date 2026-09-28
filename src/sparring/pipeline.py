"""Stage orchestration. Each stage reads the previous stages' artifacts from the run directory,
so any stage can be re-run in isolation (e.g. re-tune event thresholds without GPU inference)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sparring.config import Config
from sparring.io.video import VideoInfo, frames, probe
from sparring.skeleton import FIGHTERS

log = logging.getLogger(__name__)


@dataclass
class RunContext:
    video: Path
    run_dir: Path
    config: Config
    fighter_override: tuple[int, int] | None = None

    def path(self, name: str) -> Path:
        return self.run_dir / name

    @property
    def info(self) -> VideoInfo:
        return VideoInfo.load(self.path("video.json"))


def stage_ingest(ctx: RunContext) -> None:
    info = probe(ctx.video)
    info.save(ctx.path("video.json"))
    log.info(
        "video: %dx%d @ %.2f fps, %d frames (%.1fs)",
        info.width,
        info.height,
        info.fps,
        info.n_frames,
        info.duration_s,
    )
    if info.fps < ctx.config.ingest.min_recommended_fps:
        log.warning(
            "%.0f fps is low for punch detection: a punch lasts ~150-300 ms (%d-%d frames). "
            "Record at 60 fps if possible.",
            info.fps,
            round(0.15 * info.fps),
            round(0.3 * info.fps),
        )


def stage_track(ctx: RunContext) -> None:
    from sparring.perception import appearance
    from sparring.perception.detector import UltralyticsTracker

    tracker = UltralyticsTracker(ctx.config.detect, ctx.run_dir)
    rows, descs = [], []
    n = ctx.info.n_frames
    for fd in tracker.track(ctx.video):
        for box, tid, conf in zip(fd.boxes, fd.track_ids, fd.confs, strict=True):
            rows.append((fd.frame, int(tid), *box.tolist(), float(conf)))
            descs.append(appearance.describe(fd.image, box))
        if fd.frame % 100 == 0:
            log.info("track: frame %d/%d", fd.frame, n)
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
    df.to_parquet(ctx.path("tracks.parquet"), index=False)
    np.save(ctx.path("appearance.npy"), np.asarray(descs, np.float32))
    log.info("track: %d detections, %d track ids", len(df), df.track_id.nunique())


def stage_fighters(ctx: RunContext) -> None:
    from sparring.perception.fighters import select_fighters

    dets = pd.read_parquet(ctx.path("tracks.parquet"))
    desc = np.load(ctx.path("appearance.npy"))
    sel = select_fighters(
        dets, desc, ctx.info.height, ctx.config.fighters, override=ctx.fighter_override
    )
    sel.save(ctx.run_dir, ctx.info.n_frames)
    log.info("fighters: %s", json.dumps(sel.summary(ctx.info.n_frames)))


def stage_pose(ctx: RunContext) -> None:
    from sparring.perception.pose import PoseTrack, RTMPoseEstimator

    assign = pd.read_parquet(ctx.path("fighters.parquet"))
    estimator = RTMPoseEstimator(ctx.config.pose)
    n = ctx.info.n_frames
    track = PoseTrack.empty(n)
    by_frame = {f: g for f, g in assign.groupby("frame")}
    for i, img in frames(ctx.video):
        if i >= n:
            break
        g = by_frame.get(i)
        if g is None:
            continue
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        kpts, scores = estimator(img, boxes)
        for j, f in enumerate(g.fighter):
            fi = FIGHTERS.index(f)
            track.keypoints[i, fi] = kpts[j]
            track.scores[i, fi] = scores[j]
            track.boxes[i, fi] = boxes[j]
        if i % 100 == 0:
            log.info("pose: frame %d/%d", i, n)
    track.save(ctx.path("poses.npz"))


def stage_kinematics(ctx: RunContext) -> None:
    from sparring.kinematics.clean import clean_poses
    from sparring.kinematics.features import compute_features
    from sparring.perception.pose import PoseTrack

    raw = PoseTrack.load(ctx.path("poses.npz"))
    kpts = clean_poses(raw.keypoints, raw.scores, ctx.info.fps, ctx.config.clean)
    np.save(ctx.path("poses_clean.npy"), kpts)
    feats = compute_features(kpts, ctx.info.fps, ctx.config.kinematics)
    feats.to_parquet(ctx.path("features.parquet"), index=False)


def stage_events(ctx: RunContext) -> None:
    from sparring.events.base import save_events
    from sparring.events.detect import detect_all

    feats = pd.read_parquet(ctx.path("features.parquet"))
    events = detect_all(feats, ctx.info.fps, ctx.config)
    save_events(events, ctx.path("events.json"))
    counts: dict[str, int] = {}
    for e in events:
        counts[f"{e.fighter}:{e.type}"] = counts.get(f"{e.fighter}:{e.type}", 0) + 1
    log.info("events: %s", json.dumps(dict(sorted(counts.items()))))


def stage_report(ctx: RunContext) -> None:
    from sparring.analysis.report import build_report, render_markdown
    from sparring.events.base import load_events

    feats = pd.read_parquet(ctx.path("features.parquet"))
    events = load_events(ctx.path("events.json"))
    fighters_meta = json.loads(ctx.path("fighters.json").read_text())
    report = build_report(events, feats, ctx.info, fighters_meta, ctx.config)
    ctx.path("report.json").write_text(json.dumps(report, indent=2))
    ctx.path("report.md").write_text(render_markdown(report))


def stage_render(ctx: RunContext) -> None:
    from sparring.events.base import load_events
    from sparring.render.overlay import render_overlay

    render_overlay(
        video=ctx.video,
        out=ctx.path("overlay.mp4"),
        keypoints=np.load(ctx.path("poses_clean.npy")),
        features=pd.read_parquet(ctx.path("features.parquet")),
        events=load_events(ctx.path("events.json")),
        info=ctx.info,
    )


STAGES: dict[str, Callable[[RunContext], None]] = {
    "ingest": stage_ingest,
    "track": stage_track,
    "fighters": stage_fighters,
    "pose": stage_pose,
    "kinematics": stage_kinematics,
    "events": stage_events,
    "report": stage_report,
    "render": stage_render,
}


def run(ctx: RunContext, from_stage: str = "ingest", to_stage: str = "render") -> None:
    names = list(STAGES)
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"unknown stage; choose from {names}")
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    ctx.path("config.json").write_text(ctx.config.model_dump_json(indent=2))
    for name in names[names.index(from_stage) : names.index(to_stage) + 1]:
        t0 = time.perf_counter()
        log.info("stage %s ...", name)
        STAGES[name](ctx)
        log.info("stage %s done in %.1fs", name, time.perf_counter() - t0)
