"""Pick the two fighters out of all tracked people and keep their identities stable.

Tracker IDs are not identities: they switch in clinches, when fighters cross, and when the camera
pans. We therefore:

1. seed the two fighters from the pair of tracks that most often appear large and close together
   (or from an explicit override),
2. build a colour appearance prototype for each,
3. assign detections to fighter A / B frame by frame with a cost that combines appearance,
   motion continuity and tracker-ID continuity (Hungarian assignment, gated),
4. rebuild the prototypes from the confident assignments and repeat.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from sparring.config import FighterConfig
from sparring.perception import appearance
from sparring.skeleton import FIGHTERS

_INF = 1e6


@dataclass
class FighterSelection:
    assignments: pd.DataFrame  # frame, fighter, det, track_id, x1, y1, x2, y2, conf, app_dist, cost
    seed_tracks: dict[str, int]
    colours: dict[str, dict[str, str]]
    prototypes: dict[str, np.ndarray] = field(repr=False)

    def summary(self, n_frames: int) -> dict:
        out = {"seed_tracks": self.seed_tracks, "colours": self.colours, "fighters": {}}
        for f in FIGHTERS:
            a = self.assignments[self.assignments.fighter == f].sort_values("frame")
            tids = a.track_id.to_numpy()
            out["fighters"][f] = {
                "coverage": round(len(a) / max(n_frames, 1), 3),
                "track_id_changes": int((tids[1:] != tids[:-1]).sum()) if len(tids) else 0,
                "mean_appearance_distance": round(float(a.app_dist.mean()), 3) if len(a) else None,
            }
        return out

    def save(self, run_dir: Path, n_frames: int) -> None:
        self.assignments.to_parquet(run_dir / "fighters.parquet", index=False)
        np.savez(run_dir / "fighter_prototypes.npz", **self.prototypes)
        (run_dir / "fighters.json").write_text(json.dumps(self.summary(n_frames), indent=2))

    @classmethod
    def load(cls, run_dir: Path) -> FighterSelection:
        meta = json.loads((run_dir / "fighters.json").read_text())
        protos = dict(np.load(run_dir / "fighter_prototypes.npz"))
        return cls(
            assignments=pd.read_parquet(run_dir / "fighters.parquet"),
            seed_tracks={k: int(v) for k, v in meta["seed_tracks"].items()},
            colours=meta["colours"],
            prototypes=protos,
        )


def _centers(df: pd.DataFrame) -> np.ndarray:
    return np.stack([(df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2], axis=1)


def seed_pair(dets: pd.DataFrame, frame_height: int) -> tuple[int, int]:
    """Track pair that most often appears both large and close together."""
    scores: dict[tuple[int, int], float] = defaultdict(float)
    tracked = dets[dets.track_id >= 0]
    for _, g in tracked.groupby("frame"):
        if len(g) < 2:
            continue
        c = _centers(g)
        h = (g.y2 - g.y1).to_numpy()
        ids = g.track_id.to_numpy()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                mean_h = (h[i] + h[j]) / 2
                d = np.linalg.norm(c[i] - c[j])
                s = (min(h[i], h[j]) / frame_height) * np.exp(-d / mean_h)
                key = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                scores[key] += s
    if not scores:
        raise RuntimeError("no pair of tracked people found; cannot identify fighters")
    return max(scores, key=scores.__getitem__)


def _prototype(desc: np.ndarray) -> np.ndarray:
    p = np.median(desc, axis=0)
    n = appearance.BINS_PER_REGION
    for r in range(len(appearance.REGIONS)):
        s = p[r * n : (r + 1) * n].sum()
        if s > 0:
            p[r * n : (r + 1) * n] /= s
    return p


def _initial_state(dets: pd.DataFrame, seeds: dict[str, int]) -> tuple[dict, dict]:
    """Start each fighter at its seed track's first detection, with the seed's typical height,
    so the very first frames cannot lock onto a similarly dressed bystander."""
    state, ref_h = {}, {}
    for f, tid in seeds.items():
        t = dets[dets.track_id == tid].sort_values("frame")
        first = t.iloc[0]
        state[f] = {
            "frame": int(first.frame),
            "center": np.array([(first.x1 + first.x2) / 2, (first.y1 + first.y2) / 2]),
            "h": float(first.y2 - first.y1),
            "track_id": tid,
        }
        ref_h[f] = float((t.y2 - t.y1).median())
    return state, ref_h


def _assign(
    dets: pd.DataFrame,
    desc: np.ndarray,
    protos: dict[str, np.ndarray],
    seeds: dict[str, int],
    cfg: FighterConfig,
) -> pd.DataFrame:
    rows = []
    state, ref_h = _initial_state(dets, seeds)
    by_frame = dets.groupby("frame").indices
    for frame in sorted(by_frame):
        idx = by_frame[frame]
        g = dets.iloc[idx]
        c = _centers(g)
        h = (g.y2 - g.y1).to_numpy()
        tids = g.track_id.to_numpy()
        app = np.stack([appearance.distance(desc[idx], protos[f]) for f in FIGHTERS])
        cost = np.full((len(FIGHTERS), len(g)), _INF)
        for fi, f in enumerate(FIGHTERS):
            s = state[f]
            recent = abs(frame - s["frame"]) <= cfg.reacquire_after_frames
            total = cfg.w_appearance * app[fi]
            if recent:
                jump = np.linalg.norm(c - s["center"], axis=1) / s["h"]
                total = total + cfg.w_motion * jump / cfg.max_jump_heights
                total = total + cfg.w_track_continuity * (tids != s["track_id"])
                total = np.where(jump > cfg.max_jump_heights, _INF, total)
            # Background people are smaller than the fighters; penalise shrinking.
            total = total + 0.5 * np.maximum(0.0, np.log(ref_h[f] / h))
            cost[fi] = total
        r_idx, c_idx = linear_sum_assignment(cost)
        for fi, di in zip(r_idx, c_idx, strict=True):
            if cost[fi, di] > cfg.max_cost:
                continue
            f = FIGHTERS[fi]
            state[f] = {"frame": frame, "center": c[di], "h": h[di], "track_id": tids[di]}
            ref_h[f] = 0.95 * ref_h[f] + 0.05 * h[di]
            row = g.iloc[di]
            rows.append(
                {
                    "frame": frame,
                    "fighter": f,
                    "det": int(idx[di]),
                    "track_id": int(tids[di]),
                    "x1": row.x1,
                    "y1": row.y1,
                    "x2": row.x2,
                    "y2": row.y2,
                    "conf": row.conf,
                    "app_dist": float(app[fi, di]),
                    "cost": float(cost[fi, di]),
                }
            )
    return pd.DataFrame(rows)


def select_fighters(
    dets: pd.DataFrame,
    desc: np.ndarray,
    frame_height: int,
    cfg: FighterConfig,
    override: tuple[int, int] | None = None,
) -> FighterSelection:
    pair = override or seed_pair(dets, frame_height)
    # Deterministic labelling: A is the seed track further left when both are first in view.
    missing = [t for t in pair if not (dets.track_id == t).any()]
    if missing:
        raise ValueError(f"track ids {missing} not found in detections")
    both = dets[dets.track_id.isin(pair)]
    counts = both.groupby("frame").track_id.nunique()
    first = counts.index[counts == 2].min() if (counts == 2).any() else both.frame.min()
    at = both[both.frame == first] if (counts == 2).any() else both.groupby("track_id").head(1)
    x = {int(r.track_id): (r.x1 + r.x2) / 2 for r in at.itertuples()}
    ordered = sorted(pair, key=x.__getitem__)
    seeds = dict(zip(FIGHTERS, ordered, strict=True))
    protos = {f: _prototype(desc[(dets.track_id == t).to_numpy()]) for f, t in seeds.items()}

    assignments = pd.DataFrame()
    for _ in range(cfg.refine_passes):
        assignments = _assign(dets, desc, protos, seeds, cfg)
        for f in FIGHTERS:
            a = assignments[assignments.fighter == f]
            confident = a[a.cost <= a.cost.quantile(0.5)]
            if len(confident) >= 10:
                protos[f] = _prototype(desc[confident.det.to_numpy()])

    return FighterSelection(
        assignments=assignments,
        seed_tracks={f: int(t) for f, t in seeds.items()},
        colours={f: appearance.colour_names(p) for f, p in protos.items()},
        prototypes=protos,
    )
