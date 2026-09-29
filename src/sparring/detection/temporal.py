"""Temporal CNN strike detector.

A small dilated 1D convolutional network reads a fighter's per-frame feature sequence and outputs
a per-frame "a strike lands now" heatmap; peaks above a threshold become strikes, and the hand and
type are then read off the geometry (`event_at`). Unlike the candidate classifier it sees the whole
motion context and needs no hand-made candidate generator, but it needs more data to shine.

Sequences are resampled to a fixed 30 fps so a model trained on one frame rate applies to another.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sparring.config import PunchConfig
from sparring.detection.methods import Session, best_threshold, nms
from sparring.events.base import Event
from sparring.events.punches import event_at

log = logging.getLogger(__name__)

MODEL_FPS = 30.0
HAND_CH = (
    ("elbow_deg", 1 / 180),
    ("ext", 1.0),
    ("wrist_x", 1.0),
    ("wrist_y", 1.0),
    ("wrist_vx", 0.1),
    ("wrist_vy", 0.1),
    ("elbow_above_shoulder", 1.0),
    ("wrist_below_shoulder", 1.0),
)
BODY_CH = (("head_vx", 0.1), ("head_vy", 0.1), ("distance_tl", 1 / 3))
OPP_CH = (("head_vx", 0.1), ("head_vy", 0.1))
SIGMA_S = 0.05


def _resample(df: pd.DataFrame, cols: list[str], t_new: np.ndarray) -> np.ndarray:
    t = df.t.to_numpy()
    out = np.zeros((len(cols), len(t_new)), np.float32)
    for i, c in enumerate(cols):
        x = df[c].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() >= 2:
            out[i] = np.interp(t_new, t[ok], x[ok])
    return out


def sequence(df: pd.DataFrame, opp: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """(channels, T) input at MODEL_FPS and its time axis."""
    t_new = np.arange(df.t.min(), df.t.max() + 1e-9, 1 / MODEL_FPS)
    chans, scales = [], []
    for hand in ("lead", "rear"):
        for c, s in HAND_CH:
            chans.append(f"{hand}_{c}")
            scales.append(s)
    for c, s in BODY_CH:
        chans.append(c)
        scales.append(s)
    x = _resample(df, chans, t_new) * np.array(scales, np.float32)[:, None]
    if opp is not None and not opp.empty:
        o = (
            _resample(opp, [c for c, _ in OPP_CH], t_new)
            * np.array([s for _, s in OPP_CH], np.float32)[:, None]
        )
    else:
        o = np.zeros((len(OPP_CH), len(t_new)), np.float32)
    vis = _resample(df, ["visible"], t_new)
    return np.clip(np.concatenate([x, o, vis]), -5, 5), t_new


def targets(t: np.ndarray, session: Session, fighter: str) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian heatmap around labelled strikes, and a mask of frames that count in the loss."""
    y = np.zeros(len(t), np.float32)
    mask = np.ones(len(t), np.float32)
    for lb in session.labels.events:
        if lb.fighter != fighter or not lb.is_strike:
            continue
        if lb.dont_care:
            mask[np.abs(t - lb.t) <= 0.2] = 0
        else:
            y = np.maximum(y, np.exp(-((t - lb.t) ** 2) / (2 * SIGMA_S**2)))
    for i in session.labels.ignore:
        if i["fighter"] == fighter:
            mask[(t >= i["from"]) & (t <= i["to"])] = 0
    return y, mask


def _net(c_in: int):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, ch: int, d: int):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, 3, padding=d, dilation=d)
            self.norm = nn.BatchNorm1d(ch)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            return x + self.drop(self.norm(self.conv(x)).relu())

    return nn.Sequential(
        nn.Conv1d(c_in, 32, 5, padding=2),
        nn.ReLU(),
        Block(32, 1),
        Block(32, 2),
        Block(32, 4),
        Block(32, 8),
        nn.Conv1d(32, 1, 1),
    )


@dataclass
class TemporalMethod:
    steps: int = 600
    crop: int = 96
    seed: int = 0
    threshold: float | None = None
    punch_cfg: PunchConfig = field(default_factory=PunchConfig)
    name: str = "temporal"
    net: object | None = None

    def _data(self, sessions: list[Session]):
        data = []
        for s in sessions:
            for f, df in s.features.items():
                if len(df) < 8:
                    continue
                opp = next((d for g, d in s.features.items() if g != f), None)
                x, t = sequence(df, opp)
                y, m = targets(t, s, f)
                data.append((x, y, m))
        return data

    def _train(self, data) -> object:
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        net = _net(data[0][0].shape[0])
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
        pos = sum(float((y > 0.5).sum()) for _, y, _ in data)
        tot = sum(len(y) for _, y, _ in data)
        pos_w = torch.tensor(min(50.0, (tot - pos) / max(pos, 1.0)))
        lengths = np.array([x.shape[1] for x, _, _ in data], float)
        net.train()
        for _ in range(self.steps):
            batch_x, batch_y, batch_m = [], [], []
            for _ in range(16):
                x, y, m = data[rng.choice(len(data), p=lengths / lengths.sum())]
                n = x.shape[1]
                a = rng.integers(0, max(1, n - self.crop))
                xs = x[:, a : a + self.crop].copy()
                xs *= rng.uniform(0.9, 1.1)
                xs += rng.normal(0, 0.03, xs.shape).astype(np.float32)
                pad = self.crop - xs.shape[1]
                batch_x.append(np.pad(xs, ((0, 0), (0, pad))))
                batch_y.append(np.pad(y[a : a + self.crop], (0, pad)))
                batch_m.append(np.pad(m[a : a + self.crop], (0, pad)))
            X = torch.tensor(np.stack(batch_x))
            Y = torch.tensor(np.stack(batch_y))
            M = torch.tensor(np.stack(batch_m))
            logits = net(X)[:, 0]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, Y, weight=M * (1 + (pos_w - 1) * Y), reduction="sum"
            ) / M.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        return net

    @staticmethod
    def _heat(net, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(x[None]))[0, 0]).numpy()

    def _events(self, session: Session, heat: dict, threshold: float) -> list[Event]:
        from scipy.signal import find_peaks

        out = []
        for f, (h, t) in heat.items():
            df = session.features[f]
            peaks, _ = find_peaks(h, height=threshold, distance=int(0.15 * MODEL_FPS))
            for p in peaks:
                frame = int(np.clip(np.searchsorted(df.t.to_numpy(), t[p]), 0, len(df) - 1))
                hand = self._hand_at(df, frame, session.fps)
                e = event_at(df, f, hand, frame, session.fps, self.punch_cfg, route=self.name)
                e.peak_t = float(t[p])
                e.confidence = round(float(h[p]), 3)
                out.append(e)
        return nms(out, 0.15)

    @staticmethod
    def _hand_at(df: pd.DataFrame, frame: int, fps: float) -> str:
        w = max(1, int(0.2 * fps))
        seg = df.iloc[max(0, frame - w) : frame + 2]
        score = {}
        for hand in ("lead", "rear"):
            elbow = seg[f"{hand}_elbow_deg"]
            reach = seg[f"{hand}_wrist_x"]
            score[hand] = np.nan_to_num(elbow.max() - elbow.min()) / 90 + np.nan_to_num(
                reach.max() - reach.min()
            )
        return max(score, key=score.get)

    def _heatmaps(self, net, session: Session) -> dict:
        out = {}
        for f, df in session.features.items():
            if len(df) < 8:
                continue
            opp = next((d for g, d in session.features.items() if g != f), None)
            x, t = sequence(df, opp)
            out[f] = (self._heat(net, x), t)
        return out

    def fit(self, sessions: list[Session]) -> TemporalMethod:
        if self.threshold is None:
            self.threshold = 0.5
            if len(sessions) >= 2:
                heats = []
                for i, s in enumerate(sessions):
                    net = self._train(self._data([o for j, o in enumerate(sessions) if j != i]))
                    heats.append(self._heatmaps(net, s))
                self.threshold = best_threshold(
                    sessions,
                    lambda s, th, i: self._events(s, heats[i], th),
                    np.arange(0.2, 0.95, 0.05),
                )
        self.net = self._train(self._data(sessions))
        log.info("temporal: trained, threshold %.2f", self.threshold)
        return self

    def detect(self, session: Session) -> list[Event]:
        return self._events(session, self._heatmaps(self.net, session), self.threshold)

    def save(self, path) -> None:
        import torch

        torch.save(
            {
                "method": self.name,
                "threshold": self.threshold,
                "c_in": self.net[0].in_channels,
                "state": self.net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, punch_cfg: PunchConfig | None = None) -> TemporalMethod:
        import torch

        d = torch.load(path, weights_only=True)
        m = cls(threshold=d["threshold"], punch_cfg=punch_cfg or PunchConfig())
        m.net = _net(d["c_in"])
        m.net.load_state_dict(d["state"])
        m.net.eval()
        return m
