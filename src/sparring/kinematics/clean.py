"""Keypoint cleaning: confidence masking, short-gap interpolation and One Euro smoothing.

One Euro (Casiez et al. 2012) adapts its cutoff to signal speed: it smooths jitter when a joint is
still but barely lags when it moves fast, so punches keep their sharp velocity peaks where a moving
average would smear them.
"""

from __future__ import annotations

import math

import numpy as np

from sparring.config import CleanConfig


def fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_gap that have valid values on both sides."""
    x = x.copy()
    nan = np.isnan(x)
    if not nan.any() or nan.all():
        return x
    idx = np.flatnonzero(nan)
    # Split NaN indices into contiguous runs.
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        a, b = run[0] - 1, run[-1] + 1
        if len(run) > max_gap or a < 0 or b >= len(x):
            continue
        x[run] = np.interp(run, [a, b], [x[a], x[b]])
    return x


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(
    x: np.ndarray, fps: float, min_cutoff: float, beta: float, d_cutoff: float
) -> np.ndarray:
    """One Euro filter over a 1D series. NaNs pass through and reset the filter state."""
    out = np.full_like(x, np.nan)
    dt = 1.0 / fps
    a_d = _alpha(d_cutoff, dt)
    prev = None
    dx_prev = 0.0
    for i, v in enumerate(x):
        if np.isnan(v):
            prev = None
            continue
        if prev is None:
            out[i] = prev = v
            dx_prev = 0.0
            continue
        dx = (v - prev) / dt
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        a = _alpha(min_cutoff + beta * abs(dx_hat), dt)
        prev = a * v + (1 - a) * prev
        out[i] = prev
        dx_prev = dx_hat
    return out


def clean_poses(
    keypoints: np.ndarray, scores: np.ndarray, fps: float, cfg: CleanConfig
) -> np.ndarray:
    """(T, F, K, 2) raw pixels -> cleaned pixels, NaN where unreliable.

    The One Euro speed term is computed in pixels/s, so beta is scaled by the typical body size to
    make the filter behave the same regardless of resolution or distance to camera.
    """
    kp = keypoints.astype(np.float64).copy()
    kp[scores < cfg.min_keypoint_conf] = np.nan
    t, n_f, n_k, _ = kp.shape
    for f in range(n_f):
        # Typical body height in pixels for this fighter (nose to ankles, robust).
        span = np.nanmedian(np.abs(kp[:, f, 0, 1] - np.nanmean(kp[:, f, 15:17, 1], axis=1)))
        scale = float(span) if np.isfinite(span) and span > 0 else 500.0
        for k in range(n_k):
            for d in range(2):
                series = fill_short_gaps(kp[:, f, k, d], cfg.max_gap_frames)
                kp[:, f, k, d] = scale * one_euro(
                    series / scale,
                    fps,
                    cfg.one_euro_min_cutoff,
                    cfg.one_euro_beta,
                    cfg.one_euro_d_cutoff,
                )
    return kp.astype(np.float32)
