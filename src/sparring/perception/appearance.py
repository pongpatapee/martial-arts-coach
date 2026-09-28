"""Colour-based appearance signatures for telling the two fighters apart.

Sparring partners almost always differ in headgear, shirt, shorts or gloves, so a coarse HSV
histogram of the head, torso and leg bands of the bounding box is a cheap and robust identity
cue. It is also interpretable, which lets the report say "fighter A = gold headgear, white top".
"""

from __future__ import annotations

import cv2
import numpy as np

# (top, bottom) of each band as a fraction of bbox height; horizontally we keep the centre.
REGIONS: dict[str, tuple[float, float]] = {
    "head": (0.0, 0.16),
    "torso": (0.18, 0.48),
    "legs": (0.48, 0.72),
}
REGION_WEIGHTS = np.array([0.4, 0.4, 0.2])
H_BINS, S_BINS, V_BINS = 10, 3, 3
BINS_PER_REGION = H_BINS * S_BINS * V_BINS
DESCRIPTOR_SIZE = BINS_PER_REGION * len(REGIONS)


def describe(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """L1-normalised HSV histograms of the head/torso/leg bands, concatenated."""
    h_img, w_img = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    out = np.zeros(DESCRIPTOR_SIZE, dtype=np.float32)
    for r, (top, bottom) in enumerate(REGIONS.values()):
        ya = int(np.clip(y1 + top * bh, 0, h_img))
        yb = int(np.clip(y1 + bottom * bh, 0, h_img))
        xa = int(np.clip(x1 + 0.25 * bw, 0, w_img))
        xb = int(np.clip(x2 - 0.25 * bw, 0, w_img))
        if yb - ya < 2 or xb - xa < 2:
            continue
        hsv = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]
        ).ravel()
        total = hist.sum()
        if total > 0:
            out[r * BINS_PER_REGION : (r + 1) * BINS_PER_REGION] = hist / total
    return out


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Weighted Hellinger distance in [0, 1]. Broadcasts over leading dims of `a`."""
    a = a.reshape(*a.shape[:-1], len(REGIONS), BINS_PER_REGION)
    b = b.reshape(*b.shape[:-1], len(REGIONS), BINS_PER_REGION)
    bc = np.sqrt(a * b).sum(-1)  # Bhattacharyya coefficient per region
    d = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
    return (d * REGION_WEIGHTS).sum(-1) / REGION_WEIGHTS.sum()


_HUE_NAMES = [
    (10, "red"),
    (22, "orange"),
    (35, "gold/yellow"),
    (85, "green"),
    (130, "blue"),
    (160, "purple"),
    (180, "red"),
]


def colour_names(descriptor: np.ndarray) -> dict[str, str]:
    """Dominant colour name per region, e.g. {'head': 'red', 'torso': 'black'}."""
    names = {}
    per_region = descriptor.reshape(len(REGIONS), H_BINS, S_BINS, V_BINS)
    for region, hist in zip(REGIONS, per_region, strict=True):
        h, s, v = np.unravel_index(int(np.argmax(hist)), hist.shape)
        if v == 0:
            names[region] = "black"
        elif s == 0:
            names[region] = "white" if v == V_BINS - 1 else "grey"
        else:
            hue = (h + 0.5) * 180 / H_BINS
            names[region] = next(n for limit, n in _HUE_NAMES if hue <= limit)
    return names
