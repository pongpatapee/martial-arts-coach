"""Render labelled contact sheets of a time range to help hand-label events.

Each tile is a crop around both fighters with the timestamp burned in, so you can read off event
times (e.g. the frame of full extension of a jab) and write them into labels/<video>.json.

    uv run python scripts/contact_sheet.py runs/boxing 10.0 12.0 --fps 15 --out sheet.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sparring.io.video import VideoInfo

COLS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("start", type=float)
    ap.add_argument("end", type=float)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--tile-h", type=int, default=420)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    info = VideoInfo.load(args.run_dir / "video.json")
    boxes = pd.read_parquet(args.run_dir / "fighters.parquet")
    step = max(1, round(info.fps / args.fps))
    wanted = list(range(int(args.start * info.fps), int(args.end * info.fps), step))

    # One crop window for the whole range: union of both fighters' boxes, padded.
    sel = boxes[boxes.frame.isin(wanted)]
    x1, y1 = sel.x1.min(), sel.y1.min()
    x2, y2 = sel.x2.max(), sel.y2.max()
    pad = 0.05 * (y2 - y1)
    x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
    x2, y2 = int(min(info.width, x2 + pad)), int(min(info.height, y2 + pad))

    cap = cv2.VideoCapture(info.path)
    tiles = []
    for fr in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            break
        crop = img[y1:y2, x1:x2]
        tw = int(crop.shape[1] * args.tile_h / crop.shape[0])
        crop = cv2.resize(crop, (tw, args.tile_h))
        label = f"{fr / info.fps:.2f}s"
        cv2.putText(crop, label, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(crop, label, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        tiles.append(crop)
    while len(tiles) % COLS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + COLS]) for i in range(0, len(tiles), COLS)]
    cv2.imwrite(str(args.out), np.vstack(rows))


if __name__ == "__main__":
    main()
