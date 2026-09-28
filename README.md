# martial-arts-coach

Video analysis for **boxing sparring**. Give it a phone video of two people sparring. It finds
the two fighters, tracks their pose, detects punches, defensive responses and technical faults,
and writes a coaching report with timestamps to watch.

```
uv run sparring analyze test/video/boxing.mp4 --out runs/boxing
```

produces, in `runs/boxing/`:

| File | Contents |
|---|---|
| `report.md` / `report.json` | Per-fighter metrics, top strengths and improvement areas, each with evidence timestamps |
| `events.json` | Timeline of every detected event: fighter, type, start/peak/end, confidence, evidence |
| `overlay.mp4` | The video with skeletons, fighter labels, lead/rear hands, event labels and a timeline strip |
| `features.parquet` | Per-frame, per-fighter body-relative features (the input to all detectors) |
| `tracks.parquet`, `fighters.parquet`, `poses.npz`, ... | Intermediate stage artifacts (see [ARCHITECTURE.md](ARCHITECTURE.md)) |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` (for the overlay's H.264 and audio).

```
uv sync
```

Model weights download automatically on first run:
- YOLO11m (Ultralytics, ~40 MB) goes to `models/`.
- RTMPose-x Halpe-26 (~180 MB) goes to `~/.cache/rtmlib`.

The pipeline runs on Apple Silicon (MPS), CUDA or CPU. The 52-second sample takes about 6 minutes
on an M5, mostly spent on pose.

## Usage

```
# Full run
uv run sparring analyze VIDEO --out runs/NAME

# Re-run only the cheap stages after changing thresholds (seconds, no inference)
uv run sparring analyze VIDEO --out runs/NAME --from-stage events

# Wrong people picked as the fighters? Pass two track ids (see tracks.parquet / fighters.json)
uv run sparring analyze VIDEO --out runs/NAME --from-stage fighters --fighters 3,7

# Override thresholds from a TOML file (sections mirror src/sparring/config.py)
uv run sparring analyze VIDEO --out runs/NAME --config tuning.toml --from-stage events

# Score against hand labels
uv run sparring eval runs/NAME labels/boxing.json
```

The stages are `ingest, track, fighters, pose, kinematics, events, report, render`. Use
`--from-stage` / `--to-stage` to run a subset.

**Who is A and who is B?** A is the fighter on the left when both first appear. The report
also records an approximate colour signature for each fighter.

### Labelling

To create or extend `labels/<video>.json`, render timestamped crop grids and read event times
off them:

```
uv run python scripts/contact_sheet.py runs/boxing 10 12 --fps 15 --out sheet.jpg
```

See `src/sparring/evaluation.py` for the label format.

## Development

```
uv run pytest            # fast unit tests on synthetic keypoint sequences (no models/video)
uv run pytest -m slow    # end-to-end on test/video/boxing.mp4 (git-ignored; downloads models)
uv run ruff check && uv run ruff format
```

The detector tests build synthetic side-view skeletons (`tests/synthetic.py`): jabs, one-twos,
slips, ducks, dropped hands and arms left out after punching. The expected events are asserted
exactly.

## What it detects

- **Punches:** jab, cross, lead/rear hook, lead/rear uppercut, and combinations.
- **Defence** (the response to each opponent punch thrown in range): slip, duck, block, step
  back, no defence.
- **Faults:** guard drop (per hand), slow return to guard, chin up, square stance, over-leaning.
- **Metrics:**
  - punches per minute and punch mix
  - jab share and combination share
  - return-to-guard rate
  - defence rate and head-movement rate
  - head movement after punching
  - time the hands are held low

## Results on the sample clip

Sample clip: 52 s, portrait, 30 fps, hand-held, busy gym background.

- **Fighter identity:** correct on every frame spot-checked. That includes 24 evenly spaced
  frames and the frame after each of the 13 episodes where the two fighters' boxes overlap.
  Both fighters are covered in about 99% of frames, across 13 and 5 tracker-ID changes.
- **Punch detection:** scored against Claude-made labels for 0–6 s (`labels/boxing.json`,
  approximate and still to be verified by a human). Precision is 3/3 and recall is 3/9. The
  punches it misses are in a close-range exchange filmed from behind one fighter.

## Known limitations

These come from what was observed on the sample clip.

- **The labels are thin.** The only ground truth is 6 seconds of Claude-made labels. Treat the
  thresholds as provisional until a person labels a few rounds.
- **Single camera, 2D pose.** Punches thrown toward the camera are heavily foreshortened, and
  slips along the camera's depth axis barely register. When the camera is behind a fighter, or
  the fighters are overlapping at close range, recall drops sharply. 3D lifting or a second
  camera would help most.
- **30 fps is marginal.** A punch lasts 5–9 frames at 30 fps. Record at 60 fps.
- **Hooks and uppercuts** are the least reliable classes. In 2D a hook's arc is mostly depth.
- **Colour names** in the report are unreliable under warm gym lighting (everything lands in the
  red/orange bins). That's why fighters are identified by starting position. Identity *matching*
  still works, because it compares the two fighters' histograms rather than naming colours.
- **Landed vs. missed** punches are not detected.
- **Licence:** Ultralytics YOLO is AGPL-3.0. Swap in RTMDet/YOLOX (Apache-2.0) via `rtmlib`
  behind the `PersonTracker` protocol before any commercial use.

## Next steps

1. Label 2–3 full rounds, ideally by a coach and at 60 fps, and tune against them with
   `sparring eval`.
2. Add a learned temporal classifier behind the `Event` seam, bootstrapped from the rule
   detectors' output.
3. Add 3D pose lifting (e.g. MotionBERT) before the kinematics stage.
4. Feed `report.json` to an LLM to produce coach-style feedback.
