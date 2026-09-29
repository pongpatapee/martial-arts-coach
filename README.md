# martial-arts-coach

Video analysis for **sparring**, for boxing, kickboxing and muay thai. Give it a phone video of two
people sparring. It finds the two fighters, tracks their pose, and detects:
- strikes: punches, kicks, knees, elbows,
- defensive responses: slips, ducks, blocks, checks,
- technical faults.

It then writes a **coaching report** for each fighter. Each point says what was seen, why it
matters and how to fix it, with drills and the timestamps to watch.

```
uv run sparring analyze test/video/boxing.mp4 --out runs/boxing                  # boxing (default)
uv run sparring analyze fight.mp4 --out runs/kb --discipline kickboxing
uv run sparring analyze fight.mp4 --out runs/mt --discipline muay_thai
```

Output in the run directory:

| File | Contents |
|---|---|
| `report.md` / `report.json` | Per-fighter metrics and coaching: summary, strengths, priorities (observation / why / how / drills / watch-timestamps / confidence), next-session plan |
| `events.json` | Every detected event: fighter, type, start/peak/end, confidence, evidence |
| `overlay.mp4` | Video with skeletons, fighter labels, lead/rear hands, event labels, timeline strip |
| `features.parquet` | Per-frame, per-fighter body-relative features (input to every detector) |
| `tracks.parquet`, `fighters.parquet`, `poses.npz`, ... | Intermediate stage artifacts (see [ARCHITECTURE.md](ARCHITECTURE.md)) |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```
uv sync                 # core
uv sync --extra llm     # optional: Claude-written coach's notes
```

Model weights download on first run:
- YOLO11m (~40 MB) goes to `models/`.
- RTMPose-x (~180 MB) goes to `~/.cache/rtmlib`.

The trained strike classifier ships in `src/sparring/assets/`. Runs on Apple Silicon, CUDA or
CPU. The 52-second sample takes about 6 minutes on an M5.

## Commands

| Command | What it does |
|---|---|
| `sparring analyze VIDEO --out RUN [--discipline D] [--method M]` | Full pipeline. `--from-stage events` re-runs only detection + report (seconds). |
| `sparring coach RUN [--llm]` | Print each fighter's coaching summary. `--llm` writes `coach_note.md`, a coach-voice rewrite by Claude that only uses facts from the report. Needs the `llm` extra and Anthropic credentials. |
| `sparring label VIDEO --labels labels/x.json [--run RUN]` | Local browser labelling tool. The detector's predictions from `RUN` appear as suggestions you accept with Enter. |
| `sparring eval RUN LABELS` | Score a run against labels. |
| `sparring bench --data RUN:LABELS [--data ...] --methods rules,ensemble,...` | Cross-validated comparison of detection methods. |
| `sparring train --data RUN:LABELS [...] --method ensemble --out model.joblib` | Train a learned detector. Use it with `detection.model_path` in a config TOML. |

To override thresholds, pass `--config tuning.toml`; its sections mirror `src/sparring/config.py`.

## How accurate is it?

Strike detection on the sample clip, cross-validated over 5 time blocks: train on 4/5, test on the
held-out 1/5. There are 75 labelled strikes.

| Method | Precision | Recall | F1 (± per-fold sd) | Strike type accuracy |
|---|---|---|---|---|
| **ensemble** (default) | **0.75** | 0.67 | **0.70** ± 0.09 | **0.75** |
| rules | 0.68 | 0.68 | 0.68 ± 0.11 | 0.70 |
| temporal CNN | 0.68 | 0.67 | 0.67 ± 0.14 | 0.71 |
| hybrid (learned + rule features) | 0.63 | 0.71 | 0.67 ± 0.08 | 0.53 |
| rules, grid-tuned | 0.69 | 0.61 | 0.65 ± 0.10 | 0.67 |
| learned (candidates + boosted trees) | 0.57 | 0.75 | 0.65 ± 0.08 | 0.68 |

Read these numbers with the caveats below:
- **One clip, labelled by Claude.** The labels in `labels/boxing.json` came from frame sheets and
  are not coach-verified. The per-fold spread (±0.09) is larger than the gaps between methods.
  With about 75 labelled strikes, the learned methods can't yet beat the rules by a meaningful
  margin. The ensemble is the default because it has the best precision and type accuracy, not
  because the data proves it best.
- **Evaluate the right numbers.** `sparring eval` on the sample clip reports about 0.80 F1. That is
  in-sample (the bundled model was trained on this clip), so use the cross-validated table above.
- **Only strikes are measured.** Defence and guard detections have only a handful of labels, so
  their accuracy is unmeasured. Kick, knee and elbow detectors are tested on synthetic motion and
  checked for false positives on the boxing clip (0 kicks found), but no kickboxing or muay thai
  footage has been scored yet. The report tags every coaching point with this confidence.

**Improving accuracy.** More labelled footage is the biggest lever, then higher frame rate. The
loop:

```
sparring analyze new.mp4 --out runs/new
sparring label new.mp4 --labels labels/new.json --run runs/new   # accept/fix suggestions
sparring bench --data runs/boxing:labels/boxing.json --data runs/new:labels/new.json --methods rules,ensemble,temporal
sparring train --data runs/boxing:labels/boxing.json --data runs/new:labels/new.json --out src/sparring/assets/boxing_punches_ensemble.joblib
```

## What it detects

| | Boxing | Kickboxing | Muay thai |
|---|---|---|---|
| Punches: jab, cross, lead/rear hook, lead/rear uppercut; combinations | ✓ | ✓ | ✓ |
| Kicks: lead/rear teep, low, body, head kick | | ✓ | ✓ |
| Knees, elbows, clinch | | | ✓ |
| Defence vs punches: slip, duck/roll, block, step back, no defence | ✓ | ✓ | ✓ |
| Defence vs leg kicks: check, step back, block, no defence | | ✓ | ✓ |
| Faults: hand held low, slow return to guard, chin up, square stance, over-reaching | ✓ | ✓ | ✓ |
| Faults: kicking with the guard down | | ✓ | ✓ |

**Coaching** is a knowledge base of about 25 rules in `src/sparring/coaching/knowledge.py`. Each
has a sample-size gate, an observation built from the fighter's own numbers, why it matters, the
technical fix, drills with a dose, and a confidence tag. Examples: slow return to guard, low
output, rarely jabs, one shot at a time, doesn't counter, relies on the high guard, head on the
centre line after attacking, not checking kicks, kicks with the hands down, not using the legs.

## Known limitations

- **Single camera, 2D pose.** Motion toward or away from the camera is foreshortened. Recall
  drops sharply when the camera is behind a fighter or the fighters overlap at close range. 3D
  lifting (RTMW3D) was tried and did not help (see ARCHITECTURE.md).
- **30 fps is marginal**; a punch is 5–9 frames. Record at 60 fps.
- **Hooks, uppercuts and strike type in general** are the weakest outputs (type accuracy 0.75).
- **Landed vs missed** is not detected.
- **Colour names** in `report.json` are unreliable under warm gym light. Fighters are identified
  by starting position (A = left).
- **Licence:** Ultralytics YOLO is AGPL-3.0. Swap in RTMDet/YOLOX (Apache-2.0) via `rtmlib`
  behind the `PersonTracker` protocol before any commercial use.

## Development

```
uv run pytest            # 58 fast tests on synthetic skeletons, no models or video
uv run pytest -m slow    # end-to-end on test/video/boxing.mp4 (git-ignored; downloads models)
uv run ruff check && uv run ruff format
```
