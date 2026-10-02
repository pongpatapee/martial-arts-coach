# martial-arts-coach

Video analysis for **sparring**, for boxing, kickboxing and muay thai. Give it a phone video of two
people sparring. It finds the two fighters, tracks their pose, and detects:
- strikes: punches, kicks, knees, elbows,
- defensive responses: slips, ducks, blocks, checks,
- technical faults.

It then writes a **coaching report** for each fighter. Each point says what was seen, why it
matters and how to fix it, with drills and the timestamps to watch.

```
uv run sparring analyze boxing-sample                     # a clip from the dataset -> runs/boxing-sample
uv run sparring analyze fight.mp4 --out runs/mt --discipline muay_thai   # any video file
```

Disciplines: `boxing` (default), `kickboxing`, `muay_thai`. Output in the run directory:

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

## Web app

```
uv run sparring serve            # http://127.0.0.1:8000
```

Open the page, drop in a video, and pick the discipline and detection method. The page shows
progress stage by stage; a minute of video takes a few minutes. Then review the analysis:

- **Player:** play/pause, speeds from 0.1x to 2x, frame-by-frame stepping, ±0.5 s jumps, and
  previous/next event. It shows a time and frame readout, a skeleton overlay synced per frame,
  and a list of the events happening now.
- **Timeline:** one lane per fighter, coloured by strike / defence / fault. Hover for details,
  click to seek, scroll to zoom, drag to pan.
- **Events** tab: filter by fighter, kind, move or confidence. The list follows the playhead.
- **Highlights** tab: key moments (combinations, clean strikes, defences, coaching evidence).
  **Play highlights** runs through them slowed down.
- **Fighter A / B** tabs: the coaching, with clickable "watch" timestamps, drills and numbers.
- **Re-analyse:** switch discipline or method on a finished analysis. Only detection and
  coaching re-run, which takes seconds.

Keyboard: `Space` play, `←/→` frame, `Shift+←/→` 0.5 s, `[`/`]` event, `-`/`=` speed,
`S` skeleton, `H` highlights, `Z`/`X` zoom, `?` help.

Uploads and results are stored under `data/` (`--data` to change). Jobs run one at a time in a
separate process and resume after a restart. The server listens on localhost by default. To
share it with teammates, run `--host 0.0.0.0` behind a reverse proxy that adds authentication:
the app itself has no accounts yet.

**Frontend development** (hot reload; API calls are proxied to `sparring serve` on :8000):

```
uv run sparring serve --reload &
cd web && pnpm install && pnpm dev          # http://localhost:5173
pnpm test && pnpm build                     # unit tests; production build into src/sparring/web/static
```

**API:**

| Endpoint | Purpose |
|---|---|
| `GET /api/options` | Disciplines, methods and the move taxonomy |
| `POST /api/jobs` | Upload a video (`file`, `discipline`, `method`, optional `fighters`) |
| `GET /api/jobs`, `GET /api/jobs/{id}` | List jobs; one job with progress |
| `POST /api/jobs/{id}/reanalyze` | Re-run detection and report with new options |
| `POST /api/jobs/{id}/retry` | Re-queue a failed job |
| `DELETE /api/jobs/{id}` | Delete a job |
| `GET /api/jobs/{id}/analysis` | Report, events and video info |
| `GET /api/jobs/{id}/poses` | Keypoints as int16 binary; shape in `X-Shape` |
| `GET /api/jobs/{id}/video`, `GET /api/jobs/{id}/thumbnail` | Playback file (range requests) and thumbnail |

## Sample footage (the clip dataset)

Footage for testing, labelling and training lives in `datasets/clips/`, one folder per clip:

```
datasets/clips/boxing-homegym-2026-10-04-r2/
  video.mp4      the footage (git-ignored)
  clip.toml      discipline, split, camera, source, and the video's SHA-256 (committed)
  labels.json    your hand labels, once you've made them (committed)
```

```
uv run sparring clip add ~/Movies/round2.mp4 --discipline boxing --id boxing-homegym-2026-10-04-r2 \
    --recorded 2026-10-04 --camera "side-on, chest height" --source "own gym, both consented"
uv run sparring analyze boxing-homegym-2026-10-04-r2   # -> runs/boxing-homegym-2026-10-04-r2/
uv run sparring label boxing-homegym-2026-10-04-r2     # detector suggestions included
uv run sparring clip list                              # labels and analysis status per clip
uv run sparring clip check                             # verify every video against its hash
```

- **Ids**: `<discipline>-<place>-<date>[-r<round>]`, lowercase with hyphens.
- **Videos are not in git.** Clips are large and usually show other people. Back them up
  separately (a private bucket, or DVC later). `clip check` reports any that are missing.
- **Never re-encode or trim a clip's video.** Labels point at exact frames, so every command that
  scores or trains checks the video's hash and refuses a changed file. Add a new clip instead.
- **`split = "test"`** in `clip.toml` keeps a clip out of `sparring train`, so it can measure
  how the engine does on footage it has never seen. Set it when you add the clip and leave it.
- Runs are stamped with the video they came from; `clip list` shows `stale` if the video changed.

## Commands

Commands that take a `CLIP` also accept a clip folder. `analyze` and `label` also take a plain
video file (then `--out` / `--labels` are required); `eval` also takes `RUN LABELS`.

| Command | What it does |
|---|---|
| `sparring analyze CLIP [--discipline D] [--method M]` | Full pipeline into `runs/<clip>`. `--from-stage events` re-runs only detection + report (seconds). |
| `sparring clip add VIDEO --discipline D [--id ID] [--split test] ...` | Add a video to the dataset (copies it; `--move` to move). |
| `sparring clip list` / `sparring clip check` | Status of every clip / verify videos and labels. |
| `sparring serve [--data DIR] [--host H] [--port P]` | The web app (see above). |
| `sparring coach RUN [--llm]` | Print each fighter's coaching summary. `--llm` writes `coach_note.md`, a coach-voice rewrite by Claude that only uses facts from the report. Needs the `llm` extra and Anthropic credentials. |
| `sparring label CLIP` | Local browser labelling tool, saving to the clip's `labels.json`. The detector's predictions from the clip's run appear as suggestions you accept with Enter. |
| `sparring eval CLIP` | Score the clip's run against its labels. |
| `sparring bench [--clip ID ...] --methods rules,ensemble,...` | Cross-validated comparison of detection methods (default: all labelled clips). |
| `sparring train --out model.joblib [--clip ID ...] [--method ensemble]` | Train a learned detector (default: labelled clips in the `train` split). Use it with `detection.model_path` in a config TOML. |

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
- **One clip, labelled by Claude.** The labels in `datasets/clips/boxing-sample/labels.json` came from frame sheets and
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
sparring clip add new.mp4 --discipline boxing --id boxing-gym-2026-10-04-r1
sparring analyze boxing-gym-2026-10-04-r1
sparring label boxing-gym-2026-10-04-r1                  # accept/fix suggestions
sparring bench --methods rules,ensemble,temporal         # all labelled clips
sparring train --out src/sparring/assets/boxing_punches_ensemble.joblib   # train-split clips
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
uv run pytest            # fast tests (engine on synthetic skeletons, web API with a fake pipeline)
uv run pytest -m slow    # end-to-end on the boxing-sample clip (video not in git; downloads models)
uv run ruff check && uv run ruff format
```
