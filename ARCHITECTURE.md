# Architecture

The pipeline turns a sparring video into a pose track for exactly two fighters, a timeline of
events (punches, defences, faults), and a coaching report.

```
video ─► ingest ─► track ─► fighters ─► pose ─► kinematics ─► events ─► report ─► render
          │          │          │          │          │            │          │         │
     video.json  tracks.parquet  fighters.parquet  poses.npz  features.parquet  events.json  report.{json,md}  overlay.mp4
                 appearance.npy  fighters.json     poses_clean.npy
```

Every stage reads only earlier stages' artifacts from the run directory
(`src/sparring/pipeline.py`), so you can re-run any suffix with `--from-stage`. GPU/CPU inference
(track, pose) takes a few minutes. Everything after it takes seconds, which is what makes
threshold tuning practical.

## Stages

| Stage | Module | What it does |
|---|---|---|
| ingest | `io/video.py` | Probes fps/size/frame count by decoding, and warns below 50 fps. |
| track | `perception/detector.py`, `perception/appearance.py` | YOLO11 person detection + BoT-SORT tracking with camera-motion compensation, plus a colour signature per detection. |
| fighters | `perception/fighters.py` | Picks the two fighters and assigns A/B identities frame by frame. |
| pose | `perception/pose.py` | Top-down RTMPose-x (Halpe-26, feet included) on each fighter's box, via `rtmlib`/ONNX Runtime. |
| kinematics | `kinematics/{clean,normalize,features}.py` | Confidence masking, gap filling, One Euro smoothing, body-relative frames, and the per-frame feature table. |
| events | `events/*.py` | Rule-based detectors that emit `Event`s. |
| report | `analysis/{metrics,report}.py` | Per-fighter metrics, then ranked strengths and improvement areas with evidence timestamps. |
| render | `render/overlay.py` | Skeletons, labels and an event timeline strip, re-encoded to H.264 with the original audio. |

## Key decisions

**Tracker IDs are not identities.** On the sample clip BoT-SORT gives fighter A five different
track IDs, because IDs switch in clinches, at crossings and during camera pans. `fighters.py`
works as follows:

1. It seeds the fighters from the track pair that most often appears large and close together.
2. It builds an HSV-histogram prototype for the head, torso and leg bands of each fighter.
3. It solves a gated Hungarian assignment in every frame. The cost is appearance plus motion
   continuity plus a small tracker-ID tie-breaker.
4. It rebuilds the prototypes from confident assignments and repeats.

The tracker-ID weight (0.08) is deliberately below the measured same-vs-other appearance margin
(10th percentile of about 0.1). Otherwise a tracker ID swap would carry the wrong fighter along.
A is whichever fighter is on the left when both first appear.

**Body-relative coordinates.** In `kinematics/normalize.py`, every feature uses a frame with the
origin at the hip midpoint, +x pointing horizontally at the opponent, +y up, and one torso length
(TL) as the unit. This cancels hand-held camera pan, distance to the camera, and which side of
the screen each fighter is on. Torso length is a rolling 80th percentile, because bending and
turning only ever shorten its projection.

**Lead/rear, not left/right.** Stance comes from which ankle is nearer the opponent, with a
rolling median and hysteresis. Detectors therefore reason about the lead and rear hand, so a jab
is always a lead-hand straight.

**Smoothing that keeps punches.** The One Euro filter's cutoff rises with speed
(`beta = 4` in body-heights per second). A still joint gets about 1 Hz of smoothing, while a
punch at about 2 body-heights per second gets about 9 Hz. On the sample, `beta = 0.3` flattened
a jab's wrist-speed peak from 8.7 to 5.4 TL/s.

**Punches are found mainly from the elbow opening, not from wrist speed.** On a phone video many
punches travel partly toward the camera. Their 2D wrist speed and reach are foreshortened to
2–3 TL/s, which overlaps ordinary guard adjustments. The 2D elbow angle still visibly snaps from
about 60° to over 150°. `events/punches.py` therefore uses two routes:
- **Elbow-opening route:** detects straights and uppercuts from the elbow opening.
- **Fast bent-arm swing route:** a fast wrist with the elbow bent and raised detects hooks.

Candidates are gated on travelling toward the opponent or upward, then classified by elbow angle
and travel direction.

**Defence depends on the opponent's punch.** `events/defense.py` classifies the defender's
response to each opponent punch thrown in range, in this order: duck, slip, block, step back,
no defence. The slip threshold is 0.4 TL, about 20 cm. It was calibrated against head motion in
windows with no incoming punch: the median there is 0.17–0.24 TL, and the median while a punch
comes in is 0.45 TL.

**Faults are sustained states or linked to a punch.** Guard drops, chin up and square stance are
debounced runs, with short gaps merged. Slow return and over-lean are attached to individual
punches. The report treats a hand that is low most of the time as a habit ("carries the lead hand
low: 67% of in-range time") rather than as a count of drops.

**The event seam.** Every detector emits `events.base.Event`
(fighter, type, start/peak/end, confidence, evidence). `events/detect.py` is the single place that
composes detectors, so a learned model can replace or augment any of them without touching
metrics, report or render.

## Configuration

All thresholds live in `config.py` as a pydantic model, each with its unit and rationale. You can
override any section with a TOML file:

```toml
# tuning.toml
[punch]
min_elbow_opening_deg = 35.0

[defense]
slip_min_head_disp_tl = 0.35
```

`uv run sparring analyze video.mp4 --out runs/x --config tuning.toml --from-stage events`

The resolved config is written to `runs/<name>/config.json` on every run.

## Evaluation

`labels/<video>.json` holds hand labels: fighter, type and time, plus the labelled time segment.
`sparring eval` matches predictions to labels one-to-one within ±150 ms. A sustained fault
matches when the label falls inside its interval. The eval prints precision, recall and F1 per
type, plus a coarse "punch (any)" score that separates detection errors from classification
errors. `scripts/contact_sheet.py` renders timestamped crop grids that make labelling practical.

## Extension points

- **Learned punch/defence classifier:** a skeleton action model (ST-GCN/CTR-GCN/PoseC3D) or a
  small temporal CNN/transformer over `features.parquet` windows can be added behind the `Event`
  seam. The rule detectors can bootstrap its training labels.
- **3D lifting:** e.g. MotionBERT between `pose` and `kinematics`. This would remove the
  foreshortening problems described above.
- **Apache-licensed detector:** RTMDet/YOLOX from `rtmlib` behind the `PersonTracker` protocol,
  replacing Ultralytics, which is AGPL-3.0.
- **LLM coaching prose:** `report.json` is flat and explicit, so it can be fed to an LLM as-is.
