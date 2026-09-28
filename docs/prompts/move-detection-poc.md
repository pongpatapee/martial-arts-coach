# Build prompt: Boxing sparring analysis — move-detection PoC

You are building the first version of a video analysis engine that watches **boxing sparring** footage and produces coaching feedback. The repo (`martial-arts-coach`, branch `move-detection-poc`) is empty. Build it from scratch. Prefer quality, simplicity, robustness, and long-term maintainability over development speed.

## Goal

Given a sparring video (single camera, usually a phone, 30–60 fps, often with other people in frame), output:

1. A per-frame, per-fighter **pose track** for exactly two fighters (A and B) with stable identities.
2. A timestamped **event timeline**: punches thrown (jab, cross, lead hook, rear hook, lead uppercut, rear uppercut), defensive moves (slip, duck/roll, block/guard-catch, step back), and faults (guard dropped, hand not returned to guard after punching, chin up, square stance, over-leaning past lead foot).
3. A **coaching report**: aggregate metrics per fighter plus the top strengths and weaknesses, each backed by evidence timestamps.
4. An **annotated video**: skeletons, fighter IDs, and event labels overlaid.

## Architecture

Python 3.12, managed with `uv`. Use `ruff` and `pytest`. Build a staged pipeline where each stage reads the previous stage's cached output from a run directory. That way you can tune the event detectors without re-running GPU inference.

```
video ──► 1. ingest ──► 2. detect+track ──► 3. fighter selection ──► 4. pose ──► 5. clean+normalize
      ──► 6. features ──► 7. event detection ──► 8. metrics+report ──► 9. render overlay
```

Layout suggestion (adjust if you have a better idea, and say why):

```
src/sparring/
  config.py          # pydantic settings; every threshold lives here, nothing hard-coded in detectors
  io/video.py        # decode, fps normalization, timestamps
  perception/
    detector.py      # PersonDetector protocol + YOLO impl
    tracker.py       # multi-object tracking (ByteTrack/BoT-SORT)
    fighters.py      # pick the 2 fighters, keep identities across ID switches
    pose.py          # PoseEstimator protocol + RTMPose impl
  kinematics/
    clean.py         # gap fill, confidence masking, smoothing (One Euro filter)
    normalize.py     # body-relative frame, scale by torso length, orientation vs opponent
    features.py      # per-frame feature table
  events/
    base.py          # Event dataclass, EventDetector protocol
    punches.py
    defense.py
    faults.py
    interaction.py   # links opponent punches to defensive responses
  analysis/
    metrics.py
    report.py        # JSON + Markdown report
  render/overlay.py
  cli.py             # `sparring analyze <video> --out runs/<name>` and `--from-stage N`
tests/
```

Stage artifacts go in `runs/<name>/`: `tracks.parquet`, `poses.parquet` (frame, fighter_id, keypoint, x, y, conf), `features.parquet`, `events.json`, `report.md`, `report.json`, `overlay.mp4`.

## Stage details

**1. Ingest.** Decode the video with PyAV or OpenCV. Keep the real timestamps. Downscale frames for inference but store coordinates in original pixel space. Warn if the video is under 50 fps: a punch lasts about 150–300 ms, which is only 5–9 frames at 30 fps.

**2. Detect and track.** Use YOLO (Ultralytics, `yolo11m`, person class) with its built-in BoT-SORT or ByteTrack tracker, behind a `PersonDetector` protocol. Ultralytics is AGPL-3.0. Keep it swappable so it can later be replaced with RTMDet/YOLOX via `rtmlib` (Apache-2.0). Must run on Apple Silicon (MPS) and on CUDA.

**3. Fighter selection and identity.** A gym video contains coaches, spectators, and other pairs. Choose the two fighters by scoring tracks on duration, motion energy, mutual proximity, and centrality. Also support an explicit override (`--fighters 3,7` or a bbox hint on the first frame). Tracker IDs will switch during clinches and crossings. Re-associate each tracklet to A or B with an appearance signature (HSV histogram of torso and glove regions, or a ReID embedding) plus motion continuity. Treat identity stability as a first-class requirement and test it.

**4. Pose.** Run top-down RTMPose on each fighter's bbox via `rtmlib` (ONNX Runtime, no mmcv install). Default to RTMPose-l on Halpe26 keypoints, which include feet for stance analysis, or COCO-17 if Halpe26 isn't available. Keep per-keypoint confidence. Put it behind a `PoseEstimator` protocol.

**5. Clean and normalize.** Mask low-confidence keypoints and interpolate short gaps (≤ 5 frames). Smooth with a One Euro filter, which keeps fast punches sharp; a moving average would smear them. Build a body-relative frame for each fighter:
- origin at the hip midpoint, scale = torso length (shoulder-mid to hip-mid);
- a "toward opponent" axis = vector to the opponent's head/torso;
- stance (orthodox or southpaw) from which foot is closer to the opponent, re-evaluated over time because fighters switch stances.

Label hands as **lead/rear**, not left/right.

**6. Features.** Build a per-frame table per fighter. Examples:
- arm extension ratio (wrist–shoulder distance / arm length);
- elbow angle;
- wrist velocity and acceleration, decomposed into the toward-opponent, vertical, and lateral axes;
- wrist height relative to nose/chin;
- head offset from the hip centerline (lateral and vertical);
- shoulder-line and hip-line rotation;
- torso lean;
- stance width and foot angle;
- inter-fighter distance (in torso lengths);
- whether each fighter is "in range".

**7. Event detection (rule-based first).** Each detector is a small, pure function over the feature table that emits `Event(fighter, type, start_t, peak_t, end_t, confidence, evidence: dict)`. Make thresholds configurable.
- **Punches:** segment at wrist-velocity peaks toward the opponent. Classify by hand (lead/rear) and trajectory. Straight = extension ratio goes above threshold with a mostly forward path. Hook = elbow stays bent (~70–110°), wrist moves laterally, and the torso rotates. Uppercut = dominant upward wrist velocity with a bent elbow. Jab = lead straight, cross = rear straight. Detect combinations as punches within ~400 ms of each other.
- **Defense (relational):** in `interaction.py`, for each opponent punch find this fighter's response during the punch window. Slip = lateral head displacement off the line of attack. Duck/roll = vertical head drop with knee bend, lateral for a roll. Block = the targeted-side glove is near the head at the punch's peak. Step back = distance increases. None = no response. A slip is only a slip if a punch is coming, so defense is defined relative to opponent events.
- **Faults:** guard dropped = a wrist below chin level for > N ms while in range. Hand not returned = after a punch, the wrist takes longer than X ms to return to guard height. Chin up = the head-pitch proxy. Square stance = stance width along the opponent axis falls below a threshold. Over-leaning = the head projects past the lead foot.

Landed vs. missed detection is **out of scope** for this PoC. At most, emit a low-confidence `possible_contact` hint when the opponent's head shows a sudden velocity spike right after a punch peak near their head.

**8. Metrics and report.** Per fighter and per round:
- punches per minute, broken down by type;
- jab share;
- average combination length;
- % of punches followed by a return to guard within X ms;
- guard-drop count and total duration;
- defensive response rate to incoming punches, broken down by type;
- head-movement-after-punching rate.

Rank the 3 top strengths and 3 top improvement areas with simple scoring against the thresholds in config. Each item carries timestamps to watch. Output `report.json` (structured) and `report.md` (readable). Keep the structure LLM-friendly, so an LLM can later turn it into coach-style prose, but do not add the LLM step now.

**9. Overlay.** Draw skeletons colored by fighter, an A/B label, lead/rear hand markers, and event labels that persist ~0.5 s. Optionally add a small timeline strip.

## Evaluation (build this early, not last)

- A tiny labeling format: `labels/<video>.json` listing `{fighter, type, t}`. Add a script to create it, even a simple CLI or CSV workflow.
- `sparring eval` matches predicted events to labels within a ±150 ms tolerance and reports precision, recall, and F1 per event type, plus identity-switch count.
- Commit a short sample clip, or a script that downloads one, plus hand labels, so thresholds get tuned against numbers instead of vibes. If you can't get footage, ask me for a clip rather than inventing data.

## Testing

- Unit-test the kinematics and each event detector on **synthetic keypoint sequences**, e.g. a generated jab trajectory, a guard drop, or a slip with the opponent jab at the same time. These tests should run without models or video.
- Run one slow, marked integration test end to end on the sample clip.
- Write the detector tests first (red/green).

## Out of scope for this PoC

- learned action classifiers (see the next step)
- 3D pose lifting
- landed-punch scoring
- a web UI
- real-time processing
- other martial arts

## Design for the next step

Keep the event interface such that a learned temporal model can replace or augment any rule detector without changing downstream code. Candidates are a skeleton-based action recognizer (ST-GCN/CTR-GCN/PoseC3D via pyskl or mmaction2) or a small 1D-CNN/transformer over keypoint windows. The rule detectors will bootstrap training labels for it. Also keep 3D lifting (e.g. MotionBERT) pluggable at stage 5, since 2D depth ambiguity hurts punch classification when a fighter faces the camera.

## Deliverables

1. The working pipeline and CLI.
2. Tests.
3. A `README.md` covering setup (including model downloads), usage, and the run-directory format.
4. An `ARCHITECTURE.md` explaining the stages and the key decisions.
5. A short "known limitations and next steps" section, based on what you actually observed on real footage.

Commit in logical steps on `move-detection-poc`.
