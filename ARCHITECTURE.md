# Architecture

The pipeline turns a sparring video into three things:
- a pose track for exactly two fighters,
- a timeline of events (strikes, defences, faults),
- a coaching report.

One engine serves every discipline. A discipline only chooses which detectors run and which
coaching rules apply.

```
video ─► ingest ─► track ─► fighters ─► pose ─► [pose3d] ─► kinematics ─► events ─► report ─► render
                                                               │             │          │
                                                    features.parquet   events.json   report.{json,md}
```

Every stage reads only earlier stages' artifacts from the run directory (`pipeline.py`), so any
suffix can be re-run with `--from-stage`. Inference (track, pose) takes minutes; everything after
it takes seconds.

## Package map

| Area | Modules | Role |
|---|---|---|
| Perception | `perception/{detector,appearance,fighters,pose}.py` | YOLO11 + BoT-SORT; colour signatures; the two fighters' identities; RTMPose-x (Halpe-26); optional RTMW3D |
| Kinematics | `kinematics/{clean,normalize,features,features3d}.py` | Gap fill + One Euro smoothing; body-relative frames; per-frame features for hands, legs, head, distance, clinch grips |
| Taxonomy | `moves.py` | Every move across disciplines: kind (strike/defence/fault/other), limb, lead/rear side, family |
| Disciplines | `disciplines.py` | boxing / kickboxing / muay_thai: move set + detector list |
| Detection | `detection/detectors.py` | Detector plugins, run in two stages (strikes, then relational); limb-conflict resolution |
| Strike methods | `detection/{methods,candidates,windows,temporal,registry,bench}.py` | Interchangeable punch detectors (rules / learned / hybrid / ensemble / temporal), training, the CV benchmark |
| Rule detectors | `events/{punches,kicks,close_range,defense,faults}.py` | Geometry-based detectors; they emit `Event`s |
| Dataset | `dataset.py` | Clips (video + `clip.toml` + labels) under `datasets/clips/`, hash checks, run stamps |
| Evaluation | `evaluation.py` | Labels with confidence and ignore intervals; detection F1 and type accuracy |
| Coaching | `analysis/metrics.py`, `coaching/{knowledge,engine,llm}.py` | Metrics → rule knowledge base → prioritised feedback, drills, session plan; optional Claude rewrite |
| Tools | `labeling/`, `render/overlay.py`, `scripts/contact_sheet.py` | Browser labeller, annotated video, labelling contact sheets |

## Extending to another discipline

1. Add any new moves to `moves.py`, e.g. spinning back kick or sweep.
2. If a move needs a new detector, write it as a function over the feature table that returns
   `Event`s. Wrap it in a `Detector` in `detection/detectors.py`, with stage 1 for strikes or
   stage 2 for anything that depends on strikes.
3. Declare the discipline in `disciplines.py` (its move set and detector list).
4. Tag or add coaching rules in `coaching/knowledge.py` with `disciplines=...`.

Perception, kinematics, the event model, evaluation, metrics, report and render need no changes.
Metrics count strikes per limb, so a boxing session simply has no kicks.

## Key decisions

**Tracker IDs are not identities.** BoT-SORT gives fighter A about five IDs on the sample clip,
because IDs switch in clinches, at crossings and during camera pans. `fighters.py` handles this
in four steps:
1. It seeds the fighters from the track pair that is most often large and close together.
2. It builds HSV-histogram prototypes of the head, torso and leg bands.
3. It solves a gated Hungarian assignment every frame. The cost is appearance + motion + a
   tracker-ID tie-breaker, weighted 0.08, which is below the measured ~0.1 same-vs-other
   appearance margin.
4. It re-fits the prototypes from confident frames.

**Plausibility gates:** when a fighter is hidden and reacquired, the new box must be at least 60%
of the fighter's recent height and at least 55% of the partner's. Before these gates, a spectator
in a matching black shirt was picked as fighter B for 2 seconds. Leaving a fighter unassigned is
better than assigning the wrong person.

**Body-relative coordinates.** Features live in a per-fighter frame: origin at the hip midpoint,
+x toward the opponent, +y up, units of torso length (TL). This cancels camera pan, distance and
screen side. Hands and legs are named lead/rear from stance, so "jab" and "lead teep" mean the
same thing in either stance.

**Smoothing that keeps strikes.** The One Euro filter uses `beta = 4` (per body-height/s). A
still joint gets about 1 Hz of smoothing and a punch gets about 9 Hz. With `beta = 0.3`, a jab's
speed peak was flattened from 8.7 to 5.4 TL/s.

**Punches: three geometric routes, because 2D projection hides punches in different ways.**
1. **Elbow opening (straights, uppercuts).** A punch thrown toward the camera has 2D wrist speed
   of only 2–3 TL/s, but the elbow angle still snaps from about 60° to over 150°.
2. **Fast bent-arm swing (hooks).**
3. **Forward reach (straight-arm "flicker" jabs from a low lead hand).** The elbow barely moves
   (150°→160°), but the wrist shoots 0.3 → 1.1 TL forward. This route raised recall from 0.59 to
   0.68.

**Punches: the ensemble.** Rules are the backbone, and a learned classifier gives a second
opinion (details under "Detection methods tried").

**Kicks, knees and elbows** follow the same pattern, driven by ankle height, knee drive and elbow
speed. Each has a gate learned from false positives on the boxing clip:
- **Kicks** must drive the foot at least 0.3 TL toward the opponent at 6 TL/s or more. Footwork
  steps lift the feet 0.5–0.75 TL but move forward less than 0.15 TL.
- **Kicks land with the leg nearly straight.** A folded leg at the peak is a knee strike.
- **Elbow strikes lead with the elbow.** A close hook also folds the arm, but the fist leads.
- **A teep extends while the knee is still up.** Straightening on the way down is not a teep.

**Defence is relational.** Each response is classified against a specific opponent strike thrown
in range:
- **vs punches:** duck > slip > block > step back > none. The slip threshold is 0.4 TL (about
  20 cm), calibrated against the no-strike head motion baseline of 0.17–0.24 TL median.
- **vs leg kicks:** check (lead knee lifted) > step back > block > none.

**Counters** are strikes within 0.8 s of the opponent's strike when the fighter wasn't already
throwing in the 0.5 s before. The first version, without that condition, reported a 71% "counter
rate" in a fast exchange.

## Detection methods tried (strikes)

All numbers are from 5-fold time-blocked cross-validation on the labelled sample
(`docs/benchmarks/boxing.json`, `sparring bench`).

| Method | How | F1 | Notes |
|---|---|---|---|
| rules | three geometric routes above | 0.68 ± 0.11 | no training data needed |
| rules_tuned | grid search of rule thresholds on the training folds | 0.65 ± 0.10 | tuning overfits with about 60 training strikes per fold |
| learned | 95%-recall candidates (elbow/speed peaks) + gradient-boosted trees over time-window features incl. the opponent's reaction; event-level threshold chosen out-of-fold | 0.65 ± 0.08 | highest recall (0.75), low precision |
| hybrid | learned + rule outputs as features | 0.67 ± 0.08 | type accuracy suffers |
| **ensemble** | rules, a rule detection is vetoed if the model is confident it is not a strike, a missed candidate is added if the model is confident it is one; thresholds out-of-fold | **0.70 ± 0.09** | best precision (0.75) and type accuracy (0.75); default |
| temporal | dilated 1D-CNN over the feature sequence → per-frame strike heatmap | 0.67 ± 0.14 | needs more data to shine |
| + 3D (RTMW3D) | 3D elbow angle / reach as extra features | 0.64 | worse: RTMW3D's depth made a 167° (extended) 2D elbow read 33° |

**Conclusion:** with one clip, methods differ by less than the fold-to-fold spread. The pieces
that moved accuracy were:
- **Fixing perception:** the spectator identity bug.
- **Better geometry:** the reach route.
- **Honest evaluation:** don't-care labels and not-assessable intervals.

Model choice mattered less. The infrastructure (candidates, featurizer, CV benchmark, `train`,
labelling tool) is in place, so the learned methods can be re-compared as labelled footage grows.

## Evaluation

Footage is a dataset of clips, one folder each: `datasets/clips/<id>/` holds `video.*`
(git-ignored), `clip.toml` (discipline, split, camera, source, and the video's SHA-256) and
`labels.json`. Labels are tied to exact frames, so every command that scores or trains checks
the video against the stored hash, and each run in `runs/<id>/` is stamped with the hash it was
made from. A re-encoded video or a stale run is an error, not a silently wrong score. Clips in
the `test` split are never used by `sparring train` by default.

The labels record fighter, type, time and confidence.
- **Generic types:** `punch`, `kick` and `strike` count toward detection only.
- **Don't care:** low-confidence labels neither reward nor penalise a prediction.
- **Ignore intervals** mark stretches where a fighter's actions can't be seen, e.g. the camera is
  behind them.
- **Matching:** one-to-one within ±150 ms. A sustained fault matches when the label falls inside
  its interval.
- **Scores:** `strike (any)` measures detection, and `strike type` measures classification
  accuracy on matched strikes.

## Coaching

`analysis/metrics.py` computes rates with explicit denominators (None, not 0, when there is no
data). `coaching/knowledge.py` holds about 25 rules. Each rule has:
- a sample-size gate,
- a score used for ranking,
- an observation quoting the fighter's numbers, why it matters, and the fix,
- drills with a dose,
- evidence event types, from which the watch timestamps are taken,
- the disciplines it applies to,
- a `basis` that maps to a stated reliability: strikes (measured), guard (direct measurement,
  unlabelled), defence (unvalidated), kicks (synthetic only).

`coaching/engine.py` ranks the top three strengths and top three priorities and builds a session
plan from the priorities' drills. `coaching/llm.py` optionally rewrites the report in a coach's
voice with Claude (`claude-opus-5-5`, server-side refusal fallback enabled). It is instructed to
use only facts in the JSON.

## Configuration

All thresholds live in `config.py` (pydantic), each with its unit and rationale. Override any
section with a TOML file via `--config`. The resolved config is written to each run directory.
