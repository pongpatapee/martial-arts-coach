"""Pipeline configuration.

Every tunable threshold lives here so detectors stay pure functions of (features, config).
Units: time in seconds, body-relative distances in torso lengths ("TL"), where one TL is the
distance from the shoulder midpoint to the hip midpoint.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestConfig(_Section):
    min_recommended_fps: float = 50.0


class DetectConfig(_Section):
    model: str = "models/yolo11m.pt"
    imgsz: int = 1280
    conf: float = 0.25
    device: str | None = None  # None = auto (cuda > mps > cpu)
    track_buffer: int = 60


class FighterConfig(_Section):
    # Relative weights of the per-frame assignment cost terms.
    w_appearance: float = 1.0
    w_motion: float = 0.6
    # Tie-breaker only: tracker IDs swap in clinches, and same-vs-other appearance margins can be
    # as small as ~0.1, so this must stay well below that.
    w_track_continuity: float = 0.08
    # A detection further than this many bbox heights from a fighter's last position is not
    # considered a continuation (unless the fighter has been missing for a while).
    max_jump_heights: float = 1.2
    # Cost above which a fighter is left unassigned for the frame.
    max_cost: float = 1.1
    # Frames after which motion continuity is ignored and re-acquisition is by appearance only.
    reacquire_after_frames: int = 15
    # Passes of assign -> rebuild appearance prototype.
    refine_passes: int = 2
    # Plausibility gates (a gap is better than the wrong person): a fighter's box may not shrink
    # below this share of its own recent height, nor of its partner's height (sparring partners
    # stand close together, so perspective cannot make one less than about half the other's size).
    min_height_vs_self: float = 0.6
    min_height_vs_partner: float = 0.55


class PoseConfig(_Section):
    model_url: str = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip"
    )
    input_size: tuple[int, int] = (288, 384)  # (w, h)
    # Optional 3D pass (RTMW3D). Off by default: see ARCHITECTURE.md for the benchmark result.
    enable_3d: bool = False
    model_url_3d: str = (
        "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/"
        "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
    )
    input_size_3d: tuple[int, int] = (288, 384)
    backend: str = "onnxruntime"
    device: str = "cpu"


class CleanConfig(_Section):
    min_keypoint_conf: float = 0.3
    max_gap_frames: int = 5
    # One Euro filter: low min_cutoff = more smoothing at rest, beta = less lag when fast.
    # Speeds are measured in body heights / s; a punch (~2 body heights / s) raises the cutoff to
    # ~9 Hz so the velocity peak survives, while a still joint gets a 1 Hz cutoff.
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 4.0
    one_euro_d_cutoff: float = 1.0


class KinematicsConfig(_Section):
    torso_window_s: float = 2.0
    stance_window_s: float = 1.5
    # Fighters closer than this (hip-to-hip, in TL) are "in range" to hit each other.
    in_range_tl: float = 2.6
    # Horizontal separation (TL) below which the camera is looking along the line of attack and
    # the forward axis is unreliable.
    min_axis_separation_tl: float = 0.8
    # A wrist within this distance (TL) of the opponent's neck/face counts as a clinch grip.
    clinch_hand_tl: float = 0.6


class PunchConfig(_Section):
    # Route 1 (straights, uppercuts): the elbow snaps open. The 2D elbow angle survives
    # foreshortening far better than wrist speed or reach when a punch is thrown partly toward
    # the camera.
    min_elbow_opening_deg: float = 40.0
    min_peak_elbow_deg: float = 120.0
    min_speed_tl: float = 1.2  # the wrist must move at least this fast (TL / s) during the punch
    min_forward_tl: float = 0.1  # the wrist must travel toward the opponent (or up)
    # Route 2 (hooks): a fast wrist with the elbow kept bent and raised near shoulder height.
    hook_min_speed_tl: float = 5.0
    hook_min_elbow_deg: float = 60.0
    hook_max_elbow_deg: float = 130.0
    hook_min_elbow_height_tl: float = -0.35  # elbow height relative to the shoulder
    hook_min_travel_tl: float = 0.3
    # Route 3 (straight-arm "flicker" jabs from a low, already-extended lead hand): the elbow barely
    # changes, but the wrist shoots forward. Candidate = a peak in forward reach.
    reach_min_gain_tl: float = 0.45
    reach_min_speed_tl: float = 3.5
    # Segmentation.
    refractory_s: float = 0.2  # min time between two punches with the same hand
    max_windup_s: float = 0.3
    max_duration_s: float = 0.6
    # Classification.
    straight_min_elbow_deg: float = 140.0
    straight_min_forward_tl: float = (
        0.3  # a less-open elbow still counts if the reach is clearly forward
    )
    uppercut_min_rise_ratio: float = 1.0  # upward travel / forward travel
    uppercut_min_rise_tl: float = 0.2
    combo_gap_s: float = 0.45


class KickConfig(_Section):
    min_rise_tl: float = 0.5  # the ankle rises at least this far above its chamber height
    # Footwork lifts the feet too; a kick drives the foot toward the opponent, fast. (On the
    # boxing sample, steps rose 0.5-0.75 TL but moved < 0.15 TL forward or < 6 TL/s.)
    min_speed_tl: float = 6.0
    min_forward_tl: float = 0.3
    refractory_s: float = 0.4
    max_windup_s: float = 0.5
    max_duration_s: float = 0.9
    # Classification (ankle peak height relative to the hip; + is up).
    head_kick_min_y_tl: float = 0.9  # about shoulder height
    body_kick_min_y_tl: float = -0.4  # around the hip / lower ribs
    teep_min_forward_tl: float = 0.7
    teep_forward_ratio: float = 1.0  # forward travel at least this multiple of the rise
    teep_min_knee_opening_deg: float = 40.0
    # Kicks land with the leg (nearly) straight; a folded leg at the peak is a knee strike.
    min_knee_deg_at_impact: float = 110.0


class CloseRangeConfig(_Section):
    max_distance_tl: float = 1.6
    knee_min_rise_tl: float = 0.4
    knee_min_height_tl: float = -0.3  # knee at least this high relative to the hip
    knee_max_fold_deg: float = 100.0
    knee_min_forward_speed_tl: float = 1.5
    teep_extension_deg: float = 140.0
    elbow_min_speed_tl: float = 5.0
    elbow_max_fold_deg: float = 70.0
    elbow_min_height_tl: float = -0.3
    # An elbow strike leads with the elbow: it is ahead of the wrist toward the opponent. (A
    # close hook also has a folded arm, but the fist leads.)
    elbow_min_lead_tl: float = 0.1
    clinch_max_distance_tl: float = 1.0
    clinch_min_s: float = 1.0


class DetectionConfig(_Section):
    # Strike detector for punches: "rules", "ensemble" (rules + learned second opinion),
    # "learned", "hybrid" or "temporal". Learned methods need a model file (`sparring train`);
    # without one the pipeline falls back to "rules" and says so.
    method: str = "ensemble"
    model_path: str | None = None  # default: the bundled model for the discipline, if any


class DefenseConfig(_Section):
    window_before_s: float = 0.15
    window_after_s: float = 0.2
    # ~20 cm of head travel; ordinary in-range head motion is 0.17-0.24 TL (median).
    slip_min_head_disp_tl: float = 0.4
    duck_min_drop_tl: float = 0.3
    block_max_wrist_head_tl: float = 0.55
    step_back_min_tl: float = 0.5
    check_min_knee_rise_tl: float = 0.4  # lead knee lift that counts as checking a kick


class FaultConfig(_Section):
    # Wrist more than this far (TL) below its shoulder counts as a dropped hand. The wrist
    # keypoint sits at the bottom of the glove, so a glove held at the chin is already ~0.1 TL
    # below shoulder height.
    guard_drop_below_shoulder_tl: float = 0.4
    # A hand is "home" after a punch when the elbow is bent back and the wrist is up.
    guard_home_max_elbow_deg: float = 110.0
    guard_drop_min_s: float = 0.5
    # Drops separated by less than this are one drop (brief keypoint noise splits them).
    merge_gap_s: float = 0.4
    return_to_guard_max_s: float = 0.5
    chin_up_nose_above_ears_tl: float = 0.12
    chin_up_min_s: float = 0.5
    square_stance_max_width_tl: float = 0.35
    square_stance_min_s: float = 1.0
    over_lean_head_past_foot_tl: float = 0.5


class ReportConfig(_Section):
    max_evidence: int = 5
    evidence_min_spacing_s: float = 1.0
    # Metric targets used to rank strengths/improvement areas.
    good_return_to_guard_rate: float = 0.8
    bad_return_to_guard_rate: float = 0.6
    good_defense_rate: float = 0.5
    bad_defense_rate: float = 0.25
    bad_guard_drop_per_min: float = 4.0
    bad_guard_low_share: float = 0.25  # share of in-range time a hand is held low
    good_punches_per_min: float = 30.0
    bad_punches_per_min: float = 12.0
    good_jab_share: float = 0.35
    bad_jab_share: float = 0.15


class Config(_Section):
    # Which martial art the footage is: selects the move set, detectors and coaching rules.
    discipline: str = "boxing"
    ingest: IngestConfig = IngestConfig()
    detect: DetectConfig = DetectConfig()
    fighters: FighterConfig = FighterConfig()
    pose: PoseConfig = PoseConfig()
    clean: CleanConfig = CleanConfig()
    kinematics: KinematicsConfig = KinematicsConfig()
    detection: DetectionConfig = DetectionConfig()
    punch: PunchConfig = PunchConfig()
    kick: KickConfig = KickConfig()
    close_range: CloseRangeConfig = CloseRangeConfig()
    defense: DefenseConfig = DefenseConfig()
    faults: FaultConfig = FaultConfig()
    report: ReportConfig = ReportConfig()

    @classmethod
    def load(cls, path: Path | None) -> Config:
        if path is None:
            return cls()
        with path.open("rb") as f:
            return cls.model_validate(tomllib.load(f))
