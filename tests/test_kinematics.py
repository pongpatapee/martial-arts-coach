import numpy as np
from synthetic import FPS, Scene

from sparring.config import CleanConfig
from sparring.kinematics.clean import clean_poses, fill_short_gaps, one_euro


def test_fill_short_gaps_only_fills_bounded_short_runs():
    x = np.array([0, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, 8, np.nan])
    out = fill_short_gaps(x, max_gap=2)
    np.testing.assert_allclose(out[:4], [0, 1, 2, 3])
    assert np.isnan(out[4:8]).all()  # gap of 4 > max_gap
    assert np.isnan(out[9])  # trailing gap has no right neighbour


def test_one_euro_smooths_jitter_but_tracks_fast_motion():
    rng = np.random.default_rng(0)
    still = 1.0 + 0.01 * rng.standard_normal(90)
    out = one_euro(still, FPS, min_cutoff=1.0, beta=4.0, d_cutoff=1.0)
    assert out[30:].std() < 0.5 * still[30:].std()

    step = np.r_[np.zeros(30), np.ones(30)]
    out = one_euro(step, FPS, min_cutoff=1.0, beta=4.0, d_cutoff=1.0)
    assert out[33] > 0.8  # reaches the new position within ~100 ms


def test_one_euro_resets_on_nan():
    x = np.r_[np.zeros(5), np.nan, np.full(5, 10.0)]
    out = one_euro(x, FPS, 1.0, 4.0, 1.0)
    assert np.isnan(out[5]) and out[6] == 10.0


def test_clean_masks_low_confidence():
    kp = Scene(1).build()
    scores = np.full(kp.shape[:3], 0.9, np.float32)
    scores[10:20, 0, 9] = 0.1  # A's left wrist unreliable for 10 frames (> max gap)
    out = clean_poses(kp, scores, FPS, CleanConfig())
    assert np.isnan(out[10:20, 0, 9]).all()
    assert np.isfinite(out[:, 1]).all()


def test_body_frame_orientation_stance_and_range(features_of):
    f = features_of(Scene(2).build())
    a, b = f[f.fighter == "A"], f[f.fighter == "B"]
    assert (a.facing == 1).all() and (b.facing == -1).all()
    # Both orthodox: mirroring the image keeps B's left foot as the one nearer the opponent.
    assert (a.lead_side == 1).all() and (b.lead_side == 1).all()
    np.testing.assert_allclose(a.distance_tl, 2.2, atol=0.05)
    assert a.in_range.all() and a.axis_ok.all()
    # Guard is up: both wrists above shoulder height, elbows bent.
    assert (a.lead_wrist_below_shoulder < 0).all() and (a.rear_wrist_below_shoulder < 0).all()
    assert (a.lead_elbow_deg < 110).all()


def test_features_are_mirror_symmetric(features_of):
    f = features_of(Scene(1).straight(0, "l", 0.5).straight(1, "l", 0.5).build())
    a, b = f[f.fighter == "A"].reset_index(), f[f.fighter == "B"].reset_index()
    for col in ("lead_wrist_x", "lead_wrist_y", "lead_elbow_deg", "head_x", "distance_tl"):
        np.testing.assert_allclose(a[col], b[col], atol=1e-3)
