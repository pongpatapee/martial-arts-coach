"""Kickboxing / muay thai detectors on synthetic skeletons, and the discipline registry."""

import pytest
from synthetic import FPS, Scene

from sparring.config import Config
from sparring.disciplines import DISCIPLINES, get_discipline
from sparring.events.detect import detect_all
from sparring.moves import KICKS, MOVES, PUNCHES


def _detect(scene: Scene, discipline: str, features_of):
    cfg = Config(discipline=discipline).model_copy(
        update={"detection": Config().detection.model_copy(update={"method": "rules"})}
    )
    return detect_all(features_of(scene.build()), FPS, cfg)


def _strikes(events, fighter="A"):
    return [e.type for e in events if e.fighter == fighter and e.kind == "strike"]


# Rear (right) leg positions for A (orthodox, facing right; hip at y=600, 1 TL = 100 px).
BODY_KICK = {"r_knee": (500, 600), "r_ankle": (590, 585)}
HEAD_KICK = {"r_knee": (480, 520), "r_ankle": (560, 460)}
LOW_KICK = {"r_knee": (470, 660), "r_ankle": (560, 690)}


def _kick(scene: Scene, pose: dict, t: float = 1.0) -> Scene:
    return scene.animate(0, [(t - 0.2, {}), (t, pose), (t + 0.08, pose), (t + 0.35, {})])


@pytest.mark.parametrize(
    ("pose", "expected"),
    [(BODY_KICK, "rear_body_kick"), (HEAD_KICK, "rear_head_kick"), (LOW_KICK, "rear_low_kick")],
)
def test_round_kicks_classified_by_height(pose, expected, features_of):
    ev = _detect(_kick(Scene(2), pose), "kickboxing", features_of)
    assert _strikes(ev) == [expected]


def test_teep_is_chamber_then_forward_push(features_of):
    chamber = {"l_knee": (480, 590), "l_ankle": (470, 670)}
    push = {"l_knee": (530, 610), "l_ankle": (640, 620)}
    scene = Scene(2).animate(0, [(0.7, {}), (0.85, chamber), (1.0, push), (1.08, push), (1.35, {})])
    assert _strikes(_detect(scene, "kickboxing", features_of)) == ["lead_teep"]


def test_boxing_ignores_kicks(features_of):
    ev = _detect(_kick(Scene(2), BODY_KICK), "boxing", features_of)
    assert not any(e.type in KICKS for e in ev)


def test_knee_strike_at_close_range(features_of):
    knee = {"r_knee": (480, 575), "r_ankle": (405, 690)}
    scene = Scene(2, gap_tl=1.2).animate(0, [(0.8, {}), (1.0, knee), (1.1, knee), (1.35, {})])
    ev = _detect(scene, "muay_thai", features_of)
    assert _strikes(ev) == ["rear_knee"]


def test_knee_needs_close_range(features_of):
    knee = {"r_knee": (480, 575), "r_ankle": (405, 690)}
    scene = Scene(2, gap_tl=2.5).animate(0, [(0.8, {}), (1.0, knee), (1.1, knee), (1.35, {})])
    assert "rear_knee" not in _strikes(_detect(scene, "muay_thai", features_of))


def test_elbow_strike_at_close_range(features_of):
    elbow = {"r_elbow": (490, 505), "r_wrist": (420, 470)}
    scene = Scene(2, gap_tl=1.1).animate(0, [(0.85, {}), (1.0, elbow), (1.05, elbow), (1.3, {})])
    assert _strikes(_detect(scene, "muay_thai", features_of)) == ["rear_elbow"]


def test_clinch_when_hands_on_opponents_neck(features_of):
    grip = {
        "l_wrist": (482, 470),
        "r_wrist": (476, 480),
        "l_elbow": (450, 530),
        "r_elbow": (440, 535),
    }
    scene = Scene(3, gap_tl=0.9).animate(0, [(0.3, {}), (0.5, grip), (2.3, grip), (2.5, {})])
    ev = _detect(scene, "muay_thai", features_of)
    clinch = [e for e in ev if e.type == "clinch" and e.fighter == "A"]
    assert len(clinch) == 1 and clinch[0].duration >= 1.5


def test_lifting_the_lead_knee_checks_a_low_kick(features_of):
    scene = _kick(Scene(2), LOW_KICK)  # A low-kicks B
    # B lifts its lead knee to meet it (B is A mirrored, so animate B with A's coordinates).
    check = {"l_knee": (445, 600), "l_ankle": (455, 690)}
    scene.animate(1, [(0.75, {}), (0.95, check), (1.1, check), (1.35, {})])
    ev = _detect(scene, "kickboxing", features_of)
    assert [e.type for e in ev if e.fighter == "B" and e.kind == "defense"] == ["check"]


def test_kicking_with_the_hands_down_is_a_fault(features_of):
    scene = _kick(Scene(2), BODY_KICK).drop_hand(0, "l", 0.75, 1.4)
    ev = _detect(scene, "kickboxing", features_of)
    assert any(e.type == "kick_no_guard" and e.fighter == "A" for e in ev)


def test_discipline_move_sets():
    assert set(PUNCHES) <= set(DISCIPLINES["boxing"].moves)
    assert not set(KICKS) & set(DISCIPLINES["boxing"].moves)
    assert set(KICKS) <= set(DISCIPLINES["kickboxing"].moves)
    assert {"lead_knee", "rear_elbow", "clinch"} <= set(DISCIPLINES["muay_thai"].moves)
    assert all(m in MOVES for d in DISCIPLINES.values() for m in d.moves)
    with pytest.raises(ValueError):
        get_discipline("jousting")
