import pytest
from synthetic import FPS, Scene

from sparring.events.base import PUNCHES, Event
from sparring.events.defense import detect_defense
from sparring.events.detect import detect_all
from sparring.events.faults import close_gaps, runs
from sparring.events.punches import classify_opening, detect_combinations


def _of(events, fighter=None, types=None):
    return [
        e
        for e in events
        if (fighter is None or e.fighter == fighter) and (types is None or e.type in types)
    ]


@pytest.fixture
def detect(cfg, features_of):
    def run(scene: Scene):
        return detect_all(features_of(scene.build()), FPS, cfg)

    return run


def test_static_guard_produces_no_events(detect):
    assert detect(Scene(3)) == []


def test_lead_straight_is_a_jab(detect):
    ev = detect(Scene(2).straight(0, "l", 1.0))
    punches = _of(ev, types=PUNCHES)
    assert [(p.fighter, p.type) for p in punches] == [("A", "jab")]
    assert punches[0].peak_t == pytest.approx(1.0, abs=0.07)
    assert punches[0].evidence["in_range"]


def test_rear_straight_is_a_cross(detect):
    ev = detect(Scene(2).straight(0, "r", 1.0))
    assert [p.type for p in _of(ev, types=PUNCHES)] == ["cross"]


def test_one_two_is_a_combination(detect):
    ev = detect(Scene(2.5).straight(0, "l", 0.8).straight(0, "r", 1.15))
    assert [p.type for p in _of(ev, types=PUNCHES)] == ["jab", "cross"]
    combos = _of(ev, types={"combination"})
    assert len(combos) == 1 and combos[0].evidence["punches"] == ["jab", "cross"]


def test_punch_detection_for_fighter_facing_left(detect):
    ev = detect(Scene(2).straight(1, "l", 1.0))
    assert [(p.fighter, p.type) for p in _of(ev, types=PUNCHES)] == [("B", "jab")]


def test_classify_opening(cfg):
    c = cfg.punch
    assert classify_opening(165, forward=0.8, rise=0.0, cfg=c) == "straight"
    assert classify_opening(125, forward=0.1, rise=0.5, cfg=c) == "uppercut"
    assert classify_opening(125, forward=0.15, rise=0.0, cfg=c) == "hook"


def test_unanswered_jab_is_no_defense(detect):
    ev = detect(Scene(2).straight(0, "l", 1.0))
    # B's guard hands sit ~0.6 TL from its head in the synthetic pose; tighten nothing, expect
    # either a block (hands near head) or no defence, but never an evasion.
    responses = _of(ev, "B", {"slip", "duck", "block", "step_back", "no_defense"})
    assert len(responses) == 1 and responses[0].type in {"block", "no_defense"}
    assert responses[0].evidence["against"] == "jab"


def test_head_moving_off_line_during_jab_is_a_slip(detect):
    scene = Scene(2).straight(0, "l", 1.0).move_head(1, 0.85, 1.25, dx=-45, dy=10)
    responses = _of(detect(scene), "B", {"slip", "duck", "block", "step_back", "no_defense"})
    assert [r.type for r in responses] == ["slip"]


def test_head_dropping_during_jab_is_a_duck(detect):
    scene = Scene(2).straight(0, "l", 1.0).move_head(1, 0.85, 1.25, dx=0, dy=45)
    responses = _of(detect(scene), "B", {"slip", "duck", "block", "step_back", "no_defense"})
    assert [r.type for r in responses] == ["duck"]


def test_defense_ignores_out_of_range_punches(cfg, features_of):
    f = features_of(Scene(2).build())
    far = Event("A", "jab", 0.8, 1.0, 1.2, evidence={"in_range": False})
    assert detect_defense(f[f.fighter == "B"], "B", [far], cfg.defense) == []


def test_dropped_lead_hand_is_a_guard_drop(detect):
    ev = detect(Scene(3).drop_hand(0, "l", 0.5, 2.0))
    drops = _of(ev, "A", {"guard_drop"})
    assert len(drops) == 1
    assert drops[0].evidence["hand"] == "lead"
    assert drops[0].start_t == pytest.approx(0.5, abs=0.1)
    assert drops[0].end_t == pytest.approx(2.0, abs=0.1)


def test_brief_hand_dip_is_not_a_guard_drop(detect):
    assert _of(detect(Scene(2).drop_hand(0, "l", 0.5, 0.7)), types={"guard_drop"}) == []


def test_arm_left_out_after_punch_is_a_slow_return(detect):
    ev = detect(Scene(3).punch_and_stay_out(0, "l", 1.0, stay_s=0.8))
    slow = _of(ev, "A", {"slow_return"})
    assert len(slow) == 1 and slow[0].evidence["return_s"] > 0.5


def test_quick_return_is_recorded_on_the_punch(detect):
    ev = detect(Scene(2).straight(0, "l", 1.0))
    jab = _of(ev, "A", {"jab"})[0]
    assert jab.evidence["return_s"] <= 0.3
    assert _of(ev, types={"slow_return"}) == []


def test_runs_and_close_gaps():
    m = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    assert runs(m, 2) == [(1, 2), (4, 6)]
    assert runs(close_gaps(m, 1), 2) == [(1, 6)]


def test_combinations_respect_gap(cfg):
    def p(t):
        return Event("A", "jab", t - 0.1, t, t + 0.1)

    combos = detect_combinations([p(1.0), p(1.3), p(3.0)], cfg.punch)
    assert len(combos) == 1 and combos[0].evidence["count"] == 2


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError):
        Event("A", "spinning_backfist", 0, 0, 0)
