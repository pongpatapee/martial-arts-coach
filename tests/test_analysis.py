import pytest
from synthetic import FPS, Scene

from sparring.analysis.report import build_report, render_markdown
from sparring.evaluation import ANY_STRIKE, STRIKE_TYPE, Labels, evaluate, match
from sparring.events.base import Event
from sparring.events.detect import detect_all
from sparring.io.video import VideoInfo


def test_match_is_one_to_one_within_tolerance():
    pred = [(1.0, 1.0), (1.05, 1.05), (3.0, 3.0)]
    assert match(pred, [1.02, 2.0], tol=0.15) == [(0, 0)]


def test_match_counts_label_inside_interval():
    assert len(match([(2.0, 5.0)], [4.0], tol=0.15)) == 1
    assert match([(2.0, 5.0)], [5.5], tol=0.15) == []


def _labels(events, **kw) -> Labels:
    return Labels.from_dict({"segment": [0, 10], "events": events, **kw})


def test_evaluate_detection_and_type_accuracy():
    events = [
        Event("A", "jab", 0.9, 1.0, 1.1),
        Event("A", "cross", 1.9, 2.0, 2.1),  # labelled as a hook: right time, wrong type
        Event("B", "jab", 2.9, 3.0, 3.1),  # false positive
    ]
    labels = _labels(
        [
            {"fighter": "A", "type": "jab", "t": 1.02},
            {"fighter": "A", "type": "lead_hook", "t": 2.0},
            {"fighter": "B", "type": "punch", "t": 5.0},  # generic: detection only
        ]
    )
    s = evaluate(events, labels, tol=0.15)["scores"]
    assert s[ANY_STRIKE] == {
        "tp": 2,
        "pred": 3,
        "true": 3,
        "precision": 0.667,
        "recall": 0.667,
        "f1": 0.667,
    }
    assert s[STRIKE_TYPE]["accuracy"] == 0.5
    assert s[STRIKE_TYPE]["confusions"] == ["lead_hook->cross"]


def test_low_confidence_labels_are_dont_care():
    events = [Event("A", "jab", 0.9, 1.0, 1.1)]
    labels = _labels(
        [
            {"fighter": "A", "type": "punch", "t": 1.0, "confidence": "low"},
            {"fighter": "A", "type": "punch", "t": 4.0, "confidence": "low"},
        ]
    )
    s = evaluate(events, labels)["scores"][ANY_STRIKE]
    assert (s["tp"], s["pred"], s["true"]) == (0, 0, 0)


def test_ignore_intervals_drop_labels_and_predictions():
    events = [Event("B", "jab", 1.9, 2.0, 2.1), Event("A", "jab", 1.9, 2.0, 2.1)]
    labels = _labels(
        [{"fighter": "B", "type": "punch", "t": 5.0}],
        ignore=[{"fighter": "B", "from": 0.0, "to": 6.0}],
    )
    s = evaluate(events, labels)["scores"][ANY_STRIKE]
    assert (s["tp"], s["pred"], s["true"]) == (0, 1, 0)  # only A's (false) jab counts


@pytest.fixture
def session(cfg, features_of):
    scene = Scene(20)
    for t in (1, 3, 5, 7, 9, 11, 13, 15):
        scene.straight(0, "l", t)
    scene.drop_hand(1, "r", 2, 12)
    f = features_of(scene.build())
    events = detect_all(f, FPS, cfg)
    info = VideoInfo("x.mp4", FPS, len(f) // 2, 1080, 1920)
    meta = {
        "colours": {"A": {}, "B": {}},
        "fighters": {"A": {"coverage": 1.0}, "B": {"coverage": 1.0}},
    }
    return build_report(events, f, info, meta, cfg)


def test_report_coaching(session):
    a, b = session["fighters"]["A"], session["fighters"]["B"]
    assert a["metrics"]["offense"]["by_type"]["jab"] == 8
    assert "jab_leads" in {x["id"] for x in a["coaching"]["strengths"]}
    top = b["coaching"]["priorities"][0]
    assert top["id"] == "low_hand" and "rear" in top["title"]
    assert top["watch"] and top["drills"]
    assert b["coaching"]["session_plan"][0]["for"] == top["title"]


def test_report_markdown_renders(session):
    md = render_markdown(session)
    assert "## Fighter A" in md and "## Fighter B" in md and "Carries the rear hand low" in md


def test_events_json_is_strict_json(tmp_path):
    import json

    from sparring.events.base import load_events, save_events

    path = tmp_path / "events.json"
    save_events([Event("A", "slip", 0.9, 1.0, 1.1, evidence={"retreat_tl": float("nan")})], path)
    assert (
        json.loads(path.read_text(), parse_constant=lambda c: 1 / 0)[0]["evidence"]["retreat_tl"]
        is None
    )
    assert load_events(path)[0].type == "slip"
