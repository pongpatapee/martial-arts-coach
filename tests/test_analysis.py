import pytest
from synthetic import FPS, Scene

from sparring.analysis.report import build_report, render_markdown
from sparring.evaluation import evaluate, match
from sparring.events.base import Event
from sparring.events.detect import detect_all
from sparring.io.video import VideoInfo


def test_match_is_one_to_one_within_tolerance():
    pred = [(1.0, 1.0), (1.05, 1.05), (3.0, 3.0)]
    assert match(pred, [1.02, 2.0], tol=0.15) == 1


def test_match_counts_label_inside_interval():
    assert match([(2.0, 5.0)], [4.0], tol=0.15) == 1
    assert match([(2.0, 5.0)], [5.5], tol=0.15) == 0


def test_evaluate_scores_exact_and_any_punch():
    events = [
        Event("A", "jab", 0.9, 1.0, 1.1),
        Event("A", "cross", 1.9, 2.0, 2.1),  # labelled as a hook: wrong type, right time
        Event("B", "jab", 2.9, 3.0, 3.1),  # false positive
    ]
    labels = {
        "segment": [0, 10],
        "events": [
            {"fighter": "A", "type": "jab", "t": 1.02},
            {"fighter": "A", "type": "lead_hook", "t": 2.0},
            {"fighter": "B", "type": "punch", "t": 5.0},
        ],
    }
    s = evaluate(events, labels, tol=0.15)["scores"]
    assert s["jab"]["tp"] == 1 and s["jab"]["pred"] == 2
    assert s["lead_hook"]["recall"] == 0.0
    assert s["punch (any)"] == {
        "tp": 2,
        "pred": 3,
        "true": 3,
        "precision": 0.667,
        "recall": 0.667,
        "f1": 0.667,
    }


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


def test_report_findings(session):
    a, b = session["fighters"]["A"], session["fighters"]["B"]
    assert a["metrics"]["offense"]["by_type"]["jab"] == 8
    assert "jab" in {x["key"] for x in a["strengths"]}
    assert b["improve"][0]["key"] == "guard_drop"
    assert "rear" in b["improve"][0]["title"]
    assert b["improve"][0]["evidence_t"]


def test_report_markdown_renders(session):
    md = render_markdown(session)
    assert "## Fighter A" in md and "## Fighter B" in md and "Carries the rear hand low" in md
