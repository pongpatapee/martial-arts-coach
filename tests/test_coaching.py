from types import SimpleNamespace

from sparring.coaching.engine import coach
from sparring.coaching.knowledge import RELIABILITY, RULES
from sparring.coaching.llm import MODEL, coach_note
from sparring.events.base import Event


def metrics(**over) -> dict:
    """A neutral metrics dict (nothing fires) with overrides like offense__jab_share=0.05."""
    m = {
        "visible_s": 60.0,
        "in_range_s": 50.0,
        "offense": {
            "strikes": 30,
            "strikes_per_min": 25.0,
            "punches": 30,
            "kicks": 0,
            "kick_share": 0.0,
            "jab_share": 0.25,
            "rear_hand_share": 0.4,
            "share_in_combinations": 0.3,
            "counter_rate": 0.2,
            "return_to_guard_rate": 0.75,
            "head_move_after_striking": 0.4,
            "over_leans": 0,
            "kicks_hands_down": 0,
        },
        "defense": {
            "incoming_in_range": 20,
            "defense_rate": 0.5,
            "head_movement_rate": 0.25,
            "block_rate": 0.25,
            "kicks_incoming": 0,
            "check_rate": None,
            "kick_defense_rate": None,
        },
        "guard": {
            "low_share_by_hand": {"lead": 0.1, "rear": 0.1},
            "chin_up_share": 0.0,
            "square_stance_share": 0.0,
        },
    }
    for k, v in over.items():
        section, key = k.split("__")
        m[section][key] = v
    return m


def test_neutral_metrics_fire_nothing():
    c = coach("A", metrics(), [], "boxing")
    assert c["strengths"] == [] and c["priorities"] == []
    assert "Not enough" in c["summary"]


def test_priorities_ranked_with_drills_and_plan():
    c = coach(
        "A", metrics(offense__jab_share=0.05, offense__return_to_guard_rate=0.3), [], "boxing"
    )
    ids = [p["id"] for p in c["priorities"]]
    assert set(ids) == {"rarely_jabs", "slow_return"}
    assert all(p["drills"] and p["why"] and p["fix"] for p in c["priorities"])
    assert c["session_plan"] and c["session_plan"][0]["for"] == c["priorities"][0]["title"]
    assert c["summary"].startswith("Main focus")


def test_small_samples_are_not_judged():
    m = metrics(offense__jab_share=0.0, offense__punches=3)
    assert "rarely_jabs" not in [p["id"] for p in coach("A", m, [], "boxing")["priorities"]]


def test_watch_times_come_from_matching_events():
    events = [
        Event("A", "guard_drop", 1.0, 2.0, 3.0, 0.8, {"hand": "lead"}),
        Event("A", "guard_drop", 9.0, 10.0, 11.0, 0.8, {"hand": "rear"}),  # other hand
        Event("B", "guard_drop", 1.0, 5.0, 9.0, 0.8, {"hand": "lead"}),  # other fighter
    ]
    m = metrics(guard__low_share_by_hand={"lead": 0.6, "rear": 0.1})
    low = next(p for p in coach("A", m, events, "boxing")["priorities"] if p["id"] == "low_hand")
    assert low["title"] == "Carries the lead hand low" and low["watch"] == [2.0]


def test_kicking_rules_only_in_kicking_disciplines():
    m = metrics(offense__kick_share=0.0, offense__strikes=20)
    assert "hands_only" not in [p["id"] for p in coach("A", m, [], "boxing")["priorities"]]
    assert "hands_only" in [p["id"] for p in coach("A", m, [], "kickboxing")["priorities"]]


def test_every_rule_is_well_formed():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))
    for r in RULES:
        assert r.kind in ("strength", "improve") and r.basis in RELIABILITY
        assert r.observation and r.why and r.fix


class _FakeClient:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text="## Fighter A\\nKeep the lead up.")],
        )


def test_llm_note_is_grounded_in_the_report():
    client = _FakeClient()
    note = coach_note({"fighters": {"A": {"coaching": {"summary": "x"}}}}, client=client)
    assert note.startswith("## Fighter A")
    call = client.calls[0]
    assert call["model"] == MODEL and "Use only facts present" in call["system"]
    assert '"summary": "x"' in call["messages"][0]["content"]


def test_llm_refusal_raises():
    import pytest

    with pytest.raises(RuntimeError):
        coach_note({"fighters": {}}, client=_FakeClient("refusal"))
