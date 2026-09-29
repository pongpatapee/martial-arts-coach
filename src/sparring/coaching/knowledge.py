"""Coaching knowledge base: metric patterns -> feedback a coach would give.

Each rule reads a fighter's metrics (see `analysis/metrics.py`) and, when it fires, returns a
score (how strongly it applies; used for ranking) and the facts to quote. A rule carries:

- `observation`: what the video shows, with the fighter's own numbers,
- `why`: why it matters in a fight,
- `fix`: the technical cue to correct it (or, for strengths, how to build on it),
- `drills`: concrete drills with a dose,
- `evidence`: which event types to pull timestamps from, so the fighter can watch it happen.

Rules are tagged with the disciplines they apply to. Thresholds are deliberately conservative
and every rule needs a minimum sample size, so a short clip produces fewer, safer comments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

BOXING, KICKBOXING, MUAY_THAI = "boxing", "kickboxing", "muay_thai"
ALL = frozenset({BOXING, KICKBOXING, MUAY_THAI})
KICKING = frozenset({KICKBOXING, MUAY_THAI})


@dataclass(frozen=True)
class Drill:
    name: str
    how: str
    dose: str


@dataclass(frozen=True)
class Rule:
    id: str
    kind: str  # "strength" | "improve"
    title: str
    check: Callable[[dict], tuple[float, dict] | None]  # metrics -> (score, facts) or None
    observation: str  # str.format(**facts)
    why: str
    fix: str
    drills: tuple[Drill, ...] = ()
    evidence: tuple[str, ...] = ()
    evidence_filter: dict = field(default_factory=dict)  # e.g. {"hand": "lead"}; may use {facts}
    disciplines: frozenset[str] = ALL
    basis: str = "strikes"  # which detector's output the rule rests on (see RELIABILITY)


# How far to trust each kind of evidence, from the benchmark on the labelled sample clip.
# Update these when better labels change the numbers.
RELIABILITY = {
    "strikes": "moderate: strike detection F1 ~0.70 (cross-validated on the labelled sample)",
    "guard": "moderate: hand height is measured directly from pose, but not checked against labels",
    "defense": "low: defensive reactions are not yet validated against labels",
    "kicks": "low: kick/knee/elbow detectors are validated only on synthetic motion",
}


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _g(m: dict, path: str):
    cur = m
    for k in path.split("."):
        if cur is None:
            return None
        cur = cur.get(k)
    return cur


# -- helpers for rule conditions ---------------------------------------------------------------
def at_least(path: str, threshold: float, n_path: str, n_min: int, fmt=pct) -> Callable:
    def check(m: dict):
        v, n = _g(m, path), _g(m, n_path) or 0
        if v is None or n < n_min or v < threshold:
            return None
        return (v - threshold) / max(threshold, 1e-6) + 0.2, {"value": fmt(v), "n": n}

    return check


def at_most(path: str, threshold: float, n_path: str, n_min: int, fmt=pct) -> Callable:
    def check(m: dict):
        v, n = _g(m, path), _g(m, n_path) or 0
        if v is None or n < n_min or v > threshold:
            return None
        return (threshold - v) / max(threshold, 1e-6) + 0.3, {"value": fmt(v), "n": n}

    return check


def _low_hand(m: dict):
    low = {h: v for h, v in (_g(m, "guard.low_share_by_hand") or {}).items() if v is not None}
    if not low:
        return None
    hand = max(low, key=low.get)
    if low[hand] < 0.25:
        return None
    other = "right" if hand == "lead" else "left"  # for an orthodox fighter
    return (low[hand] - 0.25) / 0.25 + 0.6, {"hand": hand, "value": pct(low[hand]), "other": other}


def _num(v: float) -> str:
    return f"{v:g}"


# -- the rules ---------------------------------------------------------------------------------
RULES: tuple[Rule, ...] = (
    # ---------------------------------------------------------------- output
    Rule(
        "work_rate_high",
        "strength",
        "Busy, consistent output",
        at_least("offense.strikes_per_min", 35, "offense.strikes", 10, _num),
        "Throws {value} strikes a minute.",
        "Output wins rounds, keeps the opponent reacting instead of planning, and hides your "
        "real attacks among the volume.",
        "Keep the pace, and make sure the volume has purpose: mix levels and finish on a "
        "defensive move rather than just adding punches.",
        (
            Drill(
                "Pace rounds",
                "Heavy bag at sparring pace with a coach calling level changes (head/body).",
                "3 x 3 min",
            ),
        ),
        ("combination",),
    ),
    Rule(
        "work_rate_low",
        "improve",
        "Low output",
        at_most("offense.strikes_per_min", 15, "offense.visible_s", 30, _num),
        "Only {value} strikes a minute.",
        "Long gaps let the opponent set the rhythm and score freely; judges and coaches read "
        "inactivity as losing the exchange.",
        "Always have a lead-hand job: touch, feint or jab every 2 seconds, and answer every "
        "opponent attack with at least one shot back.",
        (
            Drill(
                "2-second rule",
                "Shadowbox or spar light: a jab, feint or touch at least every 2 seconds.",
                "3 x 3 min",
            ),
            Drill(
                "Return-fire",
                "Partner throws single shots; you must answer each one with a 1-2.",
                "3 x 2 min",
            ),
        ),
    ),
    Rule(
        "jab_leads",
        "strength",
        "Works behind the jab",
        at_least("offense.jab_share", 0.35, "offense.punches", 8),
        "{value} of punches are jabs.",
        "The jab measures range, blinds the opponent and sets up everything else at low risk.",
        "Now vary it: double jabs, jabs to the body, and a feint-jab before the power hand.",
        (
            Drill(
                "Jab variety on pads",
                "Pads call jab-head, jab-body, double jab, feint-jab-cross.",
                "4 x 2 min",
            ),
        ),
        ("jab",),
    ),
    Rule(
        "rarely_jabs",
        "improve",
        "Rarely jabs",
        at_most("offense.jab_share", 0.15, "offense.punches", 8),
        "Only {value} of punches are jabs.",
        "Leading with power shots is slow and telegraphed; without the jab you walk onto counters "
        "and struggle to find range.",
        "Start every exchange with the jab: jab to enter, jab to exit, jab to reset.",
        (
            Drill(
                "Jab-only round",
                "Light sparring where you may only jab; partner is free.",
                "2 x 2 min",
            ),
            Drill(
                "Double-jab entries",
                "On pads or bag, step in behind a double jab before any other punch.",
                "3 x 2 min",
            ),
        ),
        ("cross", "lead_hook", "rear_hook"),
    ),
    Rule(
        "combinations",
        "strength",
        "Throws in combinations",
        at_least("offense.share_in_combinations", 0.45, "offense.strikes", 8),
        "{value} of strikes come in combinations.",
        "Combinations overwhelm a single defensive reaction; the second and third shots land "
        "while the opponent is still dealing with the first.",
        "Build the finish: end combinations with a defensive move or an angle change.",
        (
            Drill(
                "Combo + exit",
                "Pads: 3-4 punch combinations that always finish with a slip, roll or pivot.",
                "4 x 2 min",
            ),
        ),
        ("combination",),
    ),
    Rule(
        "single_shots",
        "improve",
        "One shot at a time",
        at_most("offense.share_in_combinations", 0.2, "offense.strikes", 10),
        "Only {value} of strikes are part of a combination.",
        "Single shots are easy to read and parry; the opponent defends once and is free to counter.",
        "Put punches together: at minimum a 1-2, and follow a landed shot with another.",
        (
            Drill(
                "Numbers on pads",
                "Coach calls 2-, 3- and 4-punch combinations at random.",
                "4 x 2 min",
            ),
            Drill(
                "Combination shadowboxing",
                "Shadowbox throwing only combinations of 2 or more.",
                "3 x 3 min",
            ),
        ),
    ),
    Rule(
        "one_handed",
        "improve",
        "Rarely uses the rear hand",
        at_most("offense.rear_hand_share", 0.15, "offense.punches", 10),
        "Only {value} of punches come from the rear hand.",
        "The rear hand is the power hand; without it the opponent only has to respect the jab.",
        "Follow the jab with the cross; turn the rear hip and pivot the rear foot so it lands with weight.",
        (
            Drill(
                "1-2 ladder",
                "Bag: jab-cross, jab-cross-jab, jab-cross-hook; every set includes the rear hand.",
                "3 x 3 min",
            ),
        ),
        ("jab",),
    ),
    Rule(
        "counters",
        "strength",
        "Counters well",
        at_least("offense.counter_rate", 0.35, "defense.incoming_in_range", 8),
        "Answers {value} of incoming strikes with a counter within a second.",
        "Countering punishes the opponent for attacking and makes them hesitant to lead.",
        "Add a defensive move before the counter (slip-counter, catch-counter) so you counter from safety.",
        (
            Drill(
                "Slip-and-counter",
                "Partner jabs; slip outside and return a cross or hook.",
                "3 x 2 min",
            ),
        ),
        basis="defense",
    ),
    Rule(
        "no_counters",
        "improve",
        "Doesn't fire back",
        at_most("offense.counter_rate", 0.1, "defense.incoming_in_range", 8),
        "Answers only {value} of incoming strikes with a counter.",
        "If attacking you is free, the opponent will keep doing it; every exchange should cost them.",
        "After every block or slip, return fire immediately: the opponent is open right after they punch.",
        (
            Drill(
                "Catch-and-return",
                "Partner throws single shots; catch each one and return a 1-2 instantly.",
                "3 x 2 min",
            ),
            Drill(
                "Slip-and-counter", "Partner jabs; slip outside and return a cross.", "3 x 2 min"
            ),
        ),
        ("block", "slip", "no_defense"),
        basis="defense",
    ),
    # ---------------------------------------------------------------- guard
    Rule(
        "return_to_guard",
        "strength",
        "Hands come back after punching",
        at_least("offense.return_to_guard_rate", 0.85, "offense.punches", 8),
        "{value} of punches return to guard within half a second.",
        "Punches are the moment you are most exposed; a fast return is what keeps counters off.",
        "Keep it under fatigue: this habit usually breaks in the last minute of a round.",
        (
            Drill(
                "Tired-rounds check",
                "Film the last round of sparring and count returns; aim to match the first round.",
                "every session",
            ),
        ),
        ("jab", "cross"),
        basis="guard",
    ),
    Rule(
        "slow_return",
        "improve",
        "Leaves the hands out after punching",
        at_most("offense.return_to_guard_rate", 0.6, "offense.punches", 8),
        "Only {value} of punches come back to guard within half a second.",
        "A hand left out after punching is the most common way to get countered.",
        "Punch back to the face: the hand retracts along the same line it went out, straight to the cheek.",
        (
            Drill(
                "Jab-touch",
                "Shadowbox: after every punch, touch your cheekbone with that glove before the next one.",
                "3 x 2 min",
            ),
            Drill(
                "Pad slap-back",
                "Pad holder slaps back after every combination; you must be covered in time.",
                "4 x 2 min",
            ),
        ),
        ("slow_return",),
        basis="guard",
    ),
    Rule(
        "low_hand",
        "improve",
        "Carries the {hand} hand low",
        _low_hand,
        "The {hand} hand sits below shoulder height for {value} of in-range time.",
        "A low hand leaves the chin open, and the opponent's {other} hand has a clear path to it.",
        "Rest the {hand} glove on the cheekbone when in range; if you like a low lead as a style, "
        "only drop it at distance and bring it back up as you step in.",
        (
            Drill(
                "Mirror guard check",
                "Shadowbox facing a mirror; the glove may not drop below the cheek.",
                "3 x 2 min",
            ),
            Drill(
                "Partner tap",
                "Light sparring: partner taps your head whenever the hand drops.",
                "2 x 2 min",
            ),
        ),
        ("guard_drop",),
        {"hand": "{hand}"},
        basis="guard",
    ),
    Rule(
        "chin_up",
        "improve",
        "Chin up",
        at_least("guard.chin_up_share", 0.1, "in_range_s", 20),
        "Chin is raised for {value} of in-range time.",
        "A raised chin exposes the jaw and the neck and is how knockdowns happen.",
        "Chin tucked, eyes looking up through the brows, lead shoulder raised to cover the jaw.",
        (
            Drill(
                "Ball under the chin",
                "Shadowbox holding a tennis ball between chin and chest.",
                "3 x 2 min",
            ),
        ),
        ("chin_up",),
        basis="guard",
    ),
    Rule(
        "square_stance",
        "improve",
        "Square stance",
        at_least("guard.square_stance_share", 0.15, "in_range_s", 20),
        "Feet are square to the opponent for {value} of in-range time.",
        "Square feet cost balance, power and a smaller target, and make it hard to move off the line.",
        "Lead foot forward, rear heel up, about shoulder-width apart; the lead toe points at the opponent.",
        (
            Drill(
                "Line drill",
                "Tape a line on the floor; step, pivot and punch keeping a foot on each side of it.",
                "3 x 2 min",
            ),
        ),
        ("square_stance",),
        basis="guard",
    ),
    Rule(
        "over_reach",
        "improve",
        "Over-reaching on punches",
        at_least("offense.over_leans", 3, "offense.punches", 8, lambda v: str(v)),
        "The head goes past the lead foot on {value} punches.",
        "Leaning over the lead foot kills balance and leaves you out of position for the next "
        "punch and open to counters.",
        "Step to close distance instead of leaning; keep the head over the lead knee at the end of the punch.",
        (
            Drill(
                "Range bag work",
                "Bag: after each punch check your head is over your lead knee; step in when out of range.",
                "3 x 2 min",
            ),
        ),
        ("over_lean",),
        basis="guard",
    ),
    # ---------------------------------------------------------------- defence
    Rule(
        "defense_responsive",
        "strength",
        "Responsive defence",
        at_least("defense.defense_rate", 0.6, "defense.incoming_in_range", 6),
        "Defends {value} of incoming strikes in range.",
        "Not getting hit is the foundation everything else is built on.",
        "Turn defence into offence: every defensive move should set up a counter.",
        (
            Drill(
                "Defend-then-counter",
                "Partner attacks; every defence must be followed by a counter.",
                "3 x 2 min",
            ),
        ),
        ("slip", "duck", "block"),
        basis="defense",
    ),
    Rule(
        "defense_static",
        "improve",
        "Static when strikes come in",
        at_most("defense.defense_rate", 0.4, "defense.incoming_in_range", 6),
        "Defends only {value} of incoming strikes in range.",
        "Taking clean shots adds up over rounds and hands the opponent the initiative.",
        "React to every shot: at minimum a high guard catch; better, a slip or a step out of range.",
        (
            Drill("Slip rope", "Walk a slip rope, slipping under it every step.", "3 x 2 min"),
            Drill(
                "Reaction drill",
                "Partner throws slow single shots; you must block, slip or step back every one.",
                "3 x 2 min",
            ),
        ),
        ("no_defense",),
        basis="defense",
    ),
    Rule(
        "head_movement",
        "strength",
        "Good head movement",
        at_least("defense.head_movement_rate", 0.35, "defense.incoming_in_range", 6),
        "Slips or rolls under {value} of incoming strikes.",
        "Making the opponent miss keeps your hands free to counter and drains their energy.",
        "Keep slips small and counter off them; big slips cost balance.",
        (Drill("Slip-counter", "Partner jabs; small slip outside, return a cross.", "3 x 2 min"),),
        ("slip", "duck"),
        basis="defense",
    ),
    Rule(
        "arms_only_defense",
        "improve",
        "Relies on the high guard alone",
        lambda m: (
            (0.6, {"value": pct(_g(m, "defense.block_rate"))})
            if (_g(m, "defense.incoming_in_range") or 0) >= 8
            and (_g(m, "defense.block_rate") or 0) >= 0.5
            and (_g(m, "defense.head_movement_rate") or 0) <= 0.15
            else None
        ),
        "Blocks {value} of incoming strikes with the arms, with almost no head movement.",
        "Blocking absorbs shots and keeps you in place; a stationary target is easy to combo.",
        "Mix in head movement: slip the jab, roll under the hook, and move the feet after blocking.",
        (
            Drill(
                "Block-slip-roll",
                "Partner calls jab/hook/cross; you slip the jab, roll the hook, catch the cross.",
                "3 x 2 min",
            ),
        ),
        ("block",),
        basis="defense",
    ),
    Rule(
        "head_still_after_striking",
        "improve",
        "Head stays on the centre line after attacking",
        at_most("offense.head_move_after_striking", 0.2, "offense.strikes", 8),
        "The head moves off line after only {value} of attacks.",
        "The opponent's counter comes right after your attack; a head left on the centre line is where it lands.",
        "Finish every combination with a slip, roll, pivot or step out.",
        (
            Drill(
                "Punch-then-move",
                "Pads: every combination ends with a defensive move called by the holder.",
                "4 x 2 min",
            ),
        ),
        ("jab", "cross"),
        basis="defense",
    ),
    # ---------------------------------------------------------------- kicking arts
    Rule(
        "kicks_hands_down",
        "improve",
        "Kicks with the hands down",
        at_least("offense.kicks_hands_down", 2, "offense.kicks", 3, lambda v: str(v)),
        "The guarding hand drops on {value} kicks.",
        "While a kick is out you are on one leg; an open face is where the counter punch lands.",
        "Swing the kicking-side arm for balance but keep the opposite glove glued to the cheek.",
        (
            Drill(
                "Shadow kicks with a guard check",
                "Partner taps the open side of the head during slow kicks.",
                "3 x 10 kicks per side",
            ),
        ),
        ("kick_no_guard",),
        disciplines=KICKING,
        basis="kicks",
    ),
    Rule(
        "not_checking",
        "improve",
        "Not checking kicks",
        at_most("defense.kick_defense_rate", 0.3, "defense.kicks_incoming", 4),
        "Reacts to only {value} of incoming leg kicks.",
        "Unchecked low kicks accumulate; by the later rounds they take away movement and power.",
        "Lift and turn the lead shin out to meet low kicks, or step back out of range.",
        (
            Drill(
                "Check drill",
                "Partner throws light low kicks at random; check every one.",
                "3 x 2 min",
            ),
        ),
        ("no_defense",),
        disciplines=KICKING,
        basis="kicks",
    ),
    Rule(
        "checks_kicks",
        "strength",
        "Checks kicks",
        at_least("defense.check_rate", 0.5, "defense.kicks_incoming", 4),
        "Checks {value} of incoming leg kicks.",
        "Checking hurts the kicker and takes their best long-range weapon away.",
        "Counter straight off the check: the kicker is on one leg.",
        (
            Drill(
                "Check-and-counter",
                "Partner low kicks; check and return a cross or a kick.",
                "3 x 2 min",
            ),
        ),
        ("check",),
        disciplines=KICKING,
        basis="kicks",
    ),
    Rule(
        "hands_only",
        "improve",
        "Not using the legs",
        at_most("offense.kick_share", 0.1, "offense.strikes", 12),
        "Only {value} of strikes are kicks.",
        "Kicks score from outside punching range and attack the legs and body the opponent "
        "cannot cover with the gloves.",
        "End punch combinations with a kick (jab-cross-low kick) and use the teep to control distance.",
        (
            Drill(
                "Punch-kick combos",
                "Bag: jab-cross-rear low kick; jab-rear body kick; teep to reset.",
                "4 x 2 min",
            ),
        ),
        disciplines=KICKING,
        basis="kicks",
    ),
    Rule(
        "mixes_hands_and_feet",
        "strength",
        "Mixes hands and feet",
        lambda m: (
            (0.4, {"value": pct(_g(m, "offense.kick_share"))})
            if (_g(m, "offense.strikes") or 0) >= 12
            and 0.2 <= (_g(m, "offense.kick_share") or 0) <= 0.6
            else None
        ),
        "{value} of strikes are kicks, mixed in with punches.",
        "Mixing ranges and levels makes the attack hard to read.",
        "Link them tighter: set up kicks with punches and punches with kick feints.",
        (
            Drill(
                "Level-change combos",
                "Pads: punches high, kick low; kick feint then punch.",
                "4 x 2 min",
            ),
        ),
        disciplines=KICKING,
        basis="kicks",
    ),
)
