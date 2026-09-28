"""Coaching report: metrics -> ranked strengths / improvement areas with evidence timestamps.

The JSON is the product; the Markdown is a readable rendering of it. The structure is kept flat
and explicit so an LLM can later turn it into coach-style prose without re-deriving anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from sparring.analysis.metrics import fighter_metrics
from sparring.config import Config
from sparring.events.base import Event
from sparring.io.video import VideoInfo
from sparring.skeleton import FIGHTERS


@dataclass
class Finding:
    kind: str  # "strength" | "improve"
    key: str
    title: str
    detail: str
    score: float  # how far past the threshold, >= 0; used for ranking
    evidence: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "detail": self.detail,
            "score": round(self.score, 3),
            "evidence_t": [round(t, 2) for t in self.evidence],
        }


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _evidence(
    events: list[Event], fighter: str, types: set[str], n: int, spacing: float = 1.0, **match
) -> list[float]:
    """Timestamps of the n most confident matching events, at least `spacing` seconds apart."""
    chosen = [
        e
        for e in events
        if e.fighter == fighter
        and e.type in types
        and all(e.evidence.get(k) == v for k, v in match.items())
    ]
    chosen.sort(key=lambda e: -e.confidence)
    picked: list[float] = []
    for e in chosen:
        if all(abs(e.peak_t - t) >= spacing for t in picked):
            picked.append(e.peak_t)
        if len(picked) == n:
            break
    return sorted(picked)


def findings_for(fighter: str, m: dict, events: list[Event], cfg: Config) -> list[Finding]:
    r = cfg.report
    n = r.max_evidence
    sp = r.evidence_min_spacing_s
    out: list[Finding] = []
    off, dfn, grd = m["offense"], m["defense"], m["guard"]
    punch_types = {"jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"}

    ppm = off["punches_per_min"]
    if ppm is not None and off["punches"] >= 3:
        if ppm >= r.good_punches_per_min:
            out.append(
                Finding(
                    "strength",
                    "work_rate",
                    "High work rate",
                    f"{ppm} punches per minute.",
                    (ppm - r.good_punches_per_min) / r.good_punches_per_min,
                    _evidence(events, fighter, {"combination"}, n, sp),
                )
            )
        elif ppm <= r.bad_punches_per_min:
            out.append(
                Finding(
                    "improve",
                    "work_rate",
                    "Low output",
                    f"Only {ppm} punches per minute; look for more chances to let the hands go.",
                    (r.bad_punches_per_min - ppm) / r.bad_punches_per_min,
                )
            )

    js = off["jab_share"]
    if js is not None and off["punches"] >= 5:
        if js >= r.good_jab_share:
            out.append(
                Finding(
                    "strength",
                    "jab",
                    "Works behind the jab",
                    f"{_pct(js)} of punches are jabs.",
                    (js - r.good_jab_share) / r.good_jab_share,
                    _evidence(events, fighter, {"jab"}, n, sp),
                )
            )
        elif js <= r.bad_jab_share:
            out.append(
                Finding(
                    "improve",
                    "jab",
                    "Rarely jabs",
                    f"Only {_pct(js)} of punches are jabs; use the jab to find range and set up power shots.",
                    (r.bad_jab_share - js) / r.bad_jab_share + 0.2,
                )
            )

    sc = off["share_in_combinations"]
    if sc is not None and off["punches"] >= 5 and sc >= 0.4:
        out.append(
            Finding(
                "strength",
                "combinations",
                "Throws in combinations",
                f"{_pct(sc)} of punches come in combinations (avg {off['mean_combination_length']} punches).",
                sc - 0.4,
                _evidence(events, fighter, {"combination"}, n, sp),
            )
        )

    rg = off["return_to_guard_rate"]
    if rg is not None and off["punches"] >= 3:
        if rg >= r.good_return_to_guard_rate:
            out.append(
                Finding(
                    "strength",
                    "return_to_guard",
                    "Hands come back after punching",
                    f"{_pct(rg)} of punches return to guard within {cfg.faults.return_to_guard_max_s}s "
                    f"(median {off['median_return_s']}s).",
                    (rg - r.good_return_to_guard_rate)
                    / (1 - r.good_return_to_guard_rate + 1e-6)
                    * 0.5
                    + 0.2,
                    _evidence(events, fighter, punch_types, n, sp),
                )
            )
        elif rg <= r.bad_return_to_guard_rate:
            out.append(
                Finding(
                    "improve",
                    "return_to_guard",
                    "Leaves hands out after punching",
                    f"Only {_pct(rg)} of punches return to guard within {cfg.faults.return_to_guard_max_s}s "
                    f"(median {off['median_return_s']}s). Bring the hand straight back to the chin.",
                    (r.bad_return_to_guard_rate - rg) / r.bad_return_to_guard_rate + 0.3,
                    _evidence(events, fighter, {"slow_return"}, n, sp),
                )
            )

    dpm = grd["drops_per_min_in_range"]
    low = {h: v for h, v in grd["low_share_by_hand"].items() if v is not None}
    if low and max(low.values()) >= r.bad_guard_low_share:
        # A hand held low most of the time is a habit of the guard, not a series of lapses.
        hand = max(low, key=low.get)
        out.append(
            Finding(
                "improve",
                "guard_drop",
                f"Carries the {hand} hand low",
                f"The {hand} hand sits below shoulder height for {_pct(low[hand])} of in-range time. "
                "Keep it up by the chin; a low hand leaves the head open to the opponent's "
                f"{'right' if hand == 'lead' else 'left'} hand.",
                (low[hand] - r.bad_guard_low_share) / r.bad_guard_low_share + 0.5,
                _evidence(events, fighter, {"guard_drop"}, n, sp, hand=hand),
            )
        )
    elif dpm is not None and grd["drops"] >= 2 and dpm >= r.bad_guard_drop_per_min:
        hand = max(grd["drops_by_hand"], key=grd["drops_by_hand"].get)
        out.append(
            Finding(
                "improve",
                "guard_drop",
                f"Drops the {hand} hand",
                f"Guard dropped {grd['drops']} times in range ({dpm}/min, {grd['drop_seconds']}s total), "
                f"mostly the {hand} hand.",
                (dpm - r.bad_guard_drop_per_min) / r.bad_guard_drop_per_min + 0.4,
                _evidence(events, fighter, {"guard_drop"}, n, sp, hand=hand),
            )
        )

    dr = dfn["defense_rate"]
    if dr is not None and dfn["incoming_in_range"] >= 4:
        resp = dfn["responses"]
        mix = ", ".join(
            f"{k.replace('_', ' ')} {v}" for k, v in resp.items() if v and k != "no_defense"
        )
        if dr >= r.good_defense_rate:
            out.append(
                Finding(
                    "strength",
                    "defense",
                    "Responsive defence",
                    f"Defends {_pct(dr)} of incoming punches in range ({mix}).",
                    (dr - r.good_defense_rate) / r.good_defense_rate + 0.1,
                    _evidence(events, fighter, {"slip", "duck", "block"}, n, sp),
                )
            )
        elif dr <= r.bad_defense_rate:
            out.append(
                Finding(
                    "improve",
                    "defense",
                    "Static when punches come in",
                    f"No defensive reaction to {_pct(1 - dr)} of incoming punches in range.",
                    (r.bad_defense_rate - dr) / r.bad_defense_rate + 0.3,
                    _evidence(events, fighter, {"no_defense"}, n, sp),
                )
            )
        hm = dfn["head_movement_rate"]
        if hm is not None and hm >= 0.25:
            out.append(
                Finding(
                    "strength",
                    "head_movement",
                    "Good head movement",
                    f"Slips or ducks {_pct(hm)} of incoming punches.",
                    hm,
                    _evidence(events, fighter, {"slip", "duck"}, n, sp),
                )
            )

    hmp = off["head_move_after_punching"]
    if hmp is not None and off["punches"] >= 5:
        if hmp <= 0.2:
            out.append(
                Finding(
                    "improve",
                    "head_after_punching",
                    "Head stays on the centre line after punching",
                    f"Head moves off line after only {_pct(hmp)} of punches; finish combinations with a slip, "
                    "roll or angle change.",
                    (0.2 - hmp) / 0.2 * 0.6 + 0.1,
                    _evidence(events, fighter, punch_types, n, sp),
                )
            )
        elif hmp >= 0.5:
            out.append(
                Finding(
                    "strength",
                    "head_after_punching",
                    "Moves the head after punching",
                    f"Head moves off line after {_pct(hmp)} of punches.",
                    hmp - 0.5 + 0.1,
                )
            )

    for key, title, advice, threshold in (
        ("chin_up_share", "Chin up", "Keep the chin tucked behind the lead shoulder.", 0.1),
        (
            "square_stance_share",
            "Square stance",
            "Keep the lead foot forward and turn the body side-on.",
            0.15,
        ),
    ):
        v = grd[key]
        if v is not None and v >= threshold:
            fault = "chin_up" if key.startswith("chin") else "square_stance"
            out.append(
                Finding(
                    "improve",
                    fault,
                    title,
                    f"{_pct(v)} of in-range time. {advice}",
                    (v - threshold) / threshold * 0.5 + 0.1,
                    _evidence(events, fighter, {fault}, n, sp),
                )
            )

    if off["over_leans"] >= 3:
        out.append(
            Finding(
                "improve",
                "over_lean",
                "Over-reaching on punches",
                f"Head goes past the lead foot on {off['over_leans']} punches; stay balanced over the feet.",
                off["over_leans"] / max(off["punches"], 1) + 0.1,
                _evidence(events, fighter, {"over_lean"}, n, sp),
            )
        )
    return out


def build_report(
    events: list[Event], feats: pd.DataFrame, info: VideoInfo, fighters_meta: dict, cfg: Config
) -> dict:
    fps = info.fps
    fighters = {}
    for f in FIGHTERS:
        m = fighter_metrics(f, events, feats, fps, cfg)
        found = findings_for(f, m, events, cfg)
        strengths = sorted((x for x in found if x.kind == "strength"), key=lambda x: -x.score)[:3]
        improve = sorted((x for x in found if x.kind == "improve"), key=lambda x: -x.score)[:3]
        fighters[f] = {
            "identity": "on the left at the start" if f == "A" else "on the right at the start",
            "approximate_colours": fighters_meta["colours"][f],
            "tracking": fighters_meta["fighters"][f],
            "metrics": m,
            "strengths": [x.as_dict() for x in strengths],
            "improve": [x.as_dict() for x in improve],
        }
    axis_ok = float(feats[feats.visible].axis_ok.mean()) if feats.visible.any() else 0.0
    notes = []
    if fps < cfg.ingest.min_recommended_fps:
        notes.append(
            f"Video is {fps:.0f} fps; fast punches may be under-detected. 60 fps recommended."
        )
    if axis_ok < 0.7:
        notes.append(
            f"Camera was side-on to the action only {_pct(axis_ok)} of the time; punch direction and stance "
            "are less reliable when filming along the line of attack."
        )
    return {
        "video": {"path": info.path, "duration_s": round(info.duration_s, 1), "fps": fps},
        "data_quality": {"side_on_share": round(axis_ok, 3), "notes": notes},
        "fighters": fighters,
    }


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def render_markdown(report: dict) -> str:
    lines = ["# Sparring analysis", ""]
    v = report["video"]
    lines.append(f"Video: `{v['path']}` ({v['duration_s']}s @ {v['fps']:.0f} fps)")
    for n in report["data_quality"]["notes"]:
        lines.append(f"> Note: {n}")
    lines.append("")
    for f, d in report["fighters"].items():
        lines.append(f"## Fighter {f} ({d['identity']})")
        m = d["metrics"]
        off, dfn, grd = m["offense"], m["defense"], m["guard"]
        by = ", ".join(f"{k.replace('_', ' ')} {c}" for k, c in off["by_type"].items() if c)
        lines += [
            "",
            f"- Punches: **{off['punches']}** ({off['punches_per_min']}/min): {by or 'none'}",
            f"- Combinations: {off['combinations']} (share of punches in combos: {off['share_in_combinations']})",
            f"- Return to guard rate: {off['return_to_guard_rate']} (median {off['median_return_s']}s)",
            f"- Incoming punches in range: {dfn['incoming_in_range']}; defence rate {dfn['defense_rate']}; "
            + ", ".join(f"{k.replace('_', ' ')} {c}" for k, c in dfn["responses"].items() if c),
            f"- Guard drops in range: {grd['drops']} ({grd['drop_seconds']}s)",
            "",
        ]
        for title, key in (("Strengths", "strengths"), ("To improve", "improve")):
            lines.append(f"### {title}")
            if not d[key]:
                lines.append("- Nothing stood out.")
            for x in d[key]:
                ev = ", ".join(_fmt_t(t) for t in x["evidence_t"])
                lines.append(
                    f"- **{x['title']}**: {x['detail']}" + (f" _Watch: {ev}_" if ev else "")
                )
            lines.append("")
    return "\n".join(lines)
