"""The analysis report: per-fighter metrics plus the coaching layer's feedback.

The JSON is the product; the Markdown is a readable rendering of it. The structure is flat and
explicit so an LLM can turn it into coach-style prose without re-deriving anything.
"""

from __future__ import annotations

import pandas as pd

from sparring.analysis.metrics import fighter_metrics
from sparring.coaching.engine import coach
from sparring.config import Config
from sparring.disciplines import get_discipline
from sparring.events.base import Event
from sparring.io.video import VideoInfo
from sparring.moves import MOVES
from sparring.skeleton import FIGHTERS


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def build_report(
    events: list[Event], feats: pd.DataFrame, info: VideoInfo, fighters_meta: dict, cfg: Config
) -> dict:
    discipline = get_discipline(cfg.discipline)
    fighters = {}
    for f in FIGHTERS:
        m = fighter_metrics(f, events, feats, info.fps, cfg)
        fighters[f] = {
            "identity": "on the left at the start" if f == "A" else "on the right at the start",
            "approximate_colours": fighters_meta["colours"][f],
            "tracking": fighters_meta["fighters"][f],
            "metrics": m,
            "coaching": coach(
                f,
                m,
                events,
                discipline.name,
                n_evidence=cfg.report.max_evidence,
                spacing=cfg.report.evidence_min_spacing_s,
            ),
        }
    axis_ok = float(feats[feats.visible].axis_ok.mean()) if feats.visible.any() else 0.0
    notes = []
    if info.fps < cfg.ingest.min_recommended_fps:
        notes.append(
            f"Video is {info.fps:.0f} fps; fast strikes may be under-detected. 60 fps recommended."
        )
    if axis_ok < 0.7:
        notes.append(
            f"Camera was side-on to the action only {_pct(axis_ok)} of the time; strike direction "
            "and stance are less reliable when filming along the line of attack."
        )
    return {
        "video": {"path": info.path, "duration_s": round(info.duration_s, 1), "fps": info.fps},
        "discipline": discipline.name,
        "detection_method": cfg.detection.method,
        "data_quality": {"side_on_share": round(axis_ok, 3), "notes": notes},
        "fighters": fighters,
    }


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _label(move: str) -> str:
    return MOVES[move].label if move in MOVES else move.replace("_", " ")


def _feedback_md(fb: dict) -> list[str]:
    watch = ", ".join(_fmt_t(t) for t in fb["watch"])
    lines = [f"- **{fb['title']}.** {fb['observation']}"]
    if fb["why"]:
        lines.append(f"  - *Why it matters:* {fb['why']}")
    if fb["fix"]:
        lines.append(f"  - *How:* {fb['fix']}")
    if watch:
        lines.append(f"  - *Watch:* {watch}")
    lines.append(f"  - *Confidence:* {fb['reliability'].split(':')[0]} ({fb['basis']} detection)")
    return lines


def render_markdown(report: dict) -> str:
    v = report["video"]
    title = report["discipline"].replace("_", " ").title()
    lines = [f"# {title} sparring analysis", ""]
    lines.append(f"Video: `{v['path']}` ({v['duration_s']}s @ {v['fps']:.0f} fps)")
    for n in report["data_quality"]["notes"]:
        lines.append(f"> Note: {n}")
    lines.append("")
    for f, d in report["fighters"].items():
        m, c = d["metrics"], d["coaching"]
        off, dfn, grd = m["offense"], m["defense"], m["guard"]
        lines += [f"## Fighter {f} ({d['identity']})", "", f"**{c['summary']}**", ""]
        mix = ", ".join(f"{_label(k)} {n}" for k, n in off["by_type"].items())
        responses = ", ".join(f"{_label(k)} {n}" for k, n in dfn["responses"].items() if n)
        low = grd["low_share_by_hand"]
        lines += [
            "| | |",
            "|---|---|",
            f"| Strikes | **{off['strikes']}** ({off['strikes_per_min']}/min): {mix or 'none'} |",
            f"| Combinations | {off['combinations']} (share of strikes: "
            f"{off['share_in_combinations']}) |",
            f"| Return to guard | {off['return_to_guard_rate']} "
            f"(median {off['median_return_s']}s) |",
            f"| Counters | {off['counter_rate']} of incoming strikes answered |",
            f"| Defence | {dfn['defense_rate']} of {dfn['incoming_in_range']} incoming: "
            f"{responses or '-'} |",
            f"| Hand held low | lead {low['lead']}, rear {low['rear']} of in-range time |",
        ]
        if dfn["kicks_incoming"]:
            lines.append(
                f"| Kick defence | {dfn['kick_defense_rate']} of {dfn['kicks_incoming']} "
                f"leg kicks (checked {dfn['check_rate']}) |"
            )
        lines.append("")
        for heading, key in (("Strengths", "strengths"), ("Priorities", "priorities")):
            lines.append(f"### {heading}")
            if not c[key]:
                lines.append("- Nothing stood out.")
            for fb in c[key]:
                lines += _feedback_md(fb)
            lines.append("")
        if c["session_plan"]:
            lines.append("### Next session")
            for d in c["session_plan"]:
                lines.append(f"- **{d['name']}** ({d['dose']}): {d['how']} _({d['for']})_")
            lines.append("")
    return "\n".join(lines)
