"""Optional: turn the structured report into a coach's note with Claude.

The rule-based coaching layer is the source of truth; this step only rewrites it in a coach's
voice. The model is told to use only facts present in the report, so it cannot invent
observations the video analysis did not make. Requires the `llm` extra (`uv sync --extra llm`)
and Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""

from __future__ import annotations

import json

MODEL = "claude-opus-5-5"

SYSTEM = """You are an experienced combat-sports coach writing post-sparring notes for two \
fighters. You are given the structured output of a video-analysis system: per-fighter metrics \
and rule-based coaching (strengths, priorities, drills, timestamps).

Rules:
- Use only facts present in the JSON. Do not invent observations, numbers or events.
- Keep every timestamp you cite exactly as given (convert seconds to m:ss).
- Some findings are marked with a basis whose reliability is limited; phrase those as things to \
check on the video, not as certainties.
- For each fighter write: a two-sentence overall read, the top one or two things to fix with the \
concrete cue and one drill, and one strength to keep. Be direct and encouraging, like a coach \
talking after the round.
- Output Markdown with one "## Fighter X" section per fighter. No preamble."""


def build_messages(report: dict) -> list[dict]:
    return [{"role": "user", "content": json.dumps(report, indent=1)}]


def coach_note(report: dict, client=None) -> str:
    """Return a Markdown coach's note. Raises RuntimeError if the model declines."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=build_messages(report),
        output_config={"effort": "medium"},
        # If the request is declined by a safety classifier, re-run it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to write the coach's note")
    return "".join(block.text for block in response.content if block.type == "text")
