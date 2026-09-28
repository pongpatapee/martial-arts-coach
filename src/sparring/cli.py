"""Command line interface: `sparring analyze`, `sparring eval`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from sparring.config import Config

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def analyze(
    video: Path,
    out: Annotated[Path, typer.Option(help="Run directory for all stage artifacts.")],
    from_stage: Annotated[str, typer.Option(help="First stage to (re)run.")] = "ingest",
    to_stage: Annotated[str, typer.Option(help="Last stage to run.")] = "render",
    fighters: Annotated[
        str | None, typer.Option(help="Override fighter selection with two track ids, e.g. 3,7.")
    ] = None,
    config: Annotated[Path | None, typer.Option(help="TOML file overriding defaults.")] = None,
    verbose: bool = False,
) -> None:
    """Run the analysis pipeline on a sparring video."""
    from sparring.pipeline import RunContext, run

    _setup_logging(verbose)
    override = None
    if fighters:
        a, b = (int(x) for x in fighters.split(","))
        override = (a, b)
    ctx = RunContext(
        video=video, run_dir=out, config=Config.load(config), fighter_override=override
    )
    run(ctx, from_stage=from_stage, to_stage=to_stage)


@app.command("eval")
def evaluate(
    run_dir: Path,
    labels: Path,
    tolerance: Annotated[float, typer.Option(help="Match tolerance in seconds.")] = 0.15,
) -> None:
    """Score a run's events against hand labels (precision / recall / F1 per event type)."""
    from sparring.evaluation import evaluate_run, format_table

    result = evaluate_run(run_dir, labels, tolerance)
    typer.echo(format_table(result))
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
