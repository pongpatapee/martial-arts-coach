"""Command line interface: analyze, eval, bench, train."""

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
    discipline: Annotated[
        str | None, typer.Option(help="boxing | kickboxing | muay_thai (default: from config).")
    ] = None,
    method: Annotated[
        str | None, typer.Option(help="Strike detection method, e.g. rules or ensemble.")
    ] = None,
    verbose: bool = False,
) -> None:
    """Run the analysis pipeline on a sparring video."""
    from sparring.pipeline import RunContext, run

    _setup_logging(verbose)
    override = None
    if fighters:
        a, b = (int(x) for x in fighters.split(","))
        override = (a, b)
    cfg = Config.load(config)
    if discipline:
        cfg = cfg.model_copy(update={"discipline": discipline})
    if method:
        cfg = cfg.model_copy(
            update={"detection": cfg.detection.model_copy(update={"method": method})}
        )
    ctx = RunContext(video=video, run_dir=out, config=cfg, fighter_override=override)
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


def _sessions(data: list[str], config: Config):
    """`run_dir:labels.json` pairs -> labelled sessions."""
    import pandas as pd

    from sparring.detection.methods import Session
    from sparring.evaluation import Labels
    from sparring.io.video import VideoInfo

    sessions = []
    for item in data:
        run_dir, labels = item.split(":")
        fps = VideoInfo.load(Path(run_dir) / "video.json").fps
        feats = pd.read_parquet(Path(run_dir) / "features.parquet")
        sessions.append(Session.from_table(feats, fps, Labels.load(Path(labels)), name=run_dir))
    return sessions


DataOpt = Annotated[list[str], typer.Option("--data", help="run_dir:labels.json (repeatable).")]


@app.command()
def bench(
    data: DataOpt,
    methods: Annotated[str, typer.Option(help="Comma-separated methods.")] = "rules,ensemble",
    folds: Annotated[int, typer.Option(help="Time blocks per clip for cross-validation.")] = 5,
    config: Annotated[Path | None, typer.Option(help="TOML file overriding defaults.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write results as JSON here.")] = None,
) -> None:
    """Cross-validated comparison of strike-detection methods on labelled runs."""
    from sparring.detection.bench import cross_validate, format_results, pooled
    from sparring.detection.registry import new_method

    _setup_logging(False)
    cfg = Config.load(config)
    sessions = _sessions(data, cfg)
    results = []
    for name in methods.split(","):
        per = [cross_validate(new_method(name, cfg), s, folds) for s in sessions]
        results.append(pooled(per))
    typer.echo(format_results(results))
    if out:
        out.write_text(json.dumps([r.row() for r in results], indent=2))


@app.command()
def train(
    data: DataOpt,
    out: Annotated[Path, typer.Option(help="Where to write the trained model.")],
    method: Annotated[
        str, typer.Option(help="learned | hybrid | ensemble | temporal")
    ] = "ensemble",
    config: Annotated[Path | None, typer.Option(help="TOML file overriding defaults.")] = None,
) -> None:
    """Train a learned strike detector on labelled runs (use it via detection.model_path)."""
    from sparring.detection.bench import folds as time_folds
    from sparring.detection.registry import new_method, save_method

    _setup_logging(False)
    cfg = Config.load(config)
    # Train on time blocks so threshold selection can hold blocks out.
    blocks = [s.window(*b) for s in _sessions(data, cfg) for b in time_folds(s, 5)]
    model = new_method(method, cfg).fit(blocks)
    save_method(model, out)
    typer.echo(f"saved {method} model to {out}")


if __name__ == "__main__":
    app()
