"""Command line interface: analyze, label, eval, bench, train, clip, serve, coach.

Commands that take a TARGET accept a clip id (or clip folder) from the dataset, which supplies
the video, discipline, labels and run directory, or a plain video file / run directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from sparring.config import Config
from sparring.dataset import DEFAULT_DATASET, DEFAULT_RUNS, Clip, ClipError, Dataset

app = typer.Typer(add_completion=False, no_args_is_help=True)
clip_app = typer.Typer(no_args_is_help=True, help="Manage the clip dataset (datasets/clips).")
app.add_typer(clip_app, name="clip")

DatasetOpt = Annotated[
    Path, typer.Option("--dataset", envvar="SPARRING_DATASET", help="Dataset root.")
]
RunsOpt = Annotated[
    Path, typer.Option("--runs", envvar="SPARRING_RUNS", help="Where clip runs are stored.")
]


def _fail(msg: str) -> typer.Exit:
    typer.secho(msg, err=True, fg=typer.colors.RED)
    return typer.Exit(1)


def _resolve(target: str, dataset: Path) -> Clip | None:
    try:
        return Dataset(dataset).resolve(target)
    except ClipError as e:
        raise _fail(str(e)) from None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def analyze(
    target: Annotated[str, typer.Argument(help="Clip id, clip folder, or video file.")],
    out: Annotated[
        Path | None,
        typer.Option(help="Run directory for all stage artifacts (default for a clip: runs/<id>)."),
    ] = None,
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
    dataset: DatasetOpt = DEFAULT_DATASET,
    runs: RunsOpt = DEFAULT_RUNS,
    verbose: bool = False,
) -> None:
    """Run the analysis pipeline on a clip or a video file."""
    from sparring.pipeline import RunContext, run

    _setup_logging(verbose)
    clip = _resolve(target, dataset)
    if clip:
        try:
            clip.verify_video()
        except ClipError as e:
            raise _fail(str(e)) from None
        video, out = clip.video, out or clip.run_dir(runs)
        discipline = discipline or clip.meta.discipline
        if from_stage != "ingest" and clip.run_status(out) != "ok":
            raise _fail(f"{out} is not a current run of clip {clip.id}; re-run from ingest")
    elif out is None:
        raise _fail("--out is required when analysing a video file")
    else:
        video = Path(target)
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
    if clip and from_stage == "ingest":
        clip.stamp_run(out)
    run(ctx, from_stage=from_stage, to_stage=to_stage)


@app.command()
def label(
    target: Annotated[str, typer.Argument(help="Clip id, clip folder, or video file.")],
    labels: Annotated[
        Path | None, typer.Option(help="Labels JSON to create or edit (default: the clip's).")
    ] = None,
    run: Annotated[
        Path | None,
        typer.Option(help="Run dir whose events are shown as suggestions (default: the clip's)."),
    ] = None,
    port: int = 8765,
    dataset: DatasetOpt = DEFAULT_DATASET,
    runs: RunsOpt = DEFAULT_RUNS,
) -> None:
    """Open the labelling tool in the browser (local only)."""
    from sparring.labeling.server import serve

    clip = _resolve(target, dataset)
    if clip:
        try:
            clip.verify_video()
        except ClipError as e:
            raise _fail(str(e)) from None
        video, labels = clip.video, labels or clip.labels_path
        if run is None and clip.run_status(clip.run_dir(runs)) == "ok":
            run = clip.run_dir(runs)
    elif labels is None:
        raise _fail("--labels is required when labelling a video file")
    else:
        video = Path(target)
    serve(video, labels, run, port)


@app.command()
def serve(
    data: Annotated[Path, typer.Option(help="Where uploads and analyses are stored.")] = Path(
        "data"
    ),
    host: str = "127.0.0.1",
    port: int = 8000,
    max_upload_mb: int = 2048,
    reload: Annotated[
        bool, typer.Option(help="Auto-reload on code changes (development).")
    ] = False,
) -> None:
    """Run the web app: upload videos, watch progress, review analyses."""
    import uvicorn

    _setup_logging(False)
    if reload:
        import os

        os.environ["SPARRING_DATA"] = str(data)
        os.environ["SPARRING_MAX_UPLOAD_MB"] = str(max_upload_mb)
        uvicorn.run("sparring.web.main:app", host=host, port=port, reload=True)
        return
    from sparring.web.app import create_app

    uvicorn.run(create_app(data, max_upload_mb=max_upload_mb), host=host, port=port)


@app.command("coach")
def coach_cmd(
    run_dir: Path,
    llm: Annotated[bool, typer.Option(help="Also write a Claude-written coach's note.")] = False,
) -> None:
    """Print a run's coaching; with --llm, write coach_note.md in a coach's voice."""
    report = json.loads((run_dir / "report.json").read_text())
    for f, d in report["fighters"].items():
        typer.echo(f"Fighter {f}: {d['coaching']['summary']}")
    if llm:
        from sparring.coaching.llm import coach_note

        note = coach_note(report)
        (run_dir / "coach_note.md").write_text(note)
        typer.echo(f"wrote {run_dir / 'coach_note.md'}")


@app.command("eval")
def evaluate(
    target: Annotated[str, typer.Argument(help="Clip id, clip folder, or run directory.")],
    labels: Annotated[
        Path | None, typer.Argument(help="Labels JSON (only with a run directory).")
    ] = None,
    tolerance: Annotated[float, typer.Option(help="Match tolerance in seconds.")] = 0.15,
    dataset: DatasetOpt = DEFAULT_DATASET,
    runs: RunsOpt = DEFAULT_RUNS,
) -> None:
    """Score a run's events against hand labels (precision / recall / F1 per event type)."""
    from sparring.evaluation import evaluate_run, format_table

    clip = _resolve(target, dataset)
    if clip:
        try:
            run_dir, labels = _labelled_run(clip, runs)
        except ClipError as e:
            raise _fail(str(e)) from None
    elif labels is None:
        raise _fail("give a labels file, or a clip id instead of a run directory")
    else:
        run_dir = Path(target)
    result = evaluate_run(run_dir, labels, tolerance)
    typer.echo(format_table(result))
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2))


def _labelled_run(clip: Clip, runs: Path) -> tuple[Path, Path]:
    """A clip's current run and its labels, verifying the video they both describe."""
    if not clip.has_labels:
        raise ClipError(f"clip {clip.id} has no labels yet; run `sparring label {clip.id}`")
    clip.verify_video()
    return clip.require_run(clip.run_dir(runs)), clip.labels_path


def _training_data(
    data: list[str] | None,
    clips: list[str] | None,
    dataset: Path,
    runs: Path,
    split: str | None,
) -> list[tuple[str, Path, Path]]:
    """(name, run_dir, labels) for the chosen clips and `run:labels` pairs.

    With neither given, every labelled clip in the dataset (optionally of one split)."""
    ds = Dataset(dataset)
    chosen = [ds.get(c) for c in clips or []]
    if not data and not clips:
        chosen = [c for c in ds.clips() if c.has_labels and split in (None, c.meta.split)]
        if not chosen:
            raise ClipError(
                f"no labelled clips in {ds.clips_dir}" + (f" (split {split})" if split else "")
            )
    out = [(c.id, *_labelled_run(c, runs)) for c in chosen]
    for item in data or []:
        run_dir, labels = item.split(":")
        out.append((run_dir, Path(run_dir), Path(labels)))
    return out


def _sessions(items: list[tuple[str, Path, Path]]):
    """(name, run_dir, labels) -> labelled sessions."""
    import pandas as pd

    from sparring.detection.methods import Session
    from sparring.evaluation import Labels
    from sparring.io.video import VideoInfo

    sessions = []
    for name, run_dir, labels in items:
        fps = VideoInfo.load(run_dir / "video.json").fps
        feats = pd.read_parquet(run_dir / "features.parquet")
        sessions.append(Session.from_table(feats, fps, Labels.load(labels), name=name))
    typer.echo("data: " + ", ".join(name for name, _, _ in items))
    return sessions


DataOpt = Annotated[
    list[str] | None, typer.Option("--data", help="Extra run_dir:labels.json pair (repeatable).")
]
ClipsOpt = Annotated[
    list[str] | None,
    typer.Option("--clip", help="Labelled clip id (repeatable). Default: all labelled clips."),
]


@app.command()
def bench(
    clips: ClipsOpt = None,
    data: DataOpt = None,
    methods: Annotated[str, typer.Option(help="Comma-separated methods.")] = "rules,ensemble",
    folds: Annotated[int, typer.Option(help="Time blocks per clip for cross-validation.")] = 5,
    config: Annotated[Path | None, typer.Option(help="TOML file overriding defaults.")] = None,
    out: Annotated[Path | None, typer.Option(help="Write results as JSON here.")] = None,
    dataset: DatasetOpt = DEFAULT_DATASET,
    runs: RunsOpt = DEFAULT_RUNS,
) -> None:
    """Cross-validated comparison of strike-detection methods on labelled clips."""
    from sparring.detection.bench import cross_validate, format_results, pooled
    from sparring.detection.registry import new_method

    _setup_logging(False)
    cfg = Config.load(config)
    try:
        sessions = _sessions(_training_data(data, clips, dataset, runs, split=None))
    except ClipError as e:
        raise _fail(str(e)) from None
    results = []
    for name in methods.split(","):
        per = [cross_validate(new_method(name, cfg), s, folds) for s in sessions]
        results.append(pooled(per))
    typer.echo(format_results(results))
    if out:
        out.write_text(json.dumps([r.row() for r in results], indent=2))


@app.command()
def train(
    out: Annotated[Path, typer.Option(help="Where to write the trained model.")],
    method: Annotated[
        str, typer.Option(help="learned | hybrid | ensemble | temporal")
    ] = "ensemble",
    config: Annotated[Path | None, typer.Option(help="TOML file overriding defaults.")] = None,
    clips: ClipsOpt = None,
    data: DataOpt = None,
    dataset: DatasetOpt = DEFAULT_DATASET,
    runs: RunsOpt = DEFAULT_RUNS,
) -> None:
    """Train a learned strike detector (use it via detection.model_path).

    By default trains on every labelled clip in the train split; test clips are left out."""
    from sparring.detection.bench import folds as time_folds
    from sparring.detection.registry import new_method, save_method

    _setup_logging(False)
    cfg = Config.load(config)
    try:
        sessions = _sessions(_training_data(data, clips, dataset, runs, split="train"))
    except ClipError as e:
        raise _fail(str(e)) from None
    # Train on time blocks so threshold selection can hold blocks out.
    blocks = [s.window(*b) for s in sessions for b in time_folds(s, 5)]
    model = new_method(method, cfg).fit(blocks)
    save_method(model, out)
    typer.echo(f"saved {method} model to {out}")


# ---- clip dataset ------------------------------------------------------------------------------


@clip_app.command("add")
def clip_add(
    video: Path,
    discipline: Annotated[str, typer.Option(help="boxing | kickboxing | muay_thai")],
    clip_id: Annotated[
        str | None,
        typer.Option(
            "--id", help="Clip id, e.g. boxing-homegym-2026-10-04-r2 (default: from the file name)."
        ),
    ] = None,
    split: Annotated[
        str, typer.Option(help="train | test (test clips are never trained on).")
    ] = "train",
    recorded: Annotated[str | None, typer.Option(help="Recording date, YYYY-MM-DD.")] = None,
    camera: Annotated[str, typer.Option(help='Camera setup, e.g. "side-on, chest height".')] = "",
    source: Annotated[str, typer.Option(help="Where the footage came from / consent.")] = "",
    notes: str = "",
    move: Annotated[bool, typer.Option(help="Move the video instead of copying it.")] = False,
    dataset: DatasetOpt = DEFAULT_DATASET,
) -> None:
    """Add a video to the dataset as a new clip."""
    from datetime import date

    from pydantic import ValidationError

    from sparring.dataset import slugify

    clip_id = clip_id or slugify(f"{discipline}-{video.stem}")
    try:
        clip = Dataset(dataset).add(
            video,
            clip_id,
            discipline,
            move=move,
            split=split,
            recorded=date.fromisoformat(recorded) if recorded else None,
            camera=camera,
            source=source,
            notes=notes,
        )
    except (ClipError, ValidationError, ValueError) as e:
        raise _fail(str(e)) from None
    m = clip.meta
    typer.echo(f"added {clip.id}: {m.duration_s:.1f}s @ {m.fps:g} fps, {m.width}x{m.height}")
    if m.fps is not None and m.fps < 50:
        typer.echo("note: under 50 fps; fast punches span few frames (record at 60 fps if you can)")
    typer.echo(f"next: sparring analyze {clip.id}  then  sparring label {clip.id}")


@clip_app.command("list")
def clip_list(dataset: DatasetOpt = DEFAULT_DATASET, runs: RunsOpt = DEFAULT_RUNS) -> None:
    """List clips with their labels and analysis status."""
    import json as _json

    try:
        clips = Dataset(dataset).clips()
    except ClipError as e:
        raise _fail(str(e)) from None
    if not clips:
        typer.echo(f"no clips in {Dataset(dataset).clips_dir}; add one with `sparring clip add`")
        return
    rows = [("clip", "discipline", "split", "length", "fps", "labels", "run")]
    for c in clips:
        n = len(_json.loads(c.labels_path.read_text())["events"]) if c.has_labels else 0
        rows.append(
            (
                c.id,
                c.meta.discipline,
                c.meta.split,
                f"{c.meta.duration_s:.0f}s" if c.meta.duration_s else "?",
                f"{c.meta.fps:g}" if c.meta.fps else "?",
                str(n) if c.has_labels else "-",
                c.run_status(c.run_dir(runs)),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        typer.echo("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip())


@clip_app.command("check")
def clip_check(dataset: DatasetOpt = DEFAULT_DATASET) -> None:
    """Verify every clip: video present and unchanged (hash), labels readable and valid."""
    try:
        clips = Dataset(dataset).clips()
    except ClipError as e:
        raise _fail(str(e)) from None
    problems = [p for c in clips for p in c.problems()]
    for p in problems:
        typer.secho(p, fg=typer.colors.RED)
    if problems:
        raise typer.Exit(1)
    typer.echo(f"{len(clips)} clips ok")


if __name__ == "__main__":
    app()
