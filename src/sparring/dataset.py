"""Sample footage as a dataset of clips, one folder per clip.

    datasets/clips/<clip-id>/
      video.mp4      the footage (git-ignored: large, and usually other people's)
      clip.toml      metadata, including the video's SHA-256 (committed)
      labels.json    hand labels, if any (committed; format in sparring.evaluation)

A clip's run (the pipeline's output) goes to runs/<clip-id>/ and is stamped with the hash of
the video it was made from, so labels, video and run can't silently drift apart: replacing or
re-encoding a video breaks frame-accurate labels, and a stale run scores against the wrong
footage. Both cases are errors rather than wrong numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from sparring.evaluation import Labels

CLIP_FILE = "clip.toml"
LABELS_FILE = "labels.json"
RUN_STAMP = "clip.json"
ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
DEFAULT_DATASET = Path("datasets")
DEFAULT_RUNS = Path("runs")


class ClipError(Exception):
    """A clip is missing, malformed, or out of sync with its video or run."""


class ClipMeta(BaseModel):
    """Contents of clip.toml."""

    model_config = ConfigDict(extra="forbid")

    discipline: str
    split: Literal["train", "test"] = "train"  # test clips are never trained on
    recorded: date | None = None
    camera: str = ""  # e.g. "side-on, chest height": lets accuracy be broken down by setup
    source: str = ""  # where the footage came from, and consent to use it
    notes: str = ""
    # Filled in by `sparring clip add`.
    video: str = "video.mp4"
    sha256: str
    fps: float | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None

    @field_validator("discipline")
    @classmethod
    def _known_discipline(cls, v: str) -> str:
        from sparring.disciplines import DISCIPLINES  # lazy: pulls in the detectors

        if v not in DISCIPLINES:
            raise ValueError(f"unknown discipline {v!r}; choose from {sorted(DISCIPLINES)}")
        return v


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def check_id(clip_id: str) -> str:
    if not ID_PATTERN.match(clip_id):
        raise ClipError(
            f"invalid clip id {clip_id!r}: use lowercase letters, digits and single hyphens, "
            "e.g. boxing-homegym-2026-10-04-r2"
        )
    return clip_id


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Clip:
    dir: Path
    meta: ClipMeta

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def video(self) -> Path:
        return self.dir / self.meta.video

    @property
    def labels_path(self) -> Path:
        return self.dir / LABELS_FILE

    @property
    def has_labels(self) -> bool:
        return self.labels_path.exists()

    def labels(self) -> Labels:
        from sparring.evaluation import Labels

        if not self.has_labels:
            raise ClipError(f"clip {self.id} has no {LABELS_FILE}")
        return Labels.load(self.labels_path)

    @classmethod
    def load(cls, clip_dir: Path) -> Clip:
        path = clip_dir / CLIP_FILE
        try:
            with path.open("rb") as f:
                meta = ClipMeta.model_validate(tomllib.load(f))
        except FileNotFoundError:
            raise ClipError(f"{clip_dir} is not a clip (no {CLIP_FILE})") from None
        except (tomllib.TOMLDecodeError, ValueError) as e:
            raise ClipError(f"{path}: {e}") from None
        return cls(clip_dir, meta)

    def save(self) -> None:
        body = tomli_w.dumps(self.meta.model_dump(exclude_none=True))
        (self.dir / CLIP_FILE).write_text(body)

    def verify_video(self) -> None:
        """Raise unless the video exists and is the exact file the metadata (and labels) describe."""
        if not self.video.exists():
            raise ClipError(
                f"clip {self.id}: video missing ({self.video}). Videos are not in git; "
                "copy it into the clip folder."
            )
        if sha256_of(self.video) != self.meta.sha256:
            raise ClipError(
                f"clip {self.id}: {self.video.name} does not match the sha256 in {CLIP_FILE}. "
                "Labels are tied to exact frames, so restore the original file rather than "
                "re-encoding it, or add the new file as a new clip."
            )

    def problems(self) -> list[str]:
        """Everything wrong with this clip (hashes the video, so it can be slow)."""
        out = []
        try:
            self.verify_video()
        except ClipError as e:
            out.append(str(e))
        if self.has_labels:
            try:
                labels = self.labels()
            except (ValueError, KeyError, TypeError) as e:
                out.append(f"clip {self.id}: unreadable {LABELS_FILE}: {e}")
            else:
                from sparring.disciplines import DISCIPLINES

                allowed = set(DISCIPLINES[self.meta.discipline].moves)
                allowed |= {"punch", "kick", "strike"}
                bad = sorted({e.type for e in labels.events} - allowed)
                if bad:
                    out.append(
                        f"clip {self.id}: labels use moves outside {self.meta.discipline}: {bad}"
                    )
        return out

    # ---- runs ---------------------------------------------------------------------------------

    def run_dir(self, runs: Path = DEFAULT_RUNS) -> Path:
        return runs / self.id

    def stamp_run(self, run_dir: Path) -> None:
        """Record which clip and video a run was made from."""
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = {"clip": self.id, "sha256": self.meta.sha256}
        (run_dir / RUN_STAMP).write_text(json.dumps(stamp, indent=2))

    def run_status(self, run_dir: Path) -> Literal["none", "ok", "stale", "unstamped"]:
        if not (run_dir / "events.json").exists():
            return "none"
        stamp = run_dir / RUN_STAMP
        if not stamp.exists():
            return "unstamped"
        return "ok" if json.loads(stamp.read_text())["sha256"] == self.meta.sha256 else "stale"

    def require_run(self, run_dir: Path) -> Path:
        """The clip's run directory, or an error saying how to make or refresh it."""
        status = self.run_status(run_dir)
        if status == "ok":
            return run_dir
        fix = f"run `sparring analyze {self.id}`"
        reason = {
            "none": f"no analysis in {run_dir}",
            "stale": f"{run_dir} was made from a different version of the video",
            "unstamped": f"{run_dir} was not made from this clip",
        }[status]
        raise ClipError(f"clip {self.id}: {reason}; {fix}")


class Dataset:
    def __init__(self, root: Path = DEFAULT_DATASET) -> None:
        self.root = root

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    def clips(self) -> list[Clip]:
        if not self.clips_dir.exists():
            return []
        return [Clip.load(d) for d in sorted(self.clips_dir.iterdir()) if (d / CLIP_FILE).exists()]

    def get(self, clip_id: str) -> Clip:
        clip_dir = self.clips_dir / check_id(clip_id)
        if not (clip_dir / CLIP_FILE).exists():
            known = ", ".join(c.id for c in self.clips()) or "none"
            raise ClipError(f"no clip {clip_id!r} in {self.clips_dir} (clips: {known})")
        return Clip.load(clip_dir)

    def resolve(self, target: str | Path) -> Clip | None:
        """A clip folder or clip id -> the clip; anything else (e.g. a video file) -> None."""
        path = Path(target)
        if (path / CLIP_FILE).exists():
            return Clip.load(path)
        if not path.exists() and ID_PATTERN.match(str(target)):
            return self.get(str(target))
        return None

    def add(
        self,
        video: Path,
        clip_id: str,
        discipline: str,
        *,
        move: bool = False,
        **meta,
    ) -> Clip:
        """Copy (or move) a video into a new clip folder and write its clip.toml."""
        from sparring.io.video import probe

        clip_dir = self.clips_dir / check_id(clip_id)
        if clip_dir.exists():
            raise ClipError(f"clip {clip_id!r} already exists at {clip_dir}")
        if not video.is_file():
            raise ClipError(f"no such video: {video}")
        info = probe(video)  # also proves the file decodes before anything is copied
        name = "video" + video.suffix.lower()
        clip = Clip(
            clip_dir,
            ClipMeta(
                discipline=discipline,
                sha256=sha256_of(video),
                video=name,
                fps=round(info.fps, 3),
                duration_s=round(info.duration_s, 2),
                width=info.width,
                height=info.height,
                **meta,
            ),
        )
        clip_dir.mkdir(parents=True)
        try:
            clip.save()
            (shutil.move if move else shutil.copy2)(video, clip.video)
        except BaseException:  # never leave a half-made clip; the source video is untouched
            shutil.rmtree(clip_dir, ignore_errors=True)
            raise
        return clip
