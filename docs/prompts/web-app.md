# Build prompt: Sparring analysis web app

Build a web app on top of the existing `sparring` analysis engine (this repo). Users upload a
sparring video, choose the analysis options the engine supports, watch it being processed, and
then review the result in an interactive player. Prefer quality, robustness and long-term
maintainability over development speed. Reuse the engine as-is: the web app is a client of it,
not a fork of its logic.

## Users and scope

- Several people (a fighter, their coach, teammates) use one deployment. For now it runs on one
  machine (local or a single server with the models); no accounts. Jobs are identified by
  unguessable IDs. Design it so adding auth later is a middleware change, not a rewrite.
- A job takes minutes (≈6 min per minute of video on an M5), so processing is asynchronous.
  Results must survive a server restart.

## Upload and options

- Drag-and-drop (or pick) a video: mp4/mov/m4v/webm, with a size limit and a clear error for
  anything else.
- Options, populated from the engine (not hard-coded in the UI):
  - discipline: boxing / kickboxing / muay_thai,
  - strike-detection method: ensemble (default) / rules / temporal / learned / hybrid, with
    one-line explanations,
  - optional fighter override (two track ids) for when the wrong people were picked.
- After a job is done, changing the discipline or method re-runs only the cheap stages
  (events -> report, seconds). Perception is never recomputed for an option change.

## Processing

- One worker runs jobs sequentially in a separate process (`sparring analyze`), so a crash or
  out-of-memory in inference cannot take the web server down. Queued jobs wait their turn.
- The pipeline reports progress (stage, fraction within the stage) to a file in the run
  directory; the API exposes it and the UI shows a stage-by-stage progress view.
- Transcode the upload to an H.264/AAC playback file at the source frame rate (phones often
  record HEVC, which not every browser plays) with a keyframe every second for fast seeking.
- Failures show the error and the last log lines; the job can be retried or deleted.

## Library

- A list of past analyses: thumbnail, name, discipline, duration, status, created time. Open,
  retry and delete.

## Analysis viewer (the main screen)

- A video player with precise controls:
  - play/pause, speed 0.1x–2x, frame step back/forward, ±0.5 s jumps, jump to
    previous/next event;
  - a current time + frame number readout;
  - keyboard shortcuts for all of the above, shown in a help popover.
- Skeleton overlay drawn over the video (toggleable), in each fighter's colour, synced per
  frame. Use `requestVideoFrameCallback`, not `timeupdate`.
- An event timeline under the video, with one lane per fighter. Events are coloured by kind
  (strike / defence / fault / other). Hover shows details, clicking seeks, and there is a
  playhead. Zoom in on a section for dense exchanges.
- Filters: per fighter, per event kind, per event type, and a minimum confidence.
- An event list synced to the playhead: the current event is highlighted and a row click seeks.
- Highlights: auto-generated key moments (combinations, the most confident strikes, defences
  against strikes, and every coaching point's watch timestamps). Include a "play highlights"
  mode that plays each moment with a short lead-in/out, optionally slowed down.
- Coaching panel per fighter, from `report.json`:
  - summary, strengths and priorities with observation / why / how / drills / confidence tag;
  - watch timestamps as clickable chips that seek;
  - the metrics table and the next-session plan.
- Show data-quality notes (low fps, camera angle) prominently.
- Works on a laptop screen and usably on a tablet. Light and dark themes.

## Architecture

- Backend: FastAPI app in `src/sparring/web/`. Contents:
  - an upload endpoint (streamed to disk, size-limited);
  - a job store on disk (`data/jobs/<id>/job.json` + the run directory);
  - the worker; JSON endpoints for options/jobs/analysis/poses;
  - video served with HTTP range support;
  - the built frontend served as static files. CLI: `sparring serve`.
- Poses served compactly (quantised, per fighter per frame) for the overlay.
- Frontend: React + TypeScript + Vite in `web/`, built into the Python package's static
  directory. No UI kit beyond what's needed; hand-written CSS with design tokens. Keep
  data-fetching and player state in small, tested modules (hooks + pure functions).
- Tests: pytest for the API (with a fake pipeline so tests need no models or video), vitest for
  the frontend's pure logic (time/frame math, filtering, highlights, timeline layout), and a
  type-check + production build in CI-style commands.

## Deliverables

1. The working app, started with one command (`sparring serve`).
2. Tests passing.
3. README section: how to run (dev with hot reload, and prod), how to deploy, the API summary.
4. Honest notes on what was verified and what wasn't.

Commit in logical steps on a `web-app` branch.
