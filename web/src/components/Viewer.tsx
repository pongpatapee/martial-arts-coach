import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { Analysis, Fighter, Options, Skeleton } from "../api/types";
import { useLoad, useShortcuts } from "../hooks/common";
import { usePlayer } from "../hooks/usePlayer";
import { buildHighlights, DEFAULT_FILTER, filterEvents, nextEventTime, type EventFilter } from "../lib/events";
import { CoachingPanel, EventList, Filters, Highlights, ShortcutsHelp } from "./Panels";
import { Player } from "./Player";
import { Timeline } from "./Timeline";

type Tab = "events" | "highlights" | "A" | "B";

interface Props {
  analysis: Analysis;
  options: Options;
  skeleton: Skeleton | null;
  onReanalyzed: () => void;
}

export function Viewer({ analysis, options, skeleton, onReanalyzed }: Props) {
  const { job, video, report } = analysis;
  const moves = options.moves;
  const fps = video.fps;
  const player = usePlayer(fps);
  const poses = useLoad(() => api.poses(job.id), `${job.id}:${job.updated}`);
  const [filter, setFilter] = useState<EventFilter>(DEFAULT_FILTER);
  const [tab, setTab] = useState<Tab>("A");
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [help, setHelp] = useState(false);
  const [reelRate, setReelRate] = useState(0.5);

  const timelineEvents = useMemo(
    () => analysis.events.filter((e) => e.type !== "combination"),
    [analysis.events],
  );
  const visible = useMemo(() => filterEvents(timelineEvents, filter, moves), [timelineEvents, filter, moves]);
  const highlights = useMemo(() => buildHighlights(analysis.events, report, moves), [analysis.events, report, moves]);
  const duration = player.state.duration || video.n_frames / fps;

  const seek = (t: number) => player.seek(t);
  const jumpEvent = (dir: 1 | -1) => {
    const t = nextEventTime(visible, player.state.time, dir);
    if (t !== null) seek(t);
  };
  const playReel = (startAt: number) => {
    player.setRate(reelRate);
    player.playReel(highlights, startAt);
  };

  useShortcuts({
    " ": () => player.toggle(),
    ArrowLeft: () => player.stepFrames(-1),
    ArrowRight: () => player.stepFrames(1),
    "shift+ArrowLeft": () => player.stepSeconds(-0.5),
    "shift+ArrowRight": () => player.stepSeconds(0.5),
    "[": () => jumpEvent(-1),
    "]": () => jumpEvent(1),
    "-": () => player.changeRate(-1),
    "=": () => player.changeRate(1),
    s: () => setShowSkeleton((v) => !v),
    h: () => (player.state.reelIndex !== null ? player.stopReel() : playReel(0)),
    "?": () => setHelp((v) => !v),
    Escape: () => setHelp(false),
  });

  return (
    <div className="viewer">
      <header className="viewer-head">
        <div>
          <h1>{job.name}</h1>
          <div className="muted">
            {options.disciplines.find((d) => d.id === report.discipline)?.title ?? report.discipline} · detection:{" "}
            {report.detection_method} · {Math.round(fps)} fps
          </div>
        </div>
        <Reanalyze jobId={job.id} current={job.options} options={options} onDone={onReanalyzed} />
      </header>
      {report.data_quality.notes.length > 0 && (
        <div className="notice" role="note">
          {report.data_quality.notes.map((n) => (
            <div key={n}>⚠ {n}</div>
          ))}
        </div>
      )}
      {poses.error && <div className="notice error">Skeleton overlay unavailable: {poses.error.message}</div>}

      <div className="viewer-grid">
        <section className="viewer-main">
          <Player
            player={player}
            src={api.videoUrl(job.id)}
            fps={fps}
            sourceSize={{ width: video.width, height: video.height }}
            poses={poses.data}
            skeleton={skeleton}
            showSkeleton={showSkeleton}
            onToggleSkeleton={() => setShowSkeleton((v) => !v)}
            events={visible}
            moves={moves}
            onPrevEvent={() => jumpEvent(-1)}
            onNextEvent={() => jumpEvent(1)}
            onHelp={() => setHelp(true)}
          />
          <Timeline
            events={visible}
            duration={duration}
            time={player.state.time}
            moves={moves}
            onSeek={seek}
            highlights={highlights}
          />
        </section>

        <aside className="viewer-side">
          <nav className="tabs" role="tablist">
            {(
              [
                ["A", "Fighter A"],
                ["B", "Fighter B"],
                ["highlights", `Highlights (${highlights.length})`],
                ["events", `Events (${visible.length})`],
              ] as [Tab, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                className={`tab ${tab === id ? "on" : ""} ${id === "A" || id === "B" ? `fighter-${id}` : ""}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="tab-body">
            {tab === "events" && (
              <>
                <Filters filter={filter} onChange={setFilter} events={timelineEvents} moves={moves} />
                <EventList events={visible} time={player.state.time} moves={moves} onSeek={seek} />
              </>
            )}
            {tab === "highlights" && (
              <Highlights
                items={highlights}
                current={player.state.reelIndex}
                onSeek={seek}
                onPlay={playReel}
                onStop={player.stopReel}
                playing={player.state.reelIndex !== null}
                reelRate={reelRate}
                onReelRate={setReelRate}
              />
            )}
            {(tab === "A" || tab === "B") && report.fighters[tab as Fighter] && (
              <>
                <p className="muted small">Fighter {tab} is {report.fighters[tab as Fighter].identity}.</p>
                <CoachingPanel
                  fighter={tab as Fighter}
                  report={report.fighters[tab as Fighter]}
                  moves={moves}
                  onSeek={seek}
                />
              </>
            )}
          </div>
        </aside>
      </div>
      {help && <ShortcutsHelp onClose={() => setHelp(false)} />}
    </div>
  );
}

function Reanalyze({
  jobId,
  current,
  options,
  onDone,
}: {
  jobId: string;
  current: { discipline: string; method: string };
  options: Options;
  onDone: () => void;
}) {
  const [discipline, setDiscipline] = useState(current.discipline);
  const [method, setMethod] = useState(current.method);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const changed = discipline !== current.discipline || method !== current.method;
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.reanalyze(jobId, { discipline, method });
      onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="reanalyze">
      <select value={discipline} onChange={(e) => setDiscipline(e.target.value)} aria-label="Discipline">
        {options.disciplines.map((d) => (
          <option key={d.id} value={d.id}>
            {d.title}
          </option>
        ))}
      </select>
      <select value={method} onChange={(e) => setMethod(e.target.value)} aria-label="Detection method">
        {options.methods.map((m) => (
          <option key={m.id} value={m.id} title={m.help}>
            {m.id}
          </option>
        ))}
      </select>
      <button className="btn" onClick={run} disabled={!changed || busy} title="Re-runs detection and coaching only (seconds)">
        {busy ? "Re-analysing…" : "Re-analyse"}
      </button>
      {error && <span className="error-text">{error}</span>}
    </div>
  );
}
