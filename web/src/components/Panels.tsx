import { useEffect, useMemo, useRef } from "react";
import type { AnalysisEvent, Feedback, Fighter, FighterReport, Kind } from "../api/types";
import {
  KIND_LABEL,
  KINDS,
  kindOf,
  labelOf,
  nearestEvent,
  typeCounts,
  type EventFilter,
  type Highlight,
  type Moves,
} from "../lib/events";
import { formatTime } from "../lib/time";

// ---- filters -----------------------------------------------------------------------------------

export function Filters({
  filter,
  onChange,
  events,
  moves,
}: {
  filter: EventFilter;
  onChange: (f: EventFilter) => void;
  events: AnalysisEvent[];
  moves: Moves;
}) {
  const counts = useMemo(() => typeCounts(events), [events]);
  const toggle = <T,>(list: T[], x: T) => (list.includes(x) ? list.filter((y) => y !== x) : [...list, x]);
  const types = Object.keys(counts)
    .filter((t) => filter.kinds.includes(kindOf(t, moves)))
    .sort((a, b) => (counts[b] ?? 0) - (counts[a] ?? 0));
  return (
    <div className="filters">
      <div className="chip-row">
        {(["A", "B"] as Fighter[]).map((f) => (
          <button
            key={f}
            className={`chip fighter-${f} ${filter.fighters.includes(f) ? "on" : ""}`}
            onClick={() => onChange({ ...filter, fighters: toggle(filter.fighters, f) })}
          >
            Fighter {f}
          </button>
        ))}
        {KINDS.map((k: Kind) => (
          <button
            key={k}
            className={`chip kind-${k} ${filter.kinds.includes(k) ? "on" : ""}`}
            onClick={() => onChange({ ...filter, kinds: toggle(filter.kinds, k) })}
          >
            {KIND_LABEL[k]}
          </button>
        ))}
      </div>
      {types.length > 0 && (
        <div className="chip-row small">
          {types.map((t) => (
            <button
              key={t}
              className={`chip ${filter.hiddenTypes.includes(t) ? "" : "on"}`}
              onClick={() => onChange({ ...filter, hiddenTypes: toggle(filter.hiddenTypes, t) })}
              title="Show / hide this move"
            >
              {labelOf(t, moves)} <span className="count">{counts[t]}</span>
            </button>
          ))}
        </div>
      )}
      <label className="range">
        <span>Min confidence {Math.round(filter.minConfidence * 100)}%</span>
        <input
          type="range"
          min={0}
          max={0.95}
          step={0.05}
          value={filter.minConfidence}
          onChange={(e) => onChange({ ...filter, minConfidence: Number(e.target.value) })}
        />
      </label>
    </div>
  );
}

// ---- event list --------------------------------------------------------------------------------

export function EventList({
  events,
  time,
  moves,
  onSeek,
}: {
  events: AnalysisEvent[];
  time: number;
  moves: Moves;
  onSeek: (t: number) => void;
}) {
  const current = nearestEvent(events, time, 0.4);
  const rowRef = useRef<HTMLTableRowElement | null>(null);
  useEffect(() => rowRef.current?.scrollIntoView({ block: "nearest" }), [current]);
  if (!events.length) return <p className="empty">No events match the filters.</p>;
  return (
    <table className="event-list">
      <tbody>
        {events.map((e, i) => {
          const isCurrent = e === current;
          return (
            <tr
              key={`${e.fighter}-${e.type}-${e.peak_t}-${i}`}
              ref={isCurrent ? rowRef : undefined}
              className={isCurrent ? "current" : ""}
              onClick={() => onSeek(e.peak_t)}
            >
              <td className="t">{formatTime(e.peak_t)}</td>
              <td>
                <span className={`dot fighter-${e.fighter}`} />
                {e.fighter}
              </td>
              <td>
                <span className={`kind-tag kind-${kindOf(e.type, moves)}`}>{labelOf(e.type, moves)}</span>
              </td>
              <td className="conf" title="confidence">
                {Math.round(e.confidence * 100)}%
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ---- highlights --------------------------------------------------------------------------------

export function Highlights({
  items,
  current,
  onSeek,
  onPlay,
  onStop,
  playing,
  reelRate,
  onReelRate,
}: {
  items: Highlight[];
  current: number | null;
  onSeek: (t: number) => void;
  onPlay: (startAt: number) => void;
  onStop: () => void;
  playing: boolean;
  reelRate: number;
  onReelRate: (r: number) => void;
}) {
  return (
    <div className="highlights">
      <div className="panel-actions">
        {playing ? (
          <button className="btn primary" onClick={onStop}>
            ■ Stop highlights
          </button>
        ) : (
          <button className="btn primary" onClick={() => onPlay(0)} disabled={!items.length}>
            ▶ Play highlights ({items.length})
          </button>
        )}
        <label className="speed">
          <span>at</span>
          <select value={reelRate} onChange={(e) => onReelRate(Number(e.target.value))}>
            {[0.25, 0.5, 0.75, 1].map((r) => (
              <option key={r} value={r}>
                {r}x
              </option>
            ))}
          </select>
        </label>
      </div>
      <ol className="highlight-list">
        {items.map((h, i) => (
          <li key={`${h.t}-${i}`} className={current === i ? "current" : ""}>
            <button className="linkish" onClick={() => onSeek(h.t)}>
              <span className="t">{formatTime(h.t, 1)}</span>
              {h.fighter && <span className={`dot fighter-${h.fighter}`} />}
              <span className="title">{h.title}</span>
            </button>
            <span className="reasons">{h.reasons.join(" · ")}</span>
            <button className="btn small" onClick={() => onPlay(i)} title="Play from here">
              ▶
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}

// ---- coaching ----------------------------------------------------------------------------------

const pct = (v: unknown) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "–");
const num = (v: unknown) => (typeof v === "number" ? String(v) : "–");

function FeedbackCard({ fb, onSeek }: { fb: Feedback; onSeek: (t: number) => void }) {
  const confidence = fb.reliability.split(":")[0];
  return (
    <article className={`feedback ${fb.kind}`}>
      <header>
        <h4>{fb.title}</h4>
        <span className={`confidence c-${confidence}`} title={fb.reliability}>
          {confidence} confidence
        </span>
      </header>
      <p>{fb.observation}</p>
      {fb.why && (
        <p className="muted">
          <b>Why it matters:</b> {fb.why}
        </p>
      )}
      {fb.fix && (
        <p>
          <b>How:</b> {fb.fix}
        </p>
      )}
      {fb.drills.length > 0 && (
        <ul className="drills">
          {fb.drills.map((d) => (
            <li key={d.name}>
              <b>{d.name}</b> <span className="muted">({d.dose})</span> – {d.how}
            </li>
          ))}
        </ul>
      )}
      {fb.watch.length > 0 && (
        <div className="watch">
          <span className="muted">Watch:</span>
          {fb.watch.map((t) => (
            <button key={t} className="chip on" onClick={() => onSeek(t)}>
              {formatTime(t, 1)}
            </button>
          ))}
        </div>
      )}
    </article>
  );
}

export function CoachingPanel({
  fighter,
  report,
  moves,
  onSeek,
}: {
  fighter: Fighter;
  report: FighterReport;
  moves: Moves;
  onSeek: (t: number) => void;
}) {
  const { coaching: c, metrics: m } = report;
  const off = m.offense;
  const dfn = m.defense;
  const grd = m.guard;
  const mix = Object.entries(off.by_type)
    .map(([k, n]) => `${labelOf(k, moves)} ${n}`)
    .join(", ");
  return (
    <div className="coaching">
      <p className={`summary fighter-${fighter}`}>{c.summary}</p>
      <h3>Priorities</h3>
      {c.priorities.length ? (
        c.priorities.map((fb) => <FeedbackCard key={fb.id} fb={fb} onSeek={onSeek} />)
      ) : (
        <p className="empty">Nothing stood out.</p>
      )}
      <h3>Strengths</h3>
      {c.strengths.length ? (
        c.strengths.map((fb) => <FeedbackCard key={fb.id} fb={fb} onSeek={onSeek} />)
      ) : (
        <p className="empty">Nothing stood out.</p>
      )}
      {c.session_plan.length > 0 && (
        <>
          <h3>Next session</h3>
          <ul className="drills plan">
            {c.session_plan.map((d) => (
              <li key={d.name}>
                <b>{d.name}</b> <span className="muted">({d.dose})</span> – {d.how}
                {d.for && <div className="muted small">for: {d.for}</div>}
              </li>
            ))}
          </ul>
        </>
      )}
      <h3>Numbers</h3>
      <table className="metrics">
        <tbody>
          <tr>
            <th>Strikes</th>
            <td>
              {num(off.strikes)} ({num(off.strikes_per_min)}/min){mix && <div className="muted small">{mix}</div>}
            </td>
          </tr>
          <tr>
            <th>In combinations</th>
            <td>{pct(off.share_in_combinations)}</td>
          </tr>
          <tr>
            <th>Return to guard</th>
            <td>{pct(off.return_to_guard_rate)}</td>
          </tr>
          <tr>
            <th>Counters</th>
            <td>{pct(off.counter_rate)} of incoming strikes</td>
          </tr>
          <tr>
            <th>Defence</th>
            <td>
              {pct(dfn.defense_rate)} of {num(dfn.incoming_in_range)} incoming
            </td>
          </tr>
          <tr>
            <th>Hand held low</th>
            <td>
              lead {pct(grd.low_share_by_hand.lead)}, rear {pct(grd.low_share_by_hand.rear)} of in-range time
            </td>
          </tr>
          {typeof dfn.kicks_incoming === "number" && dfn.kicks_incoming > 0 && (
            <tr>
              <th>Kick defence</th>
              <td>
                {pct(dfn.kick_defense_rate)} of {num(dfn.kicks_incoming)} (checked {pct(dfn.check_rate)})
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

// ---- shortcuts ---------------------------------------------------------------------------------

export const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"],
  ["← / →", "Previous / next frame"],
  ["Shift + ← / →", "Back / forward 0.5 s"],
  ["[ / ]", "Previous / next event"],
  ["- / =", "Slower / faster"],
  ["S", "Toggle skeleton"],
  ["H", "Play / stop highlights"],
  ["Z / X", "Zoom timeline in / out"],
  ["?", "Show this help"],
];

export function ShortcutsHelp({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="Keyboard shortcuts" onClick={(e) => e.stopPropagation()}>
        <h3>Keyboard shortcuts</h3>
        <dl className="shortcuts">
          {SHORTCUTS.map(([k, v]) => (
            <div key={k}>
              <dt>
                <kbd>{k}</kbd>
              </dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
        <button className="btn" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
