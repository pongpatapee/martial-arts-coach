import type { AnalysisEvent, Feedback, Fighter, Kind, MoveInfo, Report } from "../api/types";

export const KINDS: Kind[] = ["strike", "defense", "fault", "other"];
export const KIND_LABEL: Record<Kind, string> = {
  strike: "Strikes",
  defense: "Defence",
  fault: "Faults",
  other: "Other",
};

export type Moves = Record<string, MoveInfo>;

export function kindOf(type: string, moves: Moves): Kind {
  return moves[type]?.kind ?? "other";
}

export function labelOf(type: string, moves: Moves): string {
  return moves[type]?.label ?? type.replaceAll("_", " ");
}

export interface EventFilter {
  fighters: Fighter[];
  kinds: Kind[];
  /** Event types to hide (empty = show all of the selected kinds). */
  hiddenTypes: string[];
  minConfidence: number;
}

export const DEFAULT_FILTER: EventFilter = {
  fighters: ["A", "B"],
  kinds: ["strike", "defense", "fault"],
  hiddenTypes: [],
  minConfidence: 0,
};

export function filterEvents(events: AnalysisEvent[], f: EventFilter, moves: Moves): AnalysisEvent[] {
  return events.filter(
    (e) =>
      f.fighters.includes(e.fighter) &&
      f.kinds.includes(kindOf(e.type, moves)) &&
      !f.hiddenTypes.includes(e.type) &&
      e.confidence >= f.minConfidence,
  );
}

/** Events to show as "happening now": those spanning t, plus a short hold after they end. */
export function activeEvents(events: AnalysisEvent[], t: number, hold = 0.35): AnalysisEvent[] {
  return events.filter((e) => e.start_t - 0.05 <= t && t <= e.end_t + hold);
}

/** The event whose peak is nearest t (for highlighting the list row). */
export function nearestEvent(events: AnalysisEvent[], t: number, within = 0.6): AnalysisEvent | null {
  let best: AnalysisEvent | null = null;
  let bestD = within;
  for (const e of events) {
    const d = Math.abs(e.peak_t - t);
    if (d <= bestD) {
      best = e;
      bestD = d;
    }
  }
  return best;
}

export function nextEventTime(events: AnalysisEvent[], t: number, direction: 1 | -1): number | null {
  const eps = 0.02; // under a frame at 60 fps, so near-simultaneous events stay reachable
  const times = events.map((e) => e.peak_t).sort((a, b) => a - b);
  if (direction === 1) return times.find((x) => x > t + eps) ?? null;
  for (let i = times.length - 1; i >= 0; i--) {
    const x = times[i]!;
    if (x < t - eps) return x;
  }
  return null;
}

/** Counts per type, for the filter panel. */
export function typeCounts(events: AnalysisEvent[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const e of events) out[e.type] = (out[e.type] ?? 0) + 1;
  return out;
}

// ---- highlights --------------------------------------------------------------------------------

export interface Highlight {
  t: number;
  start: number;
  end: number;
  fighter: Fighter | null;
  title: string;
  reasons: string[];
  score: number;
}

const LEAD_IN = 1.2;
const LEAD_OUT = 1.0;
const MERGE_S = 1.0;

/** Key moments: combinations, the most confident strikes, defences against strikes, and every
 * coaching point's watch timestamps. Moments within a second of each other are merged. */
export function buildHighlights(
  events: AnalysisEvent[],
  report: Report | null,
  moves: Moves,
  maxStrikesPerFighter = 6,
): Highlight[] {
  const raw: Highlight[] = [];
  const add = (t: number, fighter: Fighter | null, title: string, reason: string, score: number) =>
    raw.push({ t, start: t - LEAD_IN, end: t + LEAD_OUT, fighter, title, reasons: [reason], score });

  for (const e of events) {
    if (e.type === "combination") {
      const punches = (e.evidence.punches as string[] | undefined) ?? [];
      const title = punches.length ? punches.map((p) => labelOf(p, moves)).join(" – ") : "combination";
      add(e.peak_t, e.fighter, title, `${punches.length || ""}-strike combination`.trim(), 2 + punches.length * 0.3);
    }
  }
  for (const f of ["A", "B"] as Fighter[]) {
    events
      .filter((e) => e.fighter === f && kindOf(e.type, moves) === "strike")
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, maxStrikesPerFighter)
      .forEach((e) => add(e.peak_t, f, labelOf(e.type, moves), "clean strike", 1 + e.confidence));
  }
  for (const e of events) {
    if (["slip", "duck", "block", "check"].includes(e.type)) {
      add(e.peak_t, e.fighter, labelOf(e.type, moves), "defence", 1.2 + e.confidence * 0.5);
    }
  }
  if (report) {
    for (const f of ["A", "B"] as Fighter[]) {
      const c = report.fighters[f]?.coaching;
      if (!c) continue;
      const items: Feedback[] = [...c.priorities, ...c.strengths];
      for (const fb of items) {
        for (const t of fb.watch) {
          add(t, f, fb.title, fb.kind === "improve" ? "to improve" : "strength", 3 + fb.score);
        }
      }
    }
  }
  return mergeHighlights(raw);
}

export function mergeHighlights(items: Highlight[]): Highlight[] {
  const sorted = [...items].sort((a, b) => a.t - b.t);
  const out: Highlight[] = [];
  for (const h of sorted) {
    const last = out[out.length - 1];
    if (last && h.t - last.t <= MERGE_S) {
      if (h.score > last.score) {
        last.title = h.title;
        last.fighter = h.fighter;
      }
      last.score = Math.max(last.score, h.score);
      last.end = Math.max(last.end, h.end);
      for (const r of h.reasons) if (!last.reasons.includes(r)) last.reasons.push(r);
      continue;
    }
    out.push({ ...h, reasons: [...h.reasons] });
  }
  return out.map((h) => ({ ...h, start: Math.max(0, h.start) }));
}
