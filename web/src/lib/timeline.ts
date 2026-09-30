import { clamp } from "./time";

/** A visible time window on the timeline. */
export interface View {
  start: number;
  end: number;
}

export const MIN_SPAN_S = 2;

export function fullView(duration: number): View {
  return { start: 0, end: Math.max(duration, MIN_SPAN_S) };
}

export function toX(t: number, view: View, width: number): number {
  return ((t - view.start) / (view.end - view.start)) * width;
}

export function toTime(x: number, view: View, width: number): number {
  return view.start + (x / Math.max(width, 1)) * (view.end - view.start);
}

/** Zoom by `factor` (<1 zooms in) keeping `anchor` (a time) at the same screen position. */
export function zoom(view: View, factor: number, anchor: number, duration: number): View {
  const span = view.end - view.start;
  const newSpan = clamp(span * factor, MIN_SPAN_S, Math.max(duration, MIN_SPAN_S));
  const ratio = (anchor - view.start) / span;
  let start = anchor - ratio * newSpan;
  start = clamp(start, 0, Math.max(0, duration - newSpan));
  return { start, end: start + newSpan };
}

export function pan(view: View, deltaS: number, duration: number): View {
  const span = view.end - view.start;
  const start = clamp(view.start + deltaS, 0, Math.max(0, duration - span));
  return { start, end: start + span };
}

/** Keep t on screen: page the view when the playhead leaves it. */
export function follow(view: View, t: number, duration: number): View {
  const span = view.end - view.start;
  if (t >= view.start && t <= view.end) return view;
  const start = clamp(t - span * 0.1, 0, Math.max(0, duration - span));
  return { start, end: start + span };
}

/** Greedy interval packing: returns a sub-lane index per interval so overlaps stack. */
export function packLanes(intervals: { start: number; end: number }[], gap = 0.02): number[] {
  const order = intervals.map((_, i) => i).sort((a, b) => intervals[a]!.start - intervals[b]!.start);
  const laneEnds: number[] = [];
  const lanes = new Array<number>(intervals.length).fill(0);
  for (const i of order) {
    const iv = intervals[i]!;
    let lane = laneEnds.findIndex((end) => end + gap <= iv.start);
    if (lane === -1) {
      lane = laneEnds.length;
      laneEnds.push(iv.end);
    } else {
      laneEnds[lane] = iv.end;
    }
    lanes[i] = lane;
  }
  return lanes;
}

/** "Nice" tick spacing for a span in seconds. */
export function tickStep(span: number, width: number): number {
  const target = span / Math.max(2, width / 90);
  const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  return steps.find((s) => s >= target) ?? 600;
}
