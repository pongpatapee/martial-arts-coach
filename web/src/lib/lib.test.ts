import { describe, expect, it } from "vitest";
import type { AnalysisEvent, MoveInfo, Report } from "../api/types";
import {
  activeEvents,
  buildHighlights,
  DEFAULT_FILTER,
  filterEvents,
  mergeHighlights,
  nearestEvent,
  nextEventTime,
} from "./events";
import { containRect, decodePoses, parseShape, point } from "./poses";
import { formatDuration, formatTime, frameAt, nextSpeed, timeOfFrame } from "./time";
import { follow, packLanes, toTime, toX, zoom } from "./timeline";

const moves: Record<string, MoveInfo> = {
  jab: { kind: "strike", label: "jab", limb: "fist", side: "lead" },
  cross: { kind: "strike", label: "cross", limb: "fist", side: "rear" },
  slip: { kind: "defense", label: "slip", limb: "none", side: null },
  guard_drop: { kind: "fault", label: "hand held low", limb: "none", side: null },
  combination: { kind: "other", label: "combination", limb: "none", side: null },
};

const ev = (fighter: "A" | "B", type: string, t: number, confidence = 0.8, evidence = {}): AnalysisEvent => ({
  fighter,
  type,
  start_t: t - 0.1,
  peak_t: t,
  end_t: t + 0.1,
  confidence,
  evidence,
});

describe("time", () => {
  it("maps times to frames robustly at boundaries", () => {
    expect(frameAt(1.0, 30)).toBe(30);
    expect(frameAt(0.9999999, 30)).toBe(30); // float error just below the boundary
    expect(frameAt(1.02, 30)).toBe(30);
    expect(frameAt(timeOfFrame(42, 30), 30)).toBe(42);
  });
  it("formats", () => {
    expect(formatTime(75.456)).toBe("1:15.46");
    expect(formatTime(5, 0)).toBe("0:05");
    expect(formatDuration(125)).toBe("2m 05s");
    expect(formatDuration(null)).toBe("–");
  });
  it("steps through speeds and clamps at the ends", () => {
    expect(nextSpeed(1, -1)).toBe(0.75);
    expect(nextSpeed(0.1, -1)).toBe(0.1);
    expect(nextSpeed(2, 1)).toBe(2);
    expect(nextSpeed(1.3, 1)).toBe(1.5); // unknown speed restarts from 1x
  });
});

describe("events", () => {
  const events = [ev("A", "jab", 1), ev("B", "slip", 1.05, 0.5), ev("A", "guard_drop", 3), ev("B", "cross", 5, 0.3)];

  it("filters by fighter, kind, type and confidence", () => {
    expect(filterEvents(events, DEFAULT_FILTER, moves)).toHaveLength(4);
    expect(filterEvents(events, { ...DEFAULT_FILTER, fighters: ["A"] }, moves).map((e) => e.type)).toEqual(["jab", "guard_drop"]);
    expect(filterEvents(events, { ...DEFAULT_FILTER, kinds: ["strike"] }, moves)).toHaveLength(2);
    expect(filterEvents(events, { ...DEFAULT_FILTER, hiddenTypes: ["jab"] }, moves)).toHaveLength(3);
    expect(filterEvents(events, { ...DEFAULT_FILTER, minConfidence: 0.6 }, moves)).toHaveLength(2);
  });
  it("finds active, nearest and next/previous events", () => {
    expect(activeEvents(events, 1.02).map((e) => e.type)).toEqual(["jab", "slip"]);
    expect(activeEvents(events, 1.47).map((e) => e.type)).toEqual(["slip"]); // jab's hold ended at 1.45
    expect(nearestEvent(events, 2.8)?.type).toBe("guard_drop");
    expect(nearestEvent(events, 4.0)).toBeNull();
    expect(nextEventTime(events, 1.0, 1)).toBe(1.05);
    expect(nextEventTime(events, 1.0, -1)).toBeNull();
    expect(nextEventTime(events, 4, -1)).toBe(3);
  });
});

describe("highlights", () => {
  it("merges nearby moments and keeps the strongest title", () => {
    const merged = mergeHighlights([
      { t: 1, start: 0, end: 2, fighter: "A", title: "jab", reasons: ["clean strike"], score: 1 },
      { t: 1.5, start: 0.5, end: 2.5, fighter: "B", title: "slip", reasons: ["defence"], score: 2 },
      { t: 5, start: 4, end: 6, fighter: "A", title: "cross", reasons: ["clean strike"], score: 1 },
    ]);
    expect(merged).toHaveLength(2);
    expect(merged[0]).toMatchObject({ title: "slip", fighter: "B", end: 2.5, reasons: ["clean strike", "defence"] });
  });
  it("builds from combos, strikes, defences and coaching watch times", () => {
    const report = {
      fighters: {
        A: { coaching: { priorities: [{ title: "Carries the lead hand low", kind: "improve", watch: [10], score: 1 }], strengths: [] } },
        B: { coaching: { priorities: [], strengths: [] } },
      },
    } as unknown as Report;
    const hl = buildHighlights(
      [ev("A", "combination", 1, 0.9, { punches: ["jab", "cross"] }), ev("A", "jab", 1.1), ev("B", "slip", 20)],
      report,
      moves,
    );
    expect(hl.map((h) => h.title)).toEqual(["jab – cross", "Carries the lead hand low", "slip"]);
    expect(hl[0]!.start).toBe(0); // clamped at the start of the video
  });
});

describe("timeline", () => {
  const view = { start: 10, end: 20 };
  it("maps between time and pixels", () => {
    expect(toX(15, view, 200)).toBe(100);
    expect(toTime(100, view, 200)).toBe(15);
  });
  it("zooms around an anchor and clamps to the video", () => {
    const z = zoom(view, 0.5, 15, 60);
    expect(z).toEqual({ start: 12.5, end: 17.5 });
    expect(zoom(view, 100, 15, 60)).toEqual({ start: 0, end: 60 });
    expect(zoom({ start: 0, end: 4 }, 0.1, 1, 60).end - zoom({ start: 0, end: 4 }, 0.1, 1, 60).start).toBe(2);
  });
  it("follows the playhead", () => {
    expect(follow(view, 15, 60)).toBe(view);
    expect(follow(view, 25, 60).start).toBeCloseTo(24);
  });
  it("stacks overlapping intervals", () => {
    expect(packLanes([{ start: 0, end: 2 }, { start: 1, end: 3 }, { start: 2.5, end: 4 }])).toEqual([0, 1, 0]);
  });
});

describe("poses", () => {
  it("decodes keypoints and treats -1 as missing", () => {
    const shape = parseShape("2,2,3,2");
    const raw = new Int16Array(2 * 2 * 3 * 2).fill(-1);
    raw.set([100, 200], ((1 * 2 + 1) * 3 + 2) * 2); // frame 1, fighter 1, keypoint 2
    const p = decodePoses(raw.buffer, shape);
    expect(point(p, 1, 1, 2)).toEqual([100, 200]);
    expect(point(p, 0, 0, 0)).toBeNull();
    expect(point(p, 5, 0, 0)).toBeNull();
    expect(() => parseShape("1,2")).toThrow();
  });
  it("letterboxes like object-fit: contain", () => {
    const r = containRect(1000, 1000, 1080, 1920);
    expect([r.x, r.y, r.w, r.h].map((v) => Math.round(v * 100) / 100 + 0)).toEqual([218.75, 0, 562.5, 1000]);
  });
});
