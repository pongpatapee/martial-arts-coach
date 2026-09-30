import { useEffect, useMemo, useRef, useState } from "react";
import type { AnalysisEvent, Fighter } from "../api/types";
import { useShortcuts } from "../hooks/common";
import { kindOf, labelOf, type Moves } from "../lib/events";
import { formatTime } from "../lib/time";
import { follow, fullView, packLanes, pan, tickStep, toTime, toX, zoom, type View } from "../lib/timeline";

const ROW_H = 46;
const AXIS_H = 20;
const FIGHTERS: Fighter[] = ["A", "B"];
const MIN_W = 3;

interface Props {
  events: AnalysisEvent[];
  duration: number;
  time: number;
  moves: Moves;
  onSeek: (t: number) => void;
  highlights?: { start: number; end: number }[];
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  e: AnalysisEvent;
}

export function Timeline({ events, duration, time, moves, onSeek, highlights = [] }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(800);
  const [view, setView] = useState<View>(() => fullView(duration));
  const [hover, setHover] = useState<{ box: Box; x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; view: View; moved: boolean } | null>(null);
  const boxes = useRef<Box[]>([]);

  useEffect(() => setView(fullView(duration)), [duration]);
  useEffect(() => setView((v) => follow(v, time, duration)), [time, duration]);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Sub-lanes so overlapping events of one fighter stack instead of hiding each other.
  const layout = useMemo(() => {
    const out = new Map<AnalysisEvent, { lane: number; lanes: number }>();
    for (const f of FIGHTERS) {
      const mine = events.filter((e) => e.fighter === f);
      const lanes = packLanes(mine.map((e) => ({ start: e.start_t, end: Math.max(e.end_t, e.start_t + 0.15) })));
      const n = Math.max(1, ...lanes.map((l) => l + 1));
      mine.forEach((e, i) => out.set(e, { lane: lanes[i]!, lanes: n }));
    }
    return out;
  }, [events]);

  const height = AXIS_H + ROW_H * 2;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const c = (n: string) => css.getPropertyValue(n).trim();
    ctx.clearRect(0, 0, width, height);

    // Rows.
    FIGHTERS.forEach((f, i) => {
      ctx.fillStyle = i % 2 ? c("--surface-2") : c("--surface-3");
      ctx.fillRect(0, AXIS_H + i * ROW_H, width, ROW_H);
      ctx.fillStyle = c(f === "A" ? "--fighter-a" : "--fighter-b");
      ctx.fillRect(0, AXIS_H + i * ROW_H, 3, ROW_H);
    });
    // Highlight bands.
    ctx.fillStyle = c("--highlight-band");
    for (const h of highlights) {
      const x0 = toX(h.start, view, width);
      const x1 = toX(h.end, view, width);
      if (x1 < 0 || x0 > width) continue;
      ctx.fillRect(x0, AXIS_H, x1 - x0, ROW_H * 2);
    }
    // Axis ticks.
    const step = tickStep(view.end - view.start, width);
    ctx.fillStyle = c("--text-muted");
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    for (let t = Math.ceil(view.start / step) * step; t <= view.end; t += step) {
      const x = toX(t, view, width);
      ctx.fillRect(x, AXIS_H - 5, 1, 5);
      ctx.fillText(formatTime(t, step < 1 ? 1 : 0), x + 3, AXIS_H / 2);
    }
    // Events.
    const out: Box[] = [];
    for (const e of events) {
      const l = layout.get(e);
      if (!l) continue;
      const row = FIGHTERS.indexOf(e.fighter);
      const x0 = toX(e.start_t, view, width);
      const x1 = toX(e.end_t, view, width);
      if (x1 < -2 || x0 > width + 2) continue;
      const laneH = (ROW_H - 8) / l.lanes;
      const box = {
        x: x0,
        y: AXIS_H + row * ROW_H + 4 + l.lane * laneH,
        w: Math.max(MIN_W, x1 - x0),
        h: Math.max(3, laneH - 2),
        e,
      };
      ctx.globalAlpha = 0.35 + 0.65 * e.confidence;
      ctx.fillStyle = c(`--kind-${kindOf(e.type, moves)}`);
      ctx.fillRect(box.x, box.y, box.w, box.h);
      out.push(box);
    }
    ctx.globalAlpha = 1;
    boxes.current = out;
    // Playhead.
    const px = toX(time, view, width);
    ctx.fillStyle = c("--playhead");
    ctx.fillRect(px - 1, 0, 2, height);
  }, [events, layout, view, width, height, time, moves, highlights]);

  const localX = (e: React.PointerEvent | React.WheelEvent) =>
    e.clientX - (canvasRef.current?.getBoundingClientRect().left ?? 0);
  const localY = (e: React.PointerEvent) => e.clientY - (canvasRef.current?.getBoundingClientRect().top ?? 0);

  const hitTest = (x: number, y: number): Box | null => {
    for (let i = boxes.current.length - 1; i >= 0; i--) {
      const b = boxes.current[i]!;
      if (x >= b.x - 2 && x <= b.x + b.w + 2 && y >= b.y && y <= b.y + b.h) return b;
    }
    return null;
  };

  const span = view.end - view.start;
  const zoomBy = (factor: number) => setView((v) => zoom(v, factor, time, duration));
  useShortcuts({ z: () => zoomBy(0.5), x: () => zoomBy(2) });

  return (
    <div className="timeline">
      <div className="timeline-bar">
        <span className="muted">Timeline</span>
        <span className="legend">
          <i className="kind-strike" /> strike <i className="kind-defense" /> defence <i className="kind-fault" /> fault
        </span>
        <span className="grow" />
        <button className="btn small" onClick={() => zoomBy(0.5)} title="Zoom in (Z)">
          +
        </button>
        <button className="btn small" onClick={() => zoomBy(2)} title="Zoom out (X)">
          −
        </button>
        <button className="btn small" onClick={() => setView(fullView(duration))} disabled={span >= duration - 0.01}>
          Fit
        </button>
      </div>
      <div className="timeline-canvas" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height }}
          onWheel={(e) => {
            if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
              setView((v) => pan(v, (e.deltaX / width) * span, duration));
            } else {
              setView((v) => zoom(v, Math.exp(e.deltaY * 0.002), toTime(localX(e), v, width), duration));
            }
          }}
          onPointerDown={(e) => {
            (e.target as Element).setPointerCapture(e.pointerId);
            drag.current = { x: e.clientX, view, moved: false };
          }}
          onPointerMove={(e) => {
            const d = drag.current;
            if (d) {
              const dx = e.clientX - d.x;
              if (Math.abs(dx) > 3) d.moved = true;
              if (d.moved) setView(pan(d.view, (-dx / width) * (d.view.end - d.view.start), duration));
              return;
            }
            const b = hitTest(localX(e), localY(e));
            setHover(b ? { box: b, x: localX(e), y: localY(e) } : null);
          }}
          onPointerUp={(e) => {
            const d = drag.current;
            drag.current = null;
            if (d && !d.moved) {
              const b = hitTest(localX(e), localY(e));
              onSeek(b ? b.e.peak_t : toTime(localX(e), view, width));
            }
          }}
          onPointerLeave={() => setHover(null)}
          aria-label="Event timeline: click to seek, scroll to zoom, drag to pan"
          role="img"
        />
        <div className="timeline-labels" aria-hidden>
          <span style={{ top: AXIS_H + 4 }}>A</span>
          <span style={{ top: AXIS_H + ROW_H + 4 }}>B</span>
        </div>
        {hover && (
          <div className="tooltip" style={{ left: Math.min(hover.x + 12, width - 200), top: hover.y + 12 }}>
            <b>
              {hover.box.e.fighter} · {labelOf(hover.box.e.type, moves)}
            </b>
            <div className="muted">
              {formatTime(hover.box.e.peak_t)} · confidence {Math.round(hover.box.e.confidence * 100)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
