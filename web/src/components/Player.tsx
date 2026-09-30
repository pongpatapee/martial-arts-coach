import { useEffect, useRef } from "react";
import type { AnalysisEvent, Skeleton } from "../api/types";
import type { Player as PlayerApi } from "../hooks/usePlayer";
import { activeEvents, kindOf, labelOf, type Moves } from "../lib/events";
import { containRect, point, type Poses } from "../lib/poses";
import { frameAt, formatTime, SPEEDS } from "../lib/time";

const FIGHTER_COLOR = { A: "var(--fighter-a)", B: "var(--fighter-b)" } as const;

interface Props {
  player: PlayerApi;
  src: string;
  fps: number;
  sourceSize: { width: number; height: number };
  poses: Poses | null;
  skeleton: Skeleton | null;
  showSkeleton: boolean;
  onToggleSkeleton: () => void;
  events: AnalysisEvent[];
  moves: Moves;
  onPrevEvent: () => void;
  onNextEvent: () => void;
  onHelp: () => void;
}

export function Player(p: Props) {
  const { player, fps } = p;
  const { state } = player;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  // Skeleton overlay: redrawn for every presented frame, and on resize/toggle.
  useEffect(() => {
    const canvas = canvasRef.current;
    const box = boxRef.current;
    const video = player.videoRef.current;
    if (!canvas || !box || !video) return;
    const styles = getComputedStyle(document.documentElement);
    const colors = [styles.getPropertyValue("--fighter-a").trim(), styles.getPropertyValue("--fighter-b").trim()];

    const draw = (t: number) => {
      const dpr = window.devicePixelRatio || 1;
      const w = box.clientWidth;
      const h = box.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!p.showSkeleton || !p.poses || !p.skeleton) return;
      const rect = containRect(w, h, video.videoWidth || p.sourceSize.width, video.videoHeight || p.sourceSize.height);
      const sx = rect.w / p.sourceSize.width;
      const sy = rect.h / p.sourceSize.height;
      const frame = frameAt(t, fps);
      ctx.lineCap = "round";
      for (let f = 0; f < p.poses.fighters; f++) {
        ctx.strokeStyle = colors[f] ?? "#fff";
        ctx.fillStyle = colors[f] ?? "#fff";
        ctx.lineWidth = Math.max(2, rect.h / 320);
        ctx.globalAlpha = 0.9;
        for (const [a, b] of p.skeleton.bones) {
          const pa = point(p.poses, frame, f, a);
          const pb = point(p.poses, frame, f, b);
          if (!pa || !pb) continue;
          ctx.beginPath();
          ctx.moveTo(rect.x + pa[0] * sx, rect.y + pa[1] * sy);
          ctx.lineTo(rect.x + pb[0] * sx, rect.y + pb[1] * sy);
          ctx.stroke();
        }
        for (let k = 0; k < p.poses.keypoints; k++) {
          const q = point(p.poses, frame, f, k);
          if (!q) continue;
          ctx.beginPath();
          ctx.arc(rect.x + q[0] * sx, rect.y + q[1] * sy, Math.max(2, rect.h / 260), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };
    draw(video.currentTime);
    const off = player.onFrame(draw);
    const ro = new ResizeObserver(() => draw(video.currentTime));
    ro.observe(box);
    return () => {
      off();
      ro.disconnect();
    };
  }, [player, fps, p.poses, p.skeleton, p.showSkeleton, p.sourceSize.width, p.sourceSize.height]);

  const active = activeEvents(p.events, state.time).slice(-6);

  return (
    <div className="player">
      <div className="player-stage" ref={boxRef} onClick={player.toggle}>
        <video ref={player.videoRef} src={p.src} preload="auto" playsInline />
        <canvas ref={canvasRef} className="player-overlay" aria-hidden />
        <div className="player-hud" aria-live="polite">
          {active.map((e, i) => (
            <span key={`${e.fighter}-${e.type}-${e.peak_t}-${i}`} className={`hud-chip kind-${kindOf(e.type, p.moves)}`}>
              <b style={{ color: FIGHTER_COLOR[e.fighter] }}>{e.fighter}</b> {labelOf(e.type, p.moves)}
            </span>
          ))}
        </div>
        {state.reelIndex !== null && <div className="player-badge">Highlights {state.reelIndex + 1}</div>}
      </div>

      <div className="controls" role="toolbar" aria-label="Playback">
        <button className="btn icon" onClick={() => player.stepSeconds(-0.5)} title="Back 0.5 s (Shift+←)">
          «
        </button>
        <button className="btn icon" onClick={() => player.stepFrames(-1)} title="Previous frame (←)">
          ‹
        </button>
        <button className="btn primary icon wide" onClick={player.toggle} title="Play / pause (Space)">
          {state.playing ? "❚❚" : "▶"}
        </button>
        <button className="btn icon" onClick={() => player.stepFrames(1)} title="Next frame (→)">
          ›
        </button>
        <button className="btn icon" onClick={() => player.stepSeconds(0.5)} title="Forward 0.5 s (Shift+→)">
          »
        </button>
        <span className="sep" />
        <button className="btn" onClick={p.onPrevEvent} title="Previous event ([)">
          ⤺ event
        </button>
        <button className="btn" onClick={p.onNextEvent} title="Next event (])">
          event ⤻
        </button>
        <span className="sep" />
        <label className="speed" title="Playback speed (- / =)">
          <span>Speed</span>
          <select value={state.rate} onChange={(e) => player.setRate(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}x
              </option>
            ))}
          </select>
        </label>
        <label className="toggle" title="Skeleton overlay (S)">
          <input type="checkbox" checked={p.showSkeleton} onChange={p.onToggleSkeleton} /> Skeleton
        </label>
        <span className="grow" />
        <span className="readout" title="Time · frame">
          {formatTime(state.time)} <span className="muted">/ {formatTime(state.duration, 1)}</span>
          <span className="frame">f{state.frame}</span>
        </span>
        <button className="btn icon" onClick={p.onHelp} title="Keyboard shortcuts (?)">
          ?
        </button>
      </div>
    </div>
  );
}
