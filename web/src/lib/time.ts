// Time <-> frame math. The engine's frame i has timestamp i / fps; a video element shows frame i
// for currentTime in [i / fps, (i + 1) / fps).

export const SPEEDS = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2] as const;

/** Frame shown at time t (robust to float error just below a frame boundary). */
export function frameAt(t: number, fps: number): number {
  return Math.max(0, Math.floor(t * fps + 1e-3));
}

/** A currentTime that reliably displays frame f (a hair past its start). */
export function timeOfFrame(f: number, fps: number): number {
  return Math.max(0, (f + 0.05) / fps);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** 75.456 -> "1:15.45" */
export function formatTime(t: number, decimals = 2): string {
  const safe = Math.max(0, t);
  const m = Math.floor(safe / 60);
  const s = safe - m * 60;
  const width = decimals > 0 ? 3 + decimals : 2;
  return `${m}:${s.toFixed(decimals).padStart(width, "0")}`;
}

/** 5 -> "5s", 125 -> "2m 05s" */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

export function nextSpeed(current: number, direction: 1 | -1): number {
  const i = SPEEDS.findIndex((s) => Math.abs(s - current) < 1e-6);
  const from = i === -1 ? SPEEDS.indexOf(1) : i;
  return SPEEDS[clamp(from + direction, 0, SPEEDS.length - 1)] ?? 1;
}
