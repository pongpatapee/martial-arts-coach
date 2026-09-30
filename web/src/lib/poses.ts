// Decoding the engine's keypoints (int16 pixels, -1 = missing) and placing them over the video.

export interface Poses {
  frames: number;
  fighters: number;
  keypoints: number;
  data: Int16Array; // (frames, fighters, keypoints, 2)
}

export function parseShape(header: string | null): [number, number, number, number] {
  const parts = (header ?? "").split(",").map(Number);
  if (parts.length !== 4 || parts.some((n) => !Number.isInteger(n) || n < 0)) {
    throw new Error(`bad pose shape header: ${header}`);
  }
  return parts as [number, number, number, number];
}

export function decodePoses(buffer: ArrayBuffer, shape: [number, number, number, number]): Poses {
  const [frames, fighters, keypoints, dims] = shape;
  if (dims !== 2) throw new Error("expected 2D keypoints");
  const data = new Int16Array(buffer);
  if (data.length !== frames * fighters * keypoints * 2) throw new Error("pose data size mismatch");
  return { frames, fighters, keypoints, data };
}

/** Keypoint (x, y) in source-video pixels, or null when missing / out of range. */
export function point(p: Poses, frame: number, fighter: number, k: number): [number, number] | null {
  if (frame < 0 || frame >= p.frames) return null;
  const i = ((frame * p.fighters + fighter) * p.keypoints + k) * 2;
  const x = p.data[i]!;
  const y = p.data[i + 1]!;
  return x < 0 || y < 0 ? null : [x, y];
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Where a video of (vw, vh) is drawn inside a box of (bw, bh) with object-fit: contain. */
export function containRect(bw: number, bh: number, vw: number, vh: number): Rect {
  if (vw <= 0 || vh <= 0 || bw <= 0 || bh <= 0) return { x: 0, y: 0, w: bw, h: bh };
  const s = Math.min(bw / vw, bh / vh);
  const w = vw * s;
  const h = vh * s;
  return { x: (bw - w) / 2, y: (bh - h) / 2, w, h };
}
