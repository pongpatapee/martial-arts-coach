import { useCallback, useEffect, useRef, useState } from "react";
import type { Highlight } from "../lib/events";
import { clamp, frameAt, nextSpeed, timeOfFrame } from "../lib/time";

export interface PlayerState {
  time: number;
  frame: number;
  duration: number;
  playing: boolean;
  rate: number;
  /** Index into the highlight reel while "play highlights" is running. */
  reelIndex: number | null;
}

export interface Player {
  state: PlayerState;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** Called with the media time of every presented frame (for the overlay). */
  onFrame: (cb: (mediaTime: number) => void) => () => void;
  toggle: () => void;
  pause: () => void;
  seek: (t: number) => void;
  stepFrames: (n: number) => void;
  stepSeconds: (s: number) => void;
  setRate: (r: number) => void;
  changeRate: (direction: 1 | -1) => void;
  playReel: (items: Highlight[], startAt?: number) => void;
  stopReel: () => void;
}

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

/** Video transport with frame-accurate time. Time comes from requestVideoFrameCallback (the
 * time of the frame actually on screen), falling back to timeupdate where unsupported. */
export function usePlayer(fps: number): Player {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [state, setState] = useState<PlayerState>({
    time: 0,
    frame: 0,
    duration: 0,
    playing: false,
    rate: 1,
    reelIndex: null,
  });
  const listeners = useRef(new Set<(t: number) => void>());
  const reel = useRef<{ items: Highlight[]; index: number } | null>(null);

  const publish = useCallback(
    (t: number) => {
      for (const cb of listeners.current) cb(t);
      setState((s) => (Math.abs(s.time - t) < 1e-4 ? s : { ...s, time: t, frame: frameAt(t, fps) }));
      const r = reel.current;
      const v = videoRef.current;
      if (r && v && t >= r.items[r.index]!.end) {
        if (r.index + 1 < r.items.length) {
          r.index += 1;
          v.currentTime = r.items[r.index]!.start;
          setState((s) => ({ ...s, reelIndex: r.index }));
        } else {
          reel.current = null;
          v.pause();
          setState((s) => ({ ...s, reelIndex: null }));
        }
      }
    },
    [fps],
  );

  useEffect(() => {
    const v = videoRef.current as VideoWithRvfc | null;
    if (!v) return;
    let handle = 0;
    const hasRvfc = typeof v.requestVideoFrameCallback === "function";
    const loop = (_: number, meta: { mediaTime: number }) => {
      publish(meta.mediaTime);
      handle = v.requestVideoFrameCallback!(loop);
    };
    if (hasRvfc) handle = v.requestVideoFrameCallback!(loop);
    const onTime = () => !hasRvfc && publish(v.currentTime);
    const onSeeked = () => publish(v.currentTime);
    const onPlay = () => setState((s) => ({ ...s, playing: true }));
    const onPause = () => setState((s) => ({ ...s, playing: false }));
    const onMeta = () => setState((s) => ({ ...s, duration: v.duration || 0 }));
    const onRate = () => setState((s) => ({ ...s, rate: v.playbackRate }));
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("seeked", onSeeked);
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", onPause);
    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("durationchange", onMeta);
    v.addEventListener("ratechange", onRate);
    if (v.readyState >= 1) onMeta();
    return () => {
      if (hasRvfc) v.cancelVideoFrameCallback?.(handle);
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("seeked", onSeeked);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", onPause);
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("durationchange", onMeta);
      v.removeEventListener("ratechange", onRate);
    };
  }, [publish]);

  const withVideo = useCallback((fn: (v: HTMLVideoElement) => void) => {
    const v = videoRef.current;
    if (v) fn(v);
  }, []);

  const stopReel = useCallback(() => {
    reel.current = null;
    setState((s) => ({ ...s, reelIndex: null }));
  }, []);

  const seek = useCallback(
    (t: number) =>
      withVideo((v) => {
        v.currentTime = clamp(t, 0, v.duration || t);
      }),
    [withVideo],
  );

  return {
    state,
    videoRef,
    onFrame: (cb) => {
      listeners.current.add(cb);
      return () => listeners.current.delete(cb);
    },
    toggle: () => withVideo((v) => (v.paused ? void v.play() : v.pause())),
    pause: () => withVideo((v) => v.pause()),
    seek: (t) => {
      stopReel();
      seek(t);
    },
    stepFrames: (n) =>
      withVideo((v) => {
        v.pause();
        stopReel();
        v.currentTime = clamp(timeOfFrame(frameAt(v.currentTime, fps) + n, fps), 0, v.duration || 0);
      }),
    stepSeconds: (s) => {
      stopReel();
      withVideo((v) => {
        v.currentTime = clamp(v.currentTime + s, 0, v.duration || 0);
      });
    },
    setRate: (r) => withVideo((v) => (v.playbackRate = r)),
    changeRate: (d) => withVideo((v) => (v.playbackRate = nextSpeed(v.playbackRate, d))),
    playReel: (items, startAt = 0) =>
      withVideo((v) => {
        if (!items.length) return;
        reel.current = { items, index: startAt };
        setState((s) => ({ ...s, reelIndex: startAt }));
        v.currentTime = items[startAt]!.start;
        void v.play();
      }),
    stopReel,
  };
}
