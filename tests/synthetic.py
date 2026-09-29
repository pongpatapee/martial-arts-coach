"""Synthetic side-view keypoint sequences for testing kinematics and detectors without models.

Pixel coordinates (y down). One torso length (shoulder-mid to hip-mid) = 100 px. Fighter A stands
orthodox on the left facing right; fighter B is A mirrored, on the right facing left, 220 px away
(2.2 TL, i.e. in range).
"""

from __future__ import annotations

import numpy as np

from sparring.skeleton import N_KEYPOINTS, K

FPS = 30.0

GUARD: dict[str, tuple[float, float]] = {
    "nose": (415, 440),
    "l_eye": (418, 434),
    "r_eye": (411, 434),
    "l_ear": (405, 440),
    "r_ear": (398, 440),
    "head": (408, 405),
    "neck": (402, 485),
    "l_shoulder": (410, 500),
    "r_shoulder": (390, 500),
    "l_elbow": (440, 550),
    "r_elbow": (415, 555),
    "l_wrist": (445, 480),
    "r_wrist": (425, 490),
    "l_hip": (410, 600),
    "r_hip": (390, 600),
    "hip": (400, 600),
    "l_knee": (440, 680),
    "r_knee": (370, 680),
    "l_ankle": (460, 760),
    "r_ankle": (345, 760),
    "l_big_toe": (480, 765),
    "r_big_toe": (365, 765),
    "l_small_toe": (475, 765),
    "r_small_toe": (360, 765),
    "l_heel": (452, 765),
    "r_heel": (337, 765),
}

# Full extension of each arm straight at the opponent's head.
EXTENDED = {
    "l": {"l_elbow": (485, 482), "l_wrist": (560, 470)},
    "r": {"r_elbow": (470, 480), "r_wrist": (545, 465)},
}
DROPPED = {
    "l": {"l_elbow": (425, 580), "l_wrist": (440, 640)},
    "r": {"r_elbow": (405, 580), "r_wrist": (420, 640)},
}


def base_pose() -> np.ndarray:
    kp = np.zeros((N_KEYPOINTS, 2), np.float32)
    for name, xy in GUARD.items():
        kp[K[name]] = xy
    return kp


class Scene:
    """Builds a (T, 2, K, 2) keypoint array for fighters A and B, starting in guard.
    `gap_tl` is the hip-to-hip distance in torso lengths (default 2.2: punching range)."""

    def __init__(self, seconds: float, gap_tl: float = 2.2):
        self.mirror_x = 800.0 + gap_tl * 100.0
        self.n = int(seconds * FPS)
        a = np.repeat(base_pose()[None], self.n, axis=0)
        self.kp = np.stack([a, a.copy()], axis=1)  # (T, 2, K, 2), in A's coordinates

    def frame(self, t: float) -> int:
        return int(round(t * FPS))

    def _blend(self, f: int, pose: dict, start: int, frames: int, reverse: bool = False) -> None:
        for i in range(frames + 1):
            w = i / frames
            if reverse:
                w = 1 - w
            fr = start + i
            if fr >= self.n:
                return
            for name, target in pose.items():
                g = np.array(GUARD[name], np.float32)
                self.kp[fr, f, K[name]] = g + w * (np.array(target, np.float32) - g)

    def _hold(self, f: int, pose: dict, a: int, b: int) -> None:
        for name, target in pose.items():
            self.kp[a:b, f, K[name]] = target

    def straight(
        self,
        fighter: int,
        side: str,
        t: float,
        out_frames: int = 4,
        return_frames: int = 5,
        hold: int = 1,
    ) -> Scene:
        """Straight punch peaking at time t (side 'l' = lead for an orthodox fighter)."""
        peak = self.frame(t)
        start = peak - out_frames
        self._blend(fighter, EXTENDED[side], start, out_frames)
        self._hold(fighter, EXTENDED[side], peak, peak + hold)
        self._blend(fighter, EXTENDED[side], peak + hold, return_frames, reverse=True)
        return self

    def drop_hand(self, fighter: int, side: str, t0: float, t1: float) -> Scene:
        self._hold(fighter, DROPPED[side], self.frame(t0), self.frame(t1))
        return self

    def punch_and_stay_out(self, fighter: int, side: str, t: float, stay_s: float) -> Scene:
        peak = self.frame(t)
        self._blend(fighter, EXTENDED[side], peak - 4, 4)
        self._hold(fighter, EXTENDED[side], peak, peak + int(stay_s * FPS))
        self._blend(fighter, EXTENDED[side], peak + int(stay_s * FPS), 5, reverse=True)
        return self

    def move_head(self, fighter: int, t0: float, t1: float, dx: float, dy: float) -> Scene:
        """Shift the face/head keypoints by (dx, dy) pixels, easing in and back out."""
        a, b = self.frame(t0), self.frame(t1)
        names = ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "head")
        mid = (a + b) // 2
        for fr in range(a, b + 1):
            w = 1 - abs(fr - mid) / max(mid - a, 1)
            for n in names:
                self.kp[fr, fighter, K[n]] = np.array(GUARD[n]) + w * np.array([dx, dy])
        return self

    def animate(self, fighter: int, keyframes: list[tuple[float, dict]]) -> Scene:
        """Move keypoints through (time, {name: (x, y)}) keyframes with linear interpolation.
        A keypoint missing from a keyframe is at its guard position (so `(t, {})` = guard)."""
        names = sorted({n for _, pose in keyframes for n in pose})
        times = [t for t, _ in keyframes]
        poses = [{n: pose.get(n, GUARD[n]) for n in names} for _, pose in keyframes]
        for fr in range(self.frame(times[0]), min(self.n, self.frame(times[-1]) + 1)):
            t = fr / FPS
            k = max(0, min(len(times) - 2, int(np.searchsorted(times, t, side="right")) - 1))
            w = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0, 1)
            for n in names:
                a, b = np.array(poses[k][n]), np.array(poses[k + 1][n])
                self.kp[fr, fighter, K[n]] = a + w * (b - a)
        return self

    def build(self) -> np.ndarray:
        kp = self.kp.copy()
        kp[:, 1, :, 0] = self.mirror_x - kp[:, 1, :, 0]  # place B on the right, facing left
        return kp
