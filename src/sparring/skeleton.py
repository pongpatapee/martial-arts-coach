"""Halpe-26 keypoint layout (what RTMPose body7-halpe26 outputs)."""

from __future__ import annotations

KEYPOINTS: tuple[str, ...] = (
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
    "head",
    "neck",
    "hip",
    "l_big_toe",
    "r_big_toe",
    "l_small_toe",
    "r_small_toe",
    "l_heel",
    "r_heel",
)
K = {name: i for i, name in enumerate(KEYPOINTS)}
N_KEYPOINTS = len(KEYPOINTS)

BONES: tuple[tuple[str, str], ...] = (
    ("l_ankle", "l_knee"),
    ("l_knee", "l_hip"),
    ("r_ankle", "r_knee"),
    ("r_knee", "r_hip"),
    ("l_hip", "r_hip"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("neck", "head"),
    ("nose", "l_eye"),
    ("nose", "r_eye"),
    ("l_eye", "l_ear"),
    ("r_eye", "r_ear"),
    ("l_ankle", "l_heel"),
    ("l_ankle", "l_big_toe"),
    ("r_ankle", "r_heel"),
    ("r_ankle", "r_big_toe"),
)

FIGHTERS: tuple[str, str] = ("A", "B")


def opponent(fighter: str) -> str:
    return "B" if fighter == "A" else "A"
