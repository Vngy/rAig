"""Synthetic landmark sessions: deterministic motion for tests and demos."""

import math
import sys

import numpy as np

from raig.tracking.bundle import LandmarkBundle, save_replay

_R_SHOULDER, _R_ELBOW, _R_WRIST = 12, 14, 16
_L_SHOULDER, _L_ELBOW, _L_WRIST = 11, 13, 15
_L_HIP, _R_HIP = 23, 24


def _yaw_matrix(deg: float) -> np.ndarray:
    t = math.radians(deg)
    m = np.eye(4)
    m[0, 0], m[0, 2] = math.cos(t), math.sin(t)
    m[2, 0], m[2, 2] = -math.sin(t), math.cos(t)
    return m


def _base_pose() -> np.ndarray:
    pose = np.zeros((33, 4))
    pose[:, 3] = 1.0
    pose[_L_SHOULDER, :2] = (0.62, 0.45)
    pose[_R_SHOULDER, :2] = (0.38, 0.45)
    pose[_L_HIP, :2] = (0.58, 0.75)
    pose[_R_HIP, :2] = (0.42, 0.75)
    pose[_L_ELBOW, :2] = (0.66, 0.60)
    pose[_L_WRIST, :2] = (0.68, 0.74)
    return pose


def make_session(n_frames: int = 300, fps: int = 60) -> list[LandmarkBundle]:
    out: list[LandmarkBundle] = []
    for i in range(n_frames):
        t = i / fps
        blink = 1.0 if (60 <= i < 70) or (180 <= i < 190) else 0.0
        jaw = max(0.0, math.sin(2 * math.pi * t / 2.0))
        pose = _base_pose()
        # user's right arm waves: elbow orbits the shoulder
        wave = math.sin(2 * math.pi * t / 3.0) * 0.8  # radians around down
        sx, sy = pose[_R_SHOULDER, :2]
        pose[_R_ELBOW, :2] = (sx + 0.14 * math.sin(wave), sy + 0.14 * math.cos(wave))
        ex, ey = pose[_R_ELBOW, :2]
        pose[_R_WRIST, :2] = (ex + 0.13 * math.sin(wave * 1.5),
                              ey + 0.13 * math.cos(wave * 1.5))
        out.append(LandmarkBundle(
            timestamp=t,
            face_blendshapes={
                "eyeBlinkLeft": blink, "eyeBlinkRight": blink,
                "jawOpen": jaw,
                "mouthSmileLeft": 0.2, "mouthSmileRight": 0.2,
                "mouthFrownLeft": 0.0, "mouthFrownRight": 0.0,
            },
            face_matrix=_yaw_matrix(20.0 * math.sin(2 * math.pi * t / 4.0)),
            pose=pose,
        ))
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "session_synthetic.jsonl"
    save_replay(make_session(600), path)
    print(f"wrote {path}")
