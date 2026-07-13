import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class LandmarkBundle:
    timestamp: float  # seconds
    face_blendshapes: dict[str, float] | None = None
    face_matrix: np.ndarray | None = None  # (4, 4)
    pose: np.ndarray | None = None  # (33, 4): x, y, z, visibility
    hand_l: np.ndarray | None = None  # (21, 3)
    hand_r: np.ndarray | None = None  # (21, 3)


_ARRAY_FIELDS = ("face_matrix", "pose", "hand_l", "hand_r")


def save_replay(bundles: Iterable[LandmarkBundle], path: str | Path) -> None:
    with open(path, "w") as f:
        for b in bundles:
            row: dict = {"timestamp": b.timestamp}
            if b.face_blendshapes is not None:
                row["face_blendshapes"] = b.face_blendshapes
            for name in _ARRAY_FIELDS:
                arr = getattr(b, name)
                if arr is not None:
                    row[name] = np.asarray(arr).tolist()
            f.write(json.dumps(row) + "\n")


def load_replay(path: str | Path) -> list[LandmarkBundle]:
    out: list[LandmarkBundle] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.append(LandmarkBundle(
                timestamp=row["timestamp"],
                face_blendshapes=row.get("face_blendshapes"),
                **{
                    name: np.asarray(row[name], dtype=np.float64)
                    if name in row else None
                    for name in _ARRAY_FIELDS
                },
            ))
    return out
