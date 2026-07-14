import os
import urllib.request
from pathlib import Path

_MODEL_FILES = {
    "face": "face_landmarker.task",
    "pose": "pose_landmarker_full.task",
    "hand": "hand_landmarker.task",
}
_BASE = "https://storage.googleapis.com/mediapipe-models"
_MODEL_URLS = {
    "face": f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "hand": f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def models_dir() -> Path:
    env = os.environ.get("RAIG_MODELS_DIR")
    return Path(env) if env else Path.home() / ".cache" / "raig" / "models"


def _default_fetch(url: str, dst: Path) -> None:
    print(f"downloading {url} -> {dst}")
    urllib.request.urlretrieve(url, dst)


def ensure_models(fetch=None) -> dict[str, Path]:
    fetch = fetch or _default_fetch
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, filename in _MODEL_FILES.items():
        path = d / filename
        if not path.exists():
            fetch(_MODEL_URLS[key], path)
        out[key] = path
    return out
