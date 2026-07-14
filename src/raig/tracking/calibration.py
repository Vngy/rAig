import json
from dataclasses import dataclass
from pathlib import Path

from raig.core.params import ParamFrame


def _offset_eligible(name: str) -> bool:
    return name.endswith("_rot") or name.startswith(("head_angle", "body_angle"))


@dataclass
class Calibration:
    offsets: dict[str, float]

    def apply(self, values: dict[str, float]) -> dict[str, float]:
        return {
            name: v - self.offsets[name] if name in self.offsets else v
            for name, v in values.items()
        }


def collect(frames: list[ParamFrame]) -> Calibration:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f in frames:
        for name, v in f.values.items():
            if _offset_eligible(name):
                sums[name] = sums.get(name, 0.0) + v
                counts[name] = counts.get(name, 0) + 1
    return Calibration(offsets={n: sums[n] / counts[n] for n in sums})


def sidecar_path(rig_path: str | Path) -> Path:
    p = Path(rig_path)
    return p.with_name(p.name + ".calib.json")


def save_calibration(cal: Calibration, rig_path: str | Path) -> None:
    sidecar_path(rig_path).write_text(json.dumps({"offsets": cal.offsets}, indent=1))


def load_calibration(rig_path: str | Path) -> Calibration | None:
    p = sidecar_path(rig_path)
    if not p.exists():
        return None
    try:
        return Calibration(offsets=json.loads(p.read_text())["offsets"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None  # corrupt/unreadable sidecar degrades to uncalibrated
