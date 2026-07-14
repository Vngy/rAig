import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    min: float
    max: float


PARAM_SPECS: dict[str, ParamSpec] = {
    s.name: s
    for s in [
        ParamSpec("head_angle_x", 0.0, -30.0, 30.0),  # yaw (turn)
        ParamSpec("head_angle_y", 0.0, -30.0, 30.0),  # pitch (nod)
        ParamSpec("head_angle_z", 0.0, -30.0, 30.0),  # roll (tilt)
        ParamSpec("eye_l_open", 1.0, 0.0, 1.0),
        ParamSpec("eye_r_open", 1.0, 0.0, 1.0),
        ParamSpec("mouth_open", 0.0, 0.0, 1.0),
        ParamSpec("mouth_form", 0.0, -1.0, 1.0),
        ParamSpec("body_angle_x", 0.0, -10.0, 10.0),
        ParamSpec("body_angle_y", 0.0, -10.0, 10.0),
        ParamSpec("body_angle_z", 0.0, -10.0, 10.0),
    ]
}

BONE_ROT_RANGE: tuple[float, float] = (-90.0, 90.0)


def bone_rot_param(bone_name: str) -> str:
    return f"{bone_name}_rot"


@dataclass
class ParamFrame:
    values: dict[str, float]
    timestamp: float


def make_param_frame(
    raw: dict[str, float], timestamp: float | None = None
) -> ParamFrame:
    values: dict[str, float] = {}
    for name, v in raw.items():
        if math.isnan(v):
            # NaN passes min/max clamping unchanged and would otherwise
            # poison Mapper._last / OneEuroFilter state forever. Treat it
            # like an unknown param: omit it so decay/default takes over.
            continue
        spec = PARAM_SPECS.get(name)
        if spec is not None:
            values[name] = float(min(max(v, spec.min), spec.max))
        elif name.endswith("_rot"):
            lo, hi = BONE_ROT_RANGE
            values[name] = float(min(max(v, lo), hi))
        # unknown params are dropped: lenient runtime, never crash
    ts = time.monotonic() if timestamp is None else timestamp
    return ParamFrame(values=values, timestamp=ts)


def rest_frame() -> ParamFrame:
    return ParamFrame(
        values={n: s.default for n, s in PARAM_SPECS.items()},
        timestamp=time.monotonic(),
    )
