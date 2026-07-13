import math

import numpy as np

from raig.core.params import PARAM_SPECS, ParamFrame, make_param_frame
from raig.tracking.bundle import LandmarkBundle
from raig.tracking.smoothing import FilterBank

_DECAY_TAU = 0.15  # seconds; ~96% back to rest in 0.5 s
_MIN_VISIBILITY = 0.5

# mirror mode: avatar screen-left (_l) follows the user's anatomical RIGHT.
# pose indices: (shoulder, elbow, wrist) / (hip, knee, ankle)
_ARM = {"l": (12, 14, 16), "r": (11, 13, 15)}
_LEG = {"l": (24, 26, 28), "r": (23, 25, 27)}
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24


def angle_from_down(v: np.ndarray) -> float:
    """Signed degrees from straight-down; positive = CCW in canvas coords."""
    return -math.degrees(math.atan2(v[0], v[1]))


def _default(name: str) -> float:
    spec = PARAM_SPECS.get(name)
    return spec.default if spec is not None else 0.0


class Mapper:
    def __init__(self, calibration=None, smooth: bool = True):
        self.calibration = calibration
        self._bank = FilterBank() if smooth else None
        self._last: dict[str, float] = {}
        self._last_t: float | None = None

    # --- per-source extraction (return {} when source absent) -------------

    def _face_params(self, b: LandmarkBundle) -> dict[str, float]:
        out: dict[str, float] = {}
        if b.face_blendshapes is not None:
            bs = b.face_blendshapes
            out["eye_l_open"] = 1.0 - bs.get("eyeBlinkRight", 0.0)  # mirrored
            out["eye_r_open"] = 1.0 - bs.get("eyeBlinkLeft", 0.0)
            out["mouth_open"] = bs.get("jawOpen", 0.0)
            smile = (bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)) / 2
            frown = (bs.get("mouthFrownLeft", 0.0) + bs.get("mouthFrownRight", 0.0)) / 2
            out["mouth_form"] = smile - frown
        if b.face_matrix is not None:
            r = np.asarray(b.face_matrix)[:3, :3]
            out["head_angle_x"] = math.degrees(math.atan2(-r[2, 0], r[2, 2]))
            out["head_angle_y"] = math.degrees(math.atan2(r[2, 1], r[2, 2]))
            out["head_angle_z"] = math.degrees(math.atan2(r[1, 0], r[0, 0]))
        return out

    def _limb_params(self, pose: np.ndarray, side: str, idxs,
                     prefix: str) -> dict[str, float]:
        if pose[list(idxs), 3].min() < _MIN_VISIBILITY:
            return {}
        pts = pose[list(idxs), :2]
        upper = angle_from_down(pts[1] - pts[0])
        lower = angle_from_down(pts[2] - pts[1]) - upper
        out = {f"{prefix}_upper_{side}_rot": upper, f"{prefix}_lower_{side}_rot": lower}
        return out

    def _body_params(self, b: LandmarkBundle) -> dict[str, float]:
        if b.pose is None:
            return {}
        pose = b.pose
        out: dict[str, float] = {}
        for side in ("l", "r"):
            out.update(self._limb_params(pose, side, _ARM[side], "arm"))
            out.update(self._limb_params(pose, side, _LEG[side], "leg"))
        shoulders = pose[[_L_SHOULDER, _R_SHOULDER]]
        hips = pose[[_L_HIP, _R_HIP]]
        if shoulders[:, 3].min() >= _MIN_VISIBILITY:
            d = shoulders[0, :2] - shoulders[1, :2]  # right→left on screen
            out["body_angle_z"] = math.degrees(math.atan2(d[1], d[0]))
            if hips[:, 3].min() >= _MIN_VISIBILITY:
                lean = shoulders[:, 0].mean() - hips[:, 0].mean()
                out["body_angle_x"] = lean * 60.0
        return out

    def _hand_params(self, b: LandmarkBundle) -> dict[str, float]:
        out: dict[str, float] = {}
        for side, hand in (("l", b.hand_r), ("r", b.hand_l)):  # mirrored
            if hand is None:
                continue
            wrist, middle_mcp = np.asarray(hand)[0, :2], np.asarray(hand)[9, :2]
            arm_lower = self._last.get(f"arm_lower_{side}_rot", 0.0)
            arm_upper = self._last.get(f"arm_upper_{side}_rot", 0.0)
            out[f"hand_{side}_rot"] = (
                angle_from_down(middle_mcp - wrist) - arm_lower - arm_upper
            )
        return out

    # --- main entry --------------------------------------------------------

    def map(self, b: LandmarkBundle) -> ParamFrame:
        fresh: dict[str, float] = {}
        fresh.update(self._face_params(b))
        fresh.update(self._body_params(b))
        fresh.update(self._hand_params(b))

        dt = 0.0 if self._last_t is None else max(b.timestamp - self._last_t, 0.0)
        decay = math.exp(-dt / _DECAY_TAU) if dt > 0 else 0.0
        values = dict(fresh)
        for name, prev in self._last.items():
            if name not in values:  # source lost: decay toward default
                d = _default(name)
                values[name] = d + (prev - d) * decay

        if self.calibration is not None:
            values = self.calibration.apply(values)
        if self._bank is not None:
            values = self._bank.apply(values, b.timestamp)

        self._last = dict(values)
        self._last_t = b.timestamp
        return make_param_frame(values, timestamp=b.timestamp)
