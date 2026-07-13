import numpy as np

from raig.compiler.classify import HEAD_SLOTS
from raig.core.rig import Deformer

_HEAD_DEPTH = {
    "hair_front": 1.3, "iris_l": 1.15, "iris_r": 1.15,
    "brow_l": 1.1, "brow_r": 1.1,
    "eye_white_l": 1.05, "eye_white_r": 1.05, "mouth": 1.05,
    "face": 1.0, "head_misc": 1.0, "hair_back": 0.7,
}
_BODY_SLOTS = {"torso", "arm_l", "arm_r", "leg_l", "leg_r", "misc"}
_BODY_DEPTH = {"arm_l": 1.15, "arm_r": 1.15}


def _union(bboxes):
    arr = np.array(bboxes)
    return arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()


def _layers_of(layer_slots: dict[str, str], slots: set[str]) -> list[str]:
    return [name for name, slot in layer_slots.items() if slot in slots]


def build_deformers(
    layer_slots: dict[str, str],
    slot_bboxes: dict[str, tuple[float, float, float, float]],
    canvas_size: tuple[int, int],
) -> list[Deformer]:
    out: list[Deformer] = []

    for side in ("l", "r"):
        eye_slots = {f"iris_{side}", f"eye_white_{side}"}
        layers = _layers_of(layer_slots, eye_slots)
        boxes = [slot_bboxes[s] for s in eye_slots if s in slot_bboxes]
        if layers and boxes:
            _, y0, _, y1 = _union(boxes)
            out.append(Deformer(
                kind="eye_blink", param=f"eye_{side}_open", layer_names=layers,
                payload={"center_y": float((y0 + y1) / 2.0), "min_scale": 0.05},
            ))

    mouth_layers = _layers_of(layer_slots, {"mouth"})
    if mouth_layers and "mouth" in slot_bboxes:
        out.append(Deformer(
            kind="mouth_open", param="mouth_open", layer_names=mouth_layers,
            payload={"anchor_y": float(slot_bboxes["mouth"][1]), "max_stretch": 0.8},
        ))

    head_layers = _layers_of(layer_slots, set(HEAD_SLOTS))
    head_boxes = [b for s, b in slot_bboxes.items() if s in HEAD_SLOTS]
    if head_layers and head_boxes:
        hx0, hy0, hx1, hy1 = _union(head_boxes)
        hw, hh = hx1 - hx0, hy1 - hy0
        out.append(Deformer(
            kind="head_warp", param="head_angle_x", layer_names=head_layers,
            payload={
                "center": [float((hx0 + hx1) / 2), float((hy0 + hy1) / 2)],
                "radius": float(max(hw, hh) * 0.6),
                "shift_px_per_deg_x": float(hw * 0.005),
                "shift_px_per_deg_y": float(hh * 0.003),
                "roll_center": [float((hx0 + hx1) / 2), float(hy1)],
                "depth": {
                    name: _HEAD_DEPTH.get(layer_slots[name], 1.0)
                    for name in head_layers
                },
            },
        ))

    body_layers = _layers_of(layer_slots, _BODY_SLOTS)
    if body_layers and "torso" in slot_bboxes:
        tx0, ty0, tx1, ty1 = slot_bboxes["torso"]
        out.append(Deformer(
            kind="body_shift", param="body_angle_x", layer_names=body_layers,
            payload={
                "shift_px_per_deg_x": float(canvas_size[0] * 0.004),
                "shift_px_per_deg_y": float(canvas_size[1] * 0.002),
                "roll_center": [float((tx0 + tx1) / 2), float(ty1)],
                "depth": {
                    name: _BODY_DEPTH.get(layer_slots[name], 1.0)
                    for name in body_layers
                },
            },
        ))
    return out
