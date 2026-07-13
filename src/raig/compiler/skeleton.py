import numpy as np

from raig.compiler.classify import HEAD_SLOTS
from raig.compiler.errors import CompileError
from raig.core.rig import Bone

SLOT_TO_CHAIN: dict[str, list[str]] = {
    "torso": ["hips", "spine", "chest"],
    "arm_l": ["arm_upper_l", "arm_lower_l", "hand_l"],
    "arm_r": ["arm_upper_r", "arm_lower_r", "hand_r"],
    "leg_l": ["leg_upper_l", "leg_lower_l"],
    "leg_r": ["leg_upper_r", "leg_lower_r"],
    "misc": ["spine"],
    **{slot: ["head"] for slot in HEAD_SLOTS},
}


def _v(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float32)


def _union(bboxes: list[tuple[float, float, float, float]]):
    arr = np.array(bboxes)
    return arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()


def _long_axis_endpoints(bbox):
    x0, y0, x1, y1 = bbox
    if (y1 - y0) >= (x1 - x0):  # tall box: vertical axis
        cx = (x0 + x1) / 2.0
        return _v(cx, y0), _v(cx, y1)
    cy = (y0 + y1) / 2.0
    return _v(x0, cy), _v(x1, cy)


def _limb_chain(names, parents, bbox, anchor, ts):
    p0, p1 = _long_axis_endpoints(bbox)
    if np.linalg.norm(p1 - anchor) < np.linalg.norm(p0 - anchor):
        p0, p1 = p1, p0
    joints = [p0 + (p1 - p0) * t for t in ts]
    return [
        Bone(name, parent, joints[i], joints[i + 1])
        for i, (name, parent) in enumerate(zip(names, parents))
    ]


def fit_skeleton(
    slot_bboxes: dict[str, tuple[float, float, float, float]],
    canvas_size: tuple[int, int],
) -> list[Bone]:
    if "torso" not in slot_bboxes:
        raise CompileError(
            f"no torso layer classified; found slots: {sorted(slot_bboxes)}"
        )
    head_boxes = [b for s, b in slot_bboxes.items() if s in HEAD_SLOTS]
    if not head_boxes:
        raise CompileError(
            f"no head layers classified; found slots: {sorted(slot_bboxes)}"
        )
    hx0, hy0, hx1, hy1 = _union(head_boxes)
    tx0, ty0, tx1, ty1 = slot_bboxes["torso"]
    tcx, th = (tx0 + tx1) / 2.0, ty1 - ty0
    hcx, hh = (hx0 + hx1) / 2.0, hy1 - hy0

    hips = Bone("hips", None, _v(tcx, ty0 + 0.8 * th), _v(tcx, ty0 + 0.55 * th))
    spine = Bone("spine", "hips", hips.tail, _v(tcx, ty0 + 0.3 * th))
    chest = Bone("chest", "spine", spine.tail, _v(tcx, ty0 + 0.05 * th))
    neck = Bone("neck", "chest", chest.tail, _v(hcx, hy1))
    head = Bone("head", "neck", neck.tail, _v(hcx, hy0 + 0.4 * hh))
    bones = [hips, spine, chest, neck, head]

    for side in ("l", "r"):
        arm_slot, leg_slot = f"arm_{side}", f"leg_{side}"
        if arm_slot in slot_bboxes:
            bones += _limb_chain(
                [f"arm_upper_{side}", f"arm_lower_{side}", f"hand_{side}"],
                ["chest", f"arm_upper_{side}", f"arm_lower_{side}"],
                slot_bboxes[arm_slot], chest.tail, [0.0, 0.4, 0.8, 1.0],
            )
        if leg_slot in slot_bboxes:
            bones += _limb_chain(
                [f"leg_upper_{side}", f"leg_lower_{side}"],
                ["hips", f"leg_upper_{side}"],
                slot_bboxes[leg_slot], hips.head, [0.0, 0.5, 1.0],
            )
    return bones


def _dist_to_segment(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = np.clip((pts - a) @ ab / max(denom, 1e-9), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def assign_weights(
    vertices: np.ndarray, chain_names: list[str], bones: list[Bone]
) -> tuple[np.ndarray, np.ndarray]:
    n = vertices.shape[0]
    indices = np.full((n, 2), -1, dtype=np.int32)
    weights = np.zeros((n, 2), dtype=np.float32)
    name_to_idx = {b.name: i for i, b in enumerate(bones)}
    chain_idx = [name_to_idx[c] for c in chain_names if c in name_to_idx]
    if not chain_idx:
        return indices, weights
    if len(chain_idx) == 1:
        indices[:, 0] = chain_idx[0]
        weights[:, 0] = 1.0
        return indices, weights

    pts = vertices.astype(np.float64)
    dists = np.stack(
        [_dist_to_segment(pts, bones[i].head.astype(np.float64),
                          bones[i].tail.astype(np.float64)) for i in chain_idx],
        axis=1,
    )  # (N, len(chain))
    order = np.argsort(dists, axis=1)[:, :2]
    near = np.take_along_axis(dists, order, axis=1)
    inv = 1.0 / (near + 1e-6) ** 2
    w = inv / inv.sum(axis=1, keepdims=True)
    indices[:] = np.array(chain_idx, dtype=np.int32)[order]
    weights[:] = w.astype(np.float32)
    return indices, weights
