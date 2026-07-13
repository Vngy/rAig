import numpy as np

from raig.core.params import PARAM_SPECS, ParamFrame, bone_rot_param
from raig.core.rig import Deformer, Rig


def param_value(frame: ParamFrame, name: str) -> float:
    if name in frame.values:
        return frame.values[name]
    spec = PARAM_SPECS.get(name)
    return spec.default if spec is not None else 0.0


def _rot_about(theta: float, pivot: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    m = np.array([[c, -s, 0.0], [s, c, 0.0]])
    m[:, 2] = pivot - m[:, :2] @ pivot
    return m


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty((2, 3))
    out[:, :2] = a[:, :2] @ b[:, :2]
    out[:, 2] = a[:, :2] @ b[:, 2] + a[:, 2]
    return out


def _bone_world(rig: Rig, frame: ParamFrame) -> np.ndarray:
    world: dict[str, np.ndarray] = {}
    out = np.zeros((max(len(rig.bones), 1), 2, 3))
    for i, bone in enumerate(rig.bones):
        theta = np.deg2rad(param_value(frame, bone_rot_param(bone.name)))
        local = _rot_about(theta, bone.head.astype(np.float64))
        m = _compose(world[bone.parent], local) if bone.parent else local
        world[bone.name] = m
        out[i] = m
    return out


def _skin(layer, world: np.ndarray) -> np.ndarray:
    v = layer.vertices.astype(np.float64)
    homo = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)  # (N,3)
    idx = np.clip(layer.bone_indices, 0, None)  # safe gather; weight 0 where -1
    mats = world[idx]  # (N,2,2,3)
    skinned = np.einsum("nkij,nj->nki", mats, homo)  # (N,2,2)
    w = layer.bone_weights.astype(np.float64)[..., None]
    passthrough = 1.0 - layer.bone_weights.sum(axis=1, dtype=np.float64)
    return (skinned * w).sum(axis=1) + v * passthrough[:, None]


def _apply_deformer(d: Deformer, name: str, verts: np.ndarray,
                    frame: ParamFrame) -> np.ndarray:
    p = d.payload
    if d.kind == "eye_blink":
        value = param_value(frame, d.param)
        scale = p["min_scale"] + (1.0 - p["min_scale"]) * value
        verts[:, 1] = p["center_y"] + (verts[:, 1] - p["center_y"]) * scale
    elif d.kind == "mouth_open":
        value = param_value(frame, d.param)
        verts[:, 1] = p["anchor_y"] + (verts[:, 1] - p["anchor_y"]) * (
            1.0 + p["max_stretch"] * value
        )
    elif d.kind == "head_warp":
        ax = param_value(frame, "head_angle_x")
        ay = param_value(frame, "head_angle_y")
        az = param_value(frame, "head_angle_z")
        depth = p["depth"].get(name, 1.0)
        center = np.asarray(p["center"])
        dist = np.linalg.norm(verts - center, axis=1)
        falloff = np.clip(np.cos(np.pi / 2.0 * dist / p["radius"]), 0.0, 1.0)
        verts[:, 0] += ax * p["shift_px_per_deg_x"] * depth * falloff
        verts[:, 1] += ay * p["shift_px_per_deg_y"] * depth * falloff
        if az != 0.0:
            rc = np.asarray(p["roll_center"])
            theta = np.deg2rad(az) * falloff
            c, s = np.cos(theta), np.sin(theta)
            rel = verts - rc
            verts = rc + np.stack(
                [c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]],
                axis=1,
            )
    elif d.kind == "body_shift":
        bx = param_value(frame, "body_angle_x")
        by = param_value(frame, "body_angle_y")
        bz = param_value(frame, "body_angle_z")
        depth = p["depth"].get(name, 1.0)
        verts[:, 0] += bx * p["shift_px_per_deg_x"] * depth
        verts[:, 1] += by * p["shift_px_per_deg_y"] * depth
        if bz != 0.0:
            rc = np.asarray(p["roll_center"])
            m = _rot_about(np.deg2rad(bz), rc.astype(np.float64))
            verts = verts @ m[:, :2].T + m[:, 2]
    return verts


def deform_rig(rig: Rig, frame: ParamFrame) -> dict[str, np.ndarray]:
    world = _bone_world(rig, frame)
    out = {l.layer_name: _skin(l, world) for l in rig.layers}
    for d in rig.deformers:
        for name in d.layer_names:
            if name in out:
                out[name] = _apply_deformer(d, name, out[name], frame)
    return {name: v.astype(np.float32) for name, v in out.items()}
