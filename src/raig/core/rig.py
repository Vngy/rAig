import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Bone:
    name: str
    parent: str | None
    head: np.ndarray  # (2,) float32 canvas px
    tail: np.ndarray  # (2,) float32 canvas px


@dataclass
class LayerMesh:
    layer_name: str
    part_slot: str
    z_index: int
    texture: np.ndarray  # (H, W, 4) uint8 straight alpha
    offset: tuple[int, int]  # canvas position of texture top-left
    vertices: np.ndarray  # (N, 2) float32 canvas coords
    uvs: np.ndarray  # (N, 2) float32 in [0, 1]
    triangles: np.ndarray  # (M, 3) int32
    bone_indices: np.ndarray  # (N, 2) int32, -1 = unbound
    bone_weights: np.ndarray  # (N, 2) float32
    clip_to: str | None = None  # layer_name whose alpha masks this layer


@dataclass
class Deformer:
    kind: str  # "eye_blink" | "mouth_open" | "head_warp" | "body_shift"
    param: str
    layer_names: list[str]
    payload: dict = field(default_factory=dict)


@dataclass
class Rig:
    canvas_size: tuple[int, int]
    layers: list[LayerMesh]
    bones: list[Bone]
    deformers: list[Deformer]


_LAYER_ARRAYS = (
    "texture",
    "vertices",
    "uvs",
    "triangles",
    "bone_indices",
    "bone_weights",
)


def save_rig(rig: Rig, path: str | Path) -> None:
    manifest = {
        "version": 1,
        "canvas_size": list(rig.canvas_size),
        "layers": [
            {
                "layer_name": l.layer_name,
                "part_slot": l.part_slot,
                "z_index": l.z_index,
                "offset": list(l.offset),
                "clip_to": l.clip_to,
            }
            for l in rig.layers
        ],
        "bones": [
            {"name": b.name, "parent": b.parent} for b in rig.bones
        ],
        "deformers": [
            {
                "kind": d.kind,
                "param": d.param,
                "layer_names": d.layer_names,
                "payload": d.payload,
            }
            for d in rig.deformers
        ],
    }
    arrays: dict[str, np.ndarray] = {}
    for i, l in enumerate(rig.layers):
        for key in _LAYER_ARRAYS:
            arrays[f"layer{i}_{key}"] = getattr(l, key)
    for j, b in enumerate(rig.bones):
        arrays[f"bone{j}_head"] = b.head
        arrays[f"bone{j}_tail"] = b.tail

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        zf.writestr("arrays.npz", buf.getvalue())


def load_rig(path: str | Path) -> Rig:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        npz = np.load(io.BytesIO(zf.read("arrays.npz")))

    layers = []
    for i, lm in enumerate(manifest["layers"]):
        layers.append(
            LayerMesh(
                layer_name=lm["layer_name"],
                part_slot=lm["part_slot"],
                z_index=lm["z_index"],
                offset=tuple(lm["offset"]),
                clip_to=lm["clip_to"],
                **{key: npz[f"layer{i}_{key}"] for key in _LAYER_ARRAYS},
            )
        )
    layers.sort(key=lambda l: l.z_index)
    bones = [
        Bone(
            name=bm["name"],
            parent=bm["parent"],
            head=npz[f"bone{j}_head"],
            tail=npz[f"bone{j}_tail"],
        )
        for j, bm in enumerate(manifest["bones"])
    ]
    deformers = [
        Deformer(
            kind=dm["kind"],
            param=dm["param"],
            layer_names=dm["layer_names"],
            payload=dm["payload"],
        )
        for dm in manifest["deformers"]
    ]
    return Rig(
        canvas_size=tuple(manifest["canvas_size"]),
        layers=layers,
        bones=bones,
        deformers=deformers,
    )
