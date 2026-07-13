from dataclasses import dataclass
from pathlib import Path

import numpy as np
from psd_tools import PSDImage


@dataclass
class LayerRecord:
    name: str
    group_path: tuple[str, ...]
    z_index: int  # ascending draw order, 0 = backmost
    offset: tuple[int, int]  # (left, top) in canvas px
    rgba: np.ndarray  # (H, W, 4) uint8, cropped to layer bbox


def _clean_name(name: str) -> str:
    # pytoshop's Unicode-layer-name ("luni") writer counts a trailing null
    # terminator into its length field (a legacy Photoshop-plugin-SDK
    # convention); psd-tools' unicode-string reader decodes that count
    # literally and does not strip it, so every name/group name round-tripped
    # through this pytoshop->psd-tools pipeline carries one trailing "\x00".
    return name.rstrip("\x00")


def load_layers(psd_path: str | Path) -> tuple[list[LayerRecord], tuple[int, int]]:
    psd = PSDImage.open(psd_path)
    pixel_layers = [l for l in psd.descendants() if not l.is_group()]
    # psd-tools yields file storage order: bottom-most layer first, which is
    # exactly ascending draw order. (Pinned by test_z_order_is_draw_order —
    # if that test fails with inverted comparisons, reverse pixel_layers.)
    records: list[LayerRecord] = []
    for z, layer in enumerate(pixel_layers):
        pil = layer.composite()
        rgba = np.array(pil.convert("RGBA"), dtype=np.uint8)
        path: list[str] = []
        parent = layer.parent
        while parent is not None and parent is not psd:
            path.append(_clean_name(parent.name))
            parent = parent.parent
        records.append(
            LayerRecord(
                name=_clean_name(layer.name),
                group_path=tuple(reversed(path)),
                z_index=z,
                offset=(layer.left, layer.top),
                rgba=rgba,
            )
        )
    return records, (psd.width, psd.height)
