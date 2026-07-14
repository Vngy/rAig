import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from psd_tools import PSDImage

# Kinds psd-tools reports for a raster-compositable layer. Adjustment layers
# (kind is the specific adjustment class name, e.g. "brightnesscontrast",
# "curves"), fill layers, and type/text layers ("type") are excluded — they
# aren't pixel content and would otherwise be composited as if they were.
_PIXEL_KINDS = frozenset({"pixel", "shape", "smartobject"})


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


def _uniquify_names(names: list[str]) -> list[str]:
    """Disambiguate duplicate layer names in encounter order:
    "name", "name#2", "name#3", ... Every downstream consumer (mesh/render
    dicts keyed on layer name) reads the post-ingest string, so this is the
    only place duplicates need handling."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
    return out


def load_layers(psd_path: str | Path) -> tuple[list[LayerRecord], tuple[int, int]]:
    psd = PSDImage.open(psd_path)
    pixel_layers = []
    for layer in psd.descendants():
        if layer.is_group():
            continue
        if layer.kind not in _PIXEL_KINDS:
            print(
                f"warning: skipping non-pixel layer {_clean_name(layer.name)!r} "
                f"(kind={layer.kind!r})",
                file=sys.stderr,
            )
            continue
        pixel_layers.append(layer)
    # psd-tools yields file storage order: bottom-most layer first, which is
    # exactly ascending draw order. (Pinned by test_z_order_is_draw_order —
    # if that test fails with inverted comparisons, reverse pixel_layers.)
    names = _uniquify_names([_clean_name(l.name) for l in pixel_layers])
    records: list[LayerRecord] = []
    for z, (layer, name) in enumerate(zip(pixel_layers, names)):
        pil = layer.composite()
        rgba = np.array(pil.convert("RGBA"), dtype=np.uint8)
        path: list[str] = []
        parent = layer.parent
        while parent is not None and parent is not psd:
            path.append(_clean_name(parent.name))
            parent = parent.parent
        records.append(
            LayerRecord(
                name=name,
                group_path=tuple(reversed(path)),
                z_index=z,
                offset=(layer.left, layer.top),
                rgba=rgba,
            )
        )
    return records, (psd.width, psd.height)
