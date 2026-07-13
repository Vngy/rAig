"""Generates mini_model.psd, the golden test input. Run once:
    uv run python tests/fixtures/make_test_psd.py
Requires dev deps (pytoshop, pillow)."""

from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from pytoshop import enums
from pytoshop.user import nested_layers

W = H = 1024

SKIN = (255, 224, 196, 255)
HAIR = (120, 80, 160, 255)
TORSO = (80, 100, 200, 255)
WHITE = (255, 255, 255, 255)
IRIS = (255, 0, 0, 255)  # pure red: renderer tests count these pixels
MOUTH = (180, 40, 60, 255)


def shape_layer(name: str, kind: str, box: tuple[int, int, int, int], color):
    img = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if kind == "ellipse":
        draw.ellipse(box, fill=color)
    else:
        draw.rectangle(box, fill=color)
    arr = np.array(img)
    ys, xs = np.nonzero(arr[..., 3])
    top, left = int(ys.min()), int(xs.min())
    bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
    crop = np.ascontiguousarray(arr[top:bottom, left:right])
    channels = {0: crop[..., 0], 1: crop[..., 1], 2: crop[..., 2], -1: crop[..., 3]}
    return nested_layers.Image(
        name=name, top=top, left=left, bottom=bottom, right=right, channels=channels
    )


def build_layers():
    head = nested_layers.Group(
        name="head",
        layers=[  # top of stack first
            shape_layer("hair_front", "ellipse", (382, 120, 642, 240), HAIR),
            shape_layer("iris_l", "ellipse", (452, 235, 472, 255), IRIS),
            shape_layer("eye_white_l", "ellipse", (436, 229, 488, 261), WHITE),
            shape_layer("iris_r", "ellipse", (552, 235, 572, 255), IRIS),
            shape_layer("eye_white_r", "ellipse", (536, 229, 588, 261), WHITE),
            shape_layer("mouth", "ellipse", (490, 305, 534, 325), MOUTH),
            shape_layer("face", "ellipse", (392, 130, 632, 390), SKIN),
            shape_layer("hair_back", "ellipse", (362, 90, 662, 410), HAIR),
        ],
    )
    body = nested_layers.Group(
        name="body",
        layers=[
            shape_layer("arm_l", "rect", (330, 400, 400, 640), SKIN),
            shape_layer("arm_r", "rect", (624, 400, 694, 640), SKIN),
            shape_layer("torso", "rect", (400, 380, 624, 660), TORSO),
            shape_layer("leg_l", "rect", (430, 640, 500, 980), SKIN),
            shape_layer("leg_r", "rect", (524, 640, 594, 980), SKIN),
        ],
    )
    return [head, body]  # head group above body group


def main():
    # compression=zip: the installed pytoshop's default RLE path depends on
    # the `packbits` Cython extension, which has no compiled wheel for this
    # Python version (source-only .pyx/.c present, import silently
    # swallowed by pytoshop's `except ImportError: pass`). zip (zlib) avoids
    # that extension entirely; the script is throwaway so this is a safe
    # swap.
    psd = nested_layers.nested_layers_to_psd(
        build_layers(),
        color_mode=enums.ColorMode.rgb,
        size=(H, W),
        compression=enums.Compression.zip,
    )
    out = Path(__file__).parent / "mini_model.psd"
    with open(out, "wb") as f:
        psd.write(f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
