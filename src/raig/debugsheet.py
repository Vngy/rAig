import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from raig.core.rig import Rig, load_rig


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name) or "layer"


def _overview(rig: Rig) -> np.ndarray:
    w, h = rig.canvas_size
    img = np.zeros((h, w, 3), np.uint8)
    for l in rig.layers:
        lh, lw = l.texture.shape[:2]
        x0, y0 = l.offset
        cv2.rectangle(img, (x0, y0), (x0 + lw, y0 + lh), (80, 80, 80), 1)
        cv2.putText(img, l.part_slot, (x0 + 2, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    for b in rig.bones:
        p0 = tuple(np.round(b.head).astype(int))
        p1 = tuple(np.round(b.tail).astype(int))
        cv2.line(img, p0, p1, (0, 200, 255), 2)
        cv2.circle(img, p0, 4, (0, 120, 255), -1)
    return img


def _layer_sheet(layer) -> np.ndarray:
    tex = layer.texture
    alpha = tex[..., 3:4].astype(np.float32) / 255.0
    gray = np.full(tex[..., :3].shape, 64, np.float32)
    img = (tex[..., :3].astype(np.float32) * alpha + gray * (1 - alpha)).astype(np.uint8)
    img = np.ascontiguousarray(img)
    local = layer.vertices - np.array(layer.offset, np.float32)
    for tri in layer.triangles:
        pts = np.round(local[tri]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    return img


def write_debug_sheet(rig: Rig, outdir: str | Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    path = outdir / "00_overview.png"
    Image.fromarray(_overview(rig)).save(path)
    written.append(path)

    for l in rig.layers:
        path = outdir / f"{l.z_index:02d}_{_safe(l.layer_name)}.png"
        Image.fromarray(_layer_sheet(l)).save(path)
        written.append(path)
    return written


def debug_command(args) -> int:
    rig = load_rig(args.rig)
    written = write_debug_sheet(rig, args.outdir)
    print(f"wrote {len(written)} debug images to {args.outdir}")
    return 0
