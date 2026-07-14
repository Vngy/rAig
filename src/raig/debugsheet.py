import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from raig.core.rig import Rig, load_rig


def _safe(name: str) -> str:
    """ASCII-only filename sanitizer: runs of chars outside [A-Za-z0-9_-]
    become "_"; names with nothing else left (e.g. all-Japanese) become
    "layer"."""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return s if s.strip("_-") else "layer"


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

    path = outdir / "000_overview.png"
    # _overview draws with cv2 BGR-convention color tuples; PIL interprets
    # arrays as RGB, so convert here. _layer_sheet needs no conversion: its
    # base is the RGB texture and its only overlay color (0, 255, 0) is
    # channel-symmetric — converting it would swap the texture's R/B.
    Image.fromarray(cv2.cvtColor(_overview(rig), cv2.COLOR_BGR2RGB)).save(path)
    written.append(path)

    for l in rig.layers:
        # 3-digit width: real models exceed 99 layers. Offset by +1 so the
        # filename numbering starts at 001, reserving 000 for the overview —
        # otherwise a 2-vs-3-digit prefix mismatch (or an alphabetically
        # early layer name tying at "00_") can sort a layer file ahead of
        # 00_overview.png in a directory listing.
        path = outdir / f"{l.z_index + 1:03d}_{_safe(l.layer_name)}.png"
        Image.fromarray(_layer_sheet(l)).save(path)
        written.append(path)
    return written


def debug_command(args) -> int:
    try:
        rig = load_rig(args.rig)
    except Exception as e:  # missing/corrupt .raig: clean exit, not a traceback
        print(f"debug failed: cannot load rig {args.rig!r}: {e}", file=sys.stderr)
        return 1
    written = write_debug_sheet(rig, args.outdir)
    print(f"wrote {len(written)} debug images to {args.outdir}")
    return 0
