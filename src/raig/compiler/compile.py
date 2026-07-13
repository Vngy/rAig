import sys
from pathlib import Path

import numpy as np

from raig.compiler.classify import classify_layers, load_overrides
from raig.compiler.deformers import build_deformers
from raig.compiler.errors import CompileError
from raig.compiler.meshing import build_mesh, grid_step_for_slot
from raig.compiler.psd_ingest import LayerRecord, load_layers
from raig.compiler.skeleton import SLOT_TO_CHAIN, assign_weights, fit_skeleton
from raig.core.rig import LayerMesh, Rig, save_rig


def compile_from_records(
    records: list[LayerRecord],
    canvas_size: tuple[int, int],
    overrides: dict[str, str] | None = None,
) -> Rig:
    classified = classify_layers(records, canvas_size, overrides)

    meshed = []  # (Classified, MeshResult)
    for c in classified:
        try:
            mesh = build_mesh(c.record.rgba, grid_step_for_slot(c.slot))
        except CompileError as e:
            print(
                f"warning: skipping layer {c.record.name!r} ({e})",
                file=sys.stderr,
            )
            continue
        meshed.append((c, mesh))

    slot_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for c, _ in meshed:
        h, w = c.record.rgba.shape[:2]
        x0, y0 = c.record.offset
        box = (float(x0), float(y0), float(x0 + w), float(y0 + h))
        if c.slot in slot_bboxes:
            e = slot_bboxes[c.slot]
            box = (min(e[0], box[0]), min(e[1], box[1]),
                   max(e[2], box[2]), max(e[3], box[3]))
        slot_bboxes[c.slot] = box

    bones = fit_skeleton(slot_bboxes, canvas_size)  # raises CompileError if critical parts missing

    layer_slots = {c.record.name: c.slot for c, _ in meshed}
    eye_whites = {
        side: next(
            (c.record.name for c, _ in meshed if c.slot == f"eye_white_{side}"), None
        )
        for side in ("l", "r")
    }

    layers: list[LayerMesh] = []
    for c, mesh in meshed:
        canvas_verts = mesh.vertices + np.array(c.record.offset, dtype=np.float32)
        chain = SLOT_TO_CHAIN.get(c.slot, ["spine"])
        indices, weights = assign_weights(canvas_verts, chain, bones)
        clip_to = None
        for side in ("l", "r"):
            if c.slot == f"iris_{side}" and eye_whites[side]:
                clip_to = eye_whites[side]
        layers.append(LayerMesh(
            layer_name=c.record.name,
            part_slot=c.slot,
            z_index=c.record.z_index,
            texture=c.record.rgba,
            offset=c.record.offset,
            vertices=canvas_verts,
            uvs=mesh.uvs,
            triangles=mesh.triangles,
            bone_indices=indices,
            bone_weights=weights,
            clip_to=clip_to,
        ))
    layers.sort(key=lambda l: l.z_index)

    deformers = build_deformers(layer_slots, slot_bboxes, canvas_size)
    return Rig(canvas_size=canvas_size, layers=layers, bones=bones,
               deformers=deformers)


def compile_psd(
    psd_path: str | Path, overrides_path: str | Path | None = None
) -> Rig:
    records, canvas_size = load_layers(psd_path)
    overrides = load_overrides(overrides_path) if overrides_path else None
    return compile_from_records(records, canvas_size, overrides)


def rig_summary(rig: Rig) -> dict:
    return {
        "canvas_size": list(rig.canvas_size),
        "layers": [
            {
                "name": l.layer_name,
                "slot": l.part_slot,
                "z": l.z_index,
                "clip_to": l.clip_to,
                "n_vertices": int(l.vertices.shape[0]),
                "n_triangles": int(l.triangles.shape[0]),
            }
            for l in rig.layers
        ],
        "bones": sorted(b.name for b in rig.bones),
        # lists, not tuples: the summary must be identical after a JSON round
        # trip because the golden test compares against json.loads output
        "deformers": sorted([d.kind, d.param] for d in rig.deformers),
    }


def _print_report(rig: Rig, out_path: Path) -> None:
    s = rig_summary(rig)
    print(f"rAig rig: {out_path}")
    print(f"canvas: {s['canvas_size'][0]}x{s['canvas_size'][1]}")
    print(f"{'layer':<16} {'slot':<14} {'z':>3} {'verts':>6} {'tris':>6}  clip_to")
    for l in s["layers"]:
        print(f"{l['name']:<16} {l['slot']:<14} {l['z']:>3} "
              f"{l['n_vertices']:>6} {l['n_triangles']:>6}  {l['clip_to'] or '-'}")
    print(f"bones ({len(s['bones'])}): {', '.join(s['bones'])}")
    print("deformers: " + ", ".join(f"{k}[{p}]" for k, p in s["deformers"]))


def compile_command(args) -> int:
    psd_path = Path(args.psd)
    out_path = Path(args.output) if args.output else psd_path.with_suffix(".raig")
    try:
        rig = compile_psd(psd_path, args.overrides)
    except CompileError as e:
        print(f"compile failed: {e}", file=sys.stderr)
        return 1
    save_rig(rig, out_path)
    _print_report(rig, out_path)
    return 0
