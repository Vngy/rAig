# rAig

End-to-end auto-rigging for 2D vtuber avatars: layered PSD in → avatar moving
in a preview window from single-webcam full-body tracking.

## Quickstart

```bash
uv sync

# 1. Compile a layered PSD into a rig
uv run raig compile model.psd -o model.raig

# 2. Inspect what the compiler saw (part classification, meshes, skeleton)
uv run raig debug model.raig -o debug_sheet

# 3. Live preview from your webcam
uv run raig run model.raig
```

No webcam handy? Drive it with a synthetic motion session:

```bash
uv run python -m raig.tracking.synth demo.jsonl
uv run raig run model.raig --replay demo.jsonl
```

## Live controls

- `C` — recalibrate: sit in your neutral pose for ~1 s; saved next to the rig
- `G` — toggle chroma-green / dark background (green keys out in OBS)

OBS: add a Window Capture of the rAig window and a chroma-key filter.

## When classification gets a layer wrong

Create `overrides.toml` and pin layers to slots, then recompile:

```toml
[slots]
"mystery_layer_7" = "hair_front"
"heart_eyes_variant" = "exclude"
```

```bash
uv run raig compile model.psd --overrides overrides.toml
```

Valid slots: hair_front, hair_back, face, eye_white_l/r, iris_l/r, brow_l/r,
mouth, head_misc, torso, arm_l/r, leg_l/r, misc.

The special value `exclude` drops a layer from the rig entirely — real PSDs
often ship always-visible expression variants (heart eyes, blush, alternate
mouths) that shouldn't render.

## Development

```bash
uv run pytest              # unit + golden + replay tests
uv run pytest -m perf      # 60 fps budget check
uv run pytest -m integration  # needs MediaPipe models (downloads ~40 MB)
```

Spec: `docs/superpowers/specs/2026-07-12-raig-mvp-design.md`.
