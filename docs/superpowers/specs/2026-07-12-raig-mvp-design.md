# rAig MVP Design — Auto-Rigging Pipeline for 2D VTuber Avatars

**Date:** 2026-07-12
**Status:** Approved by user
**Repo:** `rAig` (Python 3.11+, MIT)

## Problem

Rigging a 2D vtuber avatar (Live2D-style) is a slow, expert, manual process: an
artist's layered illustration must be cut apart, meshed, given deformers and
parameters, and bound to tracking. rAig automates this end to end: **layered
PSD in → avatar moving on screen from webcam tracking**, with no manual rigging
step.

## Constraints discovered during research

- **Live2D's `.moc3`/`.cmo3` formats are proprietary and closed.** Only Cubism
  Editor writes them; it has no automation API; third parties cannot legally
  generate them. rAig therefore uses **its own rig format and runtime renderer**
  rather than emitting Live2D files. (Inochi2D export is a possible future
  milestone, not MVP.)
- **Pose2Sim** (openpose-to-OpenSim, multi-camera markerless mocap) is
  research-grade but requires 2+ calibrated cameras — too heavy for a
  desk-streaming MVP. Single-webcam MediaPipe tracking is the MVP choice;
  Pose2Sim can slot in later behind the same interface (see ParamFrame).

## MVP scope decisions (user-approved)

| Decision | Choice |
|---|---|
| Output target | Own rig format + own runtime renderer (no .moc3, no Inochi2D for MVP) |
| Input | Layered PSD, already cut by an artist (flat-image segmentation is post-MVP) |
| Tracking | One webcam, full body: MediaPipe face + pose + hands |
| Success bar | One specific model works end to end (robustness to arbitrary PSDs is post-MVP) |
| Delivery | Live preview window (OBS captures the window; no native OBS integration) |
| Architecture | All-Python monolith with an internal ParamFrame seam for future splitting |

**MVP acceptance test:** `raig compile model.psd` then `raig run model.raig`
on the target model produces an avatar that follows the user's face
(expressions, head turn), arms, hands, torso, and legs-when-in-frame, at 60fps,
in a window OBS can capture.

## Architecture

Two phases sharing one core data model, the **Rig**:

```
  OFFLINE (once per model)                 LIVE (every frame, 60fps target)
┌────────────┐   ┌────────────┐         ┌──────────┐   ┌────────────┐   ┌──────────┐
│ PSD Ingest │ → │  Auto-Rig  │ → Rig → │ Tracking │ → │ ParamFrame │ → │ Renderer │
└────────────┘   └────────────┘  (file) └──────────┘   └────────────┘   └──────────┘
 psd-tools        mesh + skeleton         MediaPipe      landmark→param    ModernGL
 layer parse      + weights + params      (webcam)       mapping           preview window
```

- The **offline compiler** runs once per model and writes a `.raig` file:
  JSON manifest + mesh/weight arrays + extracted layer textures. Rig bugs are
  debuggable without a webcam; rigs load instantly at stream start.
- The **live runtime** loads a `.raig`, opens the webcam, and renders. Tracking
  output is normalized into a **ParamFrame** — a flat dict of named floats
  (`head_angle_x: -12.3`, `eye_l_open: 0.8`, `arm_r_swing: 0.4`). The renderer
  consumes only ParamFrames, never raw landmarks. This seam is what lets
  Pose2Sim, VMC input, or recorded motion replace the webcam later without
  touching rig or renderer.
- The **Rig** holds: per-part triangulated meshes, a 2D skeleton with skin
  weights, and a parameter table mapping each named param to mesh deformations.

**Dependencies:** mediapipe, psd-tools, numpy, scipy, opencv-python, triangle,
moderngl, moderngl-window. Managed with `uv`.

**CLI surface (the whole MVP):**

- `raig compile model.psd -o model.raig` — build the rig
- `raig run model.raig` — live preview window
- `raig debug model.raig` — write debug sheet PNGs

## Component 1: PSD ingest + auto-rig (offline compiler)

Four independently testable steps:

### 1a. Layer classification

Map each PSD layer to a semantic part slot (`hair_front`, `eye_L`, `iris_L`,
`mouth`, `torso`, `arm_upper_R`, `leg_L`, …).

1. **Name matching** against a multilingual keyword table (English + Japanese —
   artists name layers "bangs" or「前髪」).
2. **Geometry fallback** for unmatched layers: canvas position, size, z-order,
   overlap with already-classified parts.
3. Unmatched layers attach to their nearest classified ancestor group rather
   than being dropped.
4. **`overrides.toml` escape hatch:** a hand-written file pinning any layer to
   any slot. Guarantees the one-model success bar is never blocked by a
   misclassification.

### 1b. Mesh generation

Per part: extract alpha channel → OpenCV contour → simplify → constrained
Delaunay triangulation (`triangle`) with interior grid points. Mesh density is
per-slot: dense for deformation-heavy parts (hair, torso), coarse for rigid
parts (iris).

### 1c. Skeleton fit + skin weights

Fixed humanoid 2D template: hips → spine → chest → neck → head; 3-bone arms
plus hand tip; 2-bone legs. Bones are anchored to the bounding boxes of
classified parts. Skin weights per vertex: inverse-distance to the two nearest
bones of the part's assigned chain, then smoothed. Runtime deformation is 2D
linear blend skinning.

### 1d. Face parameter table

Face parts use procedural parameter deformers instead of bones (Live2D-style):

- `eye_l_open` / `eye_r_open` — eyelid mesh scales vertically over the iris;
  iris stencil-masked by the eye white
- `mouth_open` (+ `mouth_form` if blendshapes support it) — mouth mesh interpolation
- `head_angle_x/y/z` — spherical-ish warp of the whole head group to fake 3D turn
- `body_angle_x/y/z` — torso counter-rotation with per-layer parallax offsets

## Component 2: Tracking → ParamFrame mapping (live)

- **MediaPipe Tasks (Python)**, one webcam via OpenCV capture:
  `FaceLandmarker` (with built-in 52 blendshape scores), `PoseLandmarker`
  (33 landmarks), `HandLandmarker` (finger curl). Face and pose every frame;
  hands may run at half rate if CPU-bound.
- **Mapper:** blendshape scores map ~1:1 to face params (`eyeBlinkLeft` →
  `eye_l_open` inverted, `jawOpen` → `mouth_open`); head pose from the face
  transform matrix; body params are joint angles from pose landmarks
  (shoulder–elbow–wrist → arm bones, hip–knee–ankle → legs, shoulder-line tilt
  → `body_angle_z`).
- **Calibration:** on startup (and via hotkey), capture ~1s of rest pose; store
  per-param offsets and observed ranges in a sidecar next to the `.raig` so
  the user's neutral posture maps to the avatar's rest pose, persistently.
- **Smoothing:** One Euro filter on every param (low latency, low jitter).
- **Dropout handling:** when a landmark group is absent (legs out of frame,
  hand occluded), its params decay to rest pose over ~0.5s instead of
  snapping. This is what makes seated-at-desk degrade gracefully.
- **Threading:** tracking runs in its own thread writing ParamFrames to a
  latest-value slot (no queue buildup); the render loop reads the freshest.

## Component 3: Renderer

- ModernGL window via `moderngl-window`. One draw call per rig part: layer
  texture on deformed mesh, painter's-algorithm z-order from the PSD,
  premultiplied alpha, stencil masks for eye/mouth clipping.
- Vertex deformation (skinning + param deformers) in NumPy on CPU per frame.
  At vtuber mesh densities (thousands of vertices) this hits 60fps; GPU
  skinning is a later optimization, not an MVP need.
- Background: solid chroma green, toggleable, so OBS window capture keys it out.

## Error handling

- **Offline compiler is strict.** Missing critical parts (no head, no torso)
  fail loudly with a report of what was found and classified. A silent bad rig
  wastes downstream debugging time.
- **Live runtime is lenient.** Tracking dropout → decay to rest pose; webcam
  disconnect → avatar freezes with on-screen warning; malformed ParamFrame →
  skipped, never crashed on.
- **Debuggability:** `raig compile` prints a rig report (parts found, slots
  filled, vertex counts). `raig debug` dumps a debug sheet — each classified
  part rendered with mesh wireframe and bone overlay as PNGs — so
  misclassification is visible at a glance.

## Testing

- **Unit:** classification heuristics on synthetic layer lists; mesh invariants
  (no degenerate triangles, contour coverage); mapper math (known landmark
  sets → expected angles).
- **Golden files:** a small hand-made ~10-layer test PSD compiles to a rig
  snapshot diffed in CI.
- **Replay:** recorded landmark sessions (JSONL) drive mapper + renderer
  headless (offscreen rendering) to catch regressions without webcam or human.
- **Acceptance (manual):** the one target model, live, looking right.

## Out of scope for MVP (future milestones)

1. **Flat-image input:** AI segmentation + occlusion inpainting to cut an
   un-layered illustration (the "drop in a drawing" dream).
2. **Physics:** spring-damper follow-through on hair/accessory bones — likely
   the first post-MVP addition; large perceived-quality win.
3. **Pose2Sim / multi-camera tracking** behind the ParamFrame seam.
4. **VMC protocol input/output.**
5. **Inochi2D export** for portability into an existing ecosystem.
6. **OBS-native output** (virtual camera / Syphon / browser source with alpha).
7. **Robustness to arbitrary artist PSDs**, GUI, packaging/distribution.
