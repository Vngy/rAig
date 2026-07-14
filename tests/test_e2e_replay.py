import time

import numpy as np
import pytest

from raig.render.renderer import RigRenderer
from raig.tracking.mapper import Mapper
from raig.tracking.synth import make_session


def test_session_shape():
    bundles = make_session(n_frames=120)
    assert len(bundles) == 120
    assert bundles[1].timestamp > bundles[0].timestamp
    assert bundles[0].face_blendshapes is not None
    assert bundles[0].pose is not None and bundles[0].pose.shape == (33, 4)


def test_e2e_compile_replay_render(gl_ctx, mini_rig):
    bundles = make_session(n_frames=180)
    mapper = Mapper(smooth=True)
    renderer = RigRenderer(gl_ctx, mini_rig)
    fbo = gl_ctx.framebuffer(
        color_attachments=[gl_ctx.texture(mini_rig.canvas_size, 4)]
    )

    def snapshot():
        return np.frombuffer(fbo.read(components=4), dtype=np.uint8).copy()

    frame_times = []
    first = mid = None
    for i, b in enumerate(bundles):
        frame = mapper.map(b)
        t0 = time.perf_counter()
        renderer.render_frame(frame, fbo)
        gl_ctx.finish()
        frame_times.append(time.perf_counter() - t0)
        if i == 0:
            first = snapshot()
        if i == len(bundles) // 2:
            mid = snapshot()

    assert not np.array_equal(first, mid), "avatar did not visibly move"
    median = sorted(frame_times)[len(frame_times) // 2]
    print(f"\nmedian deform+render: {median * 1000:.2f} ms")
    assert median < 0.033, f"median frame {median * 1000:.1f} ms exceeds 33 ms floor"


@pytest.mark.perf
def test_meets_60fps_target(gl_ctx, mini_rig):
    renderer = RigRenderer(gl_ctx, mini_rig)
    fbo = gl_ctx.framebuffer(
        color_attachments=[gl_ctx.texture(mini_rig.canvas_size, 4)]
    )
    mapper = Mapper(smooth=True)
    times = []
    for b in make_session(n_frames=120):
        frame = mapper.map(b)
        t0 = time.perf_counter()
        renderer.render_frame(frame, fbo)
        gl_ctx.finish()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    assert median < 0.016, f"{median * 1000:.1f} ms/frame — misses the 60 fps spec target"
