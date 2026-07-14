import numpy as np

from raig.tracking.bundle import LandmarkBundle
from raig.tracking.mapper import Mapper
from raig.render.window import ReplaySource


def bundles_ramp():
    return [
        LandmarkBundle(t / 3.0, face_blendshapes={"jawOpen": v})
        for t, v in ((0, 0.0), (1, 0.5), (2, 1.0))
    ]


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_replay_source_paces_and_loops():
    clock = FakeClock()
    src = ReplaySource(bundles_ramp(), Mapper(smooth=False), clock=clock)
    assert src.status == "ok"

    clock.t = 100.05  # inside first bundle window
    f = src.latest()
    assert abs(f.values["mouth_open"] - 0.0) < 1e-6

    clock.t = 100.4  # second bundle reached
    assert abs(src.latest().values["mouth_open"] - 0.5) < 1e-6

    clock.t = 100.7  # third
    assert abs(src.latest().values["mouth_open"] - 1.0) < 1e-6

    clock.t = 101.05  # wrapped: back to the first bundle
    assert abs(src.latest().values["mouth_open"] - 0.0) < 1e-6


def test_replay_source_single_bundle_no_crash():
    src = ReplaySource(
        [LandmarkBundle(0.0, face_blendshapes={"jawOpen": 1.0})],
        Mapper(smooth=False),
        clock=FakeClock(),
    )
    f = src.latest()
    assert f is not None and abs(f.values["mouth_open"] - 1.0) < 1e-6
