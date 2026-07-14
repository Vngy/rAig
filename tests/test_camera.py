import time

import numpy as np

from raig.tracking.bundle import LandmarkBundle
from raig.tracking.camera import TrackerThread
from raig.tracking.mapper import Mapper


class FakeCamera:
    def __init__(self, ok=True):
        self.ok = ok
        self.t = 0.0

    def read(self):
        if not self.ok:
            return False, None, 0.0
        self.t += 1 / 60
        return True, np.zeros((48, 64, 3), np.uint8), self.t

    def release(self):
        pass


class RaisingCamera:
    def read(self):
        raise RuntimeError("device disconnected")

    def release(self):
        pass


class FakeExtractor:
    def extract(self, frame_rgb, timestamp_ms):
        return LandmarkBundle(
            timestamp=timestamp_ms / 1000.0,
            face_blendshapes={"jawOpen": 0.5},
        )


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_tracker_thread_produces_frames():
    t = TrackerThread(FakeCamera(), FakeExtractor(), Mapper(smooth=False))
    t.start()
    assert wait_for(lambda: t.latest() is not None)
    frame = t.latest()
    assert abs(frame.values["mouth_open"] - 0.5) < 1e-6
    assert t.status == "ok"
    t.stop()
    assert not t.is_alive()
    assert t.status == "stopped"


def test_tracker_thread_reports_camera_loss():
    t = TrackerThread(FakeCamera(ok=False), FakeExtractor(), Mapper(smooth=False))
    t.start()
    assert wait_for(lambda: t.status == "no_camera")
    assert t.latest() is None
    t.stop()
    assert not t.is_alive()
    assert t.status == "stopped"


def test_tracker_thread_survives_read_exception(capfd):
    t = TrackerThread(RaisingCamera(), FakeExtractor(), Mapper(smooth=False))
    t.start()
    assert wait_for(lambda: t.status == "no_camera")
    assert t.is_alive()
    assert t.latest() is None
    # Let a few more 0.1s poll cycles elapse while still down: the
    # ok->no_camera transition log must fire once, not once per iteration.
    time.sleep(0.35)
    t.stop()
    assert not t.is_alive()
    assert t.status == "stopped"
    err = capfd.readouterr().err
    assert err.count("camera read failed") == 1
