import sys
import threading
import time

import cv2
import numpy as np

from raig.core.params import ParamFrame


class Camera:
    def __init__(self, index: int = 0):
        self._cap = cv2.VideoCapture(index)

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            return False, None, time.monotonic()
        return True, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), time.monotonic()

    def release(self) -> None:
        self._cap.release()


class TrackerThread(threading.Thread):
    def __init__(self, camera, extractor, mapper):
        super().__init__(daemon=True)
        self._camera = camera
        self._extractor = extractor
        self.mapper = mapper  # public: window swaps calibration on it
        self._latest: ParamFrame | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.status = "ok"

    def latest(self) -> ParamFrame | None:
        with self._lock:
            return self._latest

    def run(self) -> None:
        t0 = time.monotonic()
        while not self._stop_event.is_set():
            try:
                ok, frame_rgb, _ = self._camera.read()
            except Exception:  # lenient runtime: never crash the loop
                ok, frame_rgb = False, None
            if not ok:
                if self.status != "no_camera":
                    # Log once on the transition, not per-iteration: this
                    # loop polls at 10 Hz and a disconnected camera would
                    # otherwise spam stderr for as long as it stays down.
                    print("camera read failed: no frame (camera disconnected?)",
                          file=sys.stderr)
                self.status = "no_camera"
                self._stop_event.wait(0.1)
                continue
            timestamp_ms = int((time.monotonic() - t0) * 1000)
            try:
                bundle = self._extractor.extract(frame_rgb, timestamp_ms)
                frame = self.mapper.map(bundle)
            except Exception as e:  # lenient runtime: never crash the loop
                print(f"tracking error (skipped frame): {e}")
                continue
            with self._lock:
                self._latest = frame
            self.status = "ok"
        self._camera.release()
        self.status = "stopped"

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=2.0)
