import time

import moderngl_window as mglw

from raig.core.params import ParamFrame, rest_frame
from raig.core.rig import load_rig
from raig.render.renderer import CHROMA_GREEN, RigRenderer
from raig.tracking.bundle import LandmarkBundle, load_replay
from raig.tracking.calibration import collect, load_calibration, save_calibration
from raig.tracking.mapper import Mapper

_DARK_GRAY = (0.1, 0.1, 0.1, 1.0)
_CALIB_FRAMES = 60


class ReplaySource:
    def __init__(self, bundles: list[LandmarkBundle], mapper: Mapper,
                 clock=time.monotonic):
        self._bundles = bundles
        self.mapper = mapper
        self._clock = clock
        self._start = clock()
        self._idx = 0
        self._prev_t = -1.0
        self._frame: ParamFrame | None = None
        self.status = "ok"

    def latest(self) -> ParamFrame | None:
        # loop period extends one frame-gap past the last timestamp so the
        # final bundle actually plays before wrapping
        if len(self._bundles) > 1:
            gap = self._bundles[-1].timestamp - self._bundles[-2].timestamp
        else:
            gap = 1.0 / 60.0
        duration = max(self._bundles[-1].timestamp + max(gap, 1e-6), 1.0 / 60.0)
        target = (self._clock() - self._start) % duration
        if target < self._prev_t:  # looped
            self._idx = 0
        self._prev_t = target
        while (self._idx < len(self._bundles)
               and self._bundles[self._idx].timestamp <= target):
            self._frame = self.mapper.map(self._bundles[self._idx])
            self._idx += 1
        if self._frame is None and self._bundles:  # before first timestamp
            self._frame = self.mapper.map(self._bundles[0])
            self._idx = max(self._idx, 1)
        return self._frame

    def stop(self) -> None:
        self.status = "stopped"


class PreviewWindow(mglw.WindowConfig):
    title = "rAig"
    gl_version = (3, 3)
    window_size = (768, 768)
    resizable = True
    vsync = True

    rig = None  # set by run_command before launch
    source = None
    rig_path: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.renderer = RigRenderer(self.ctx, self.rig)
        self.background = CHROMA_GREEN
        self._calib: list[ParamFrame] | None = None
        self._frames = 0
        self._fps_t0 = time.monotonic()

    def on_render(self, t: float, frame_time: float):
        # never-crash guard: a malformed bundle (replay) or a mapping/render
        # hiccup must never take down the window — fall back to rest_frame()
        # for mapping failures, and skip the draw entirely on render failures.
        try:
            frame = self.source.latest() or rest_frame()
        except Exception as e:
            print(f"tracking error (frame skipped): {e}")
            frame = rest_frame()

        if self._calib is not None:
            self._calib.append(frame)
            if len(self._calib) >= _CALIB_FRAMES:
                cal = collect(self._calib)
                if self.rig_path:
                    save_calibration(cal, self.rig_path)
                self.source.mapper.calibration = cal
                self._calib = None

        try:
            self.renderer.render_frame(frame, self.ctx.screen, self.background)
        except Exception as e:
            print(f"render error (frame skipped): {e}")

        self._frames += 1
        if self._frames % 30 == 0:
            now = time.monotonic()
            fps = 30.0 / max(now - self._fps_t0, 1e-6)
            self._fps_t0 = now
            calibrating = " CALIBRATING" if self._calib is not None else ""
            self.wnd.title = f"rAig  {fps:.0f} fps  [{self.source.status}]{calibrating}"

    # moderngl-window < 3.0 calls `render`; >= 3.0 calls `on_render`
    def render(self, t: float, frame_time: float):
        self.on_render(t, frame_time)

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action != keys.ACTION_PRESS:
            return
        if key == keys.C:
            self.source.mapper.calibration = None  # collect raw frames
            self._calib = []
        elif key == keys.G:
            self.background = (
                _DARK_GRAY if self.background == CHROMA_GREEN else CHROMA_GREEN
            )

    def key_event(self, key, action, modifiers):
        self.on_key_event(key, action, modifiers)


def run_command(args) -> int:
    rig = load_rig(args.rig)
    if args.replay:
        source = ReplaySource(load_replay(args.replay), Mapper())
    else:
        from raig.tracking.camera import Camera, TrackerThread
        from raig.tracking.landmarks import LandmarkExtractor
        from raig.tracking.models import ensure_models

        extractor = LandmarkExtractor(ensure_models())
        mapper = Mapper(calibration=load_calibration(args.rig))
        source = TrackerThread(Camera(args.camera), extractor, mapper)
        source.start()

    PreviewWindow.rig = rig
    PreviewWindow.source = source
    PreviewWindow.rig_path = args.rig
    try:
        mglw.run_window_config(PreviewWindow, args=[])
    finally:
        source.stop()
    return 0
