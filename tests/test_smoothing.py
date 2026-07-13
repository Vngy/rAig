import numpy as np

from raig.tracking.smoothing import FilterBank, OneEuroFilter


def test_first_sample_passes_through():
    f = OneEuroFilter()
    assert f(5.0, 0.0) == 5.0


def test_constant_signal_stays_constant():
    f = OneEuroFilter()
    for i in range(100):
        out = f(3.3, i / 60.0)
    assert abs(out - 3.3) < 1e-6


def test_jitter_is_attenuated():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    raw, smoothed = [], []
    for i in range(300):
        x = 10.0 + rng.normal(0, 0.5)
        raw.append(x)
        smoothed.append(f(x, i / 60.0))
    assert np.var(smoothed[50:]) < 0.25 * np.var(raw[50:])


def test_fast_motion_tracks_quickly():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    out = 0.0
    for i in range(30):  # 0.5 s of a big step
        out = f(100.0 if i > 0 else 0.0, i / 60.0)
    assert out > 85.0  # within 15% after half a second


def test_filter_bank_smooths_per_param():
    bank = FilterBank()
    out = {}
    for i in range(60):
        out = bank.apply({"a": 1.0, "b": float(i % 2)}, i / 60.0)
    assert abs(out["a"] - 1.0) < 1e-6
    assert 0.0 < out["b"] < 1.0  # alternating signal lands in between
