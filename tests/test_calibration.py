from raig.core.params import make_param_frame
from raig.tracking.calibration import (
    Calibration,
    collect,
    load_calibration,
    save_calibration,
    sidecar_path,
)


def test_collect_means_offset_params_only():
    frames = [
        make_param_frame({"head_angle_x": 4.0, "eye_l_open": 0.9,
                          "arm_upper_l_rot": 10.0}, timestamp=0.0),
        make_param_frame({"head_angle_x": 6.0, "eye_l_open": 0.8,
                          "arm_upper_l_rot": 20.0}, timestamp=0.1),
    ]
    cal = collect(frames)
    assert cal.offsets["head_angle_x"] == 5.0
    assert cal.offsets["arm_upper_l_rot"] == 15.0
    assert "eye_l_open" not in cal.offsets


def test_apply_subtracts_offsets_passes_rest():
    cal = Calibration(offsets={"head_angle_x": 5.0})
    out = cal.apply({"head_angle_x": 7.0, "eye_l_open": 0.5})
    assert out["head_angle_x"] == 2.0
    assert out["eye_l_open"] == 0.5


def test_sidecar_round_trip(tmp_path):
    rig_path = tmp_path / "model.raig"
    cal = Calibration(offsets={"body_angle_z": -3.5})
    save_calibration(cal, rig_path)
    assert sidecar_path(rig_path).exists()
    loaded = load_calibration(rig_path)
    assert loaded.offsets == {"body_angle_z": -3.5}


def test_load_missing_returns_none(tmp_path):
    assert load_calibration(tmp_path / "nope.raig") is None
