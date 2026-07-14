import math

from raig.core.params import (
    PARAM_SPECS,
    bone_rot_param,
    make_param_frame,
    rest_frame,
)


def test_registry_has_core_params_with_ranges():
    assert PARAM_SPECS["head_angle_x"].min == -30.0
    assert PARAM_SPECS["eye_l_open"].default == 1.0
    assert PARAM_SPECS["mouth_form"].min == -1.0
    assert PARAM_SPECS["body_angle_z"].max == 10.0


def test_clamping_registered_param():
    f = make_param_frame({"head_angle_x": 999.0, "eye_l_open": -3.0}, timestamp=1.0)
    assert f.values["head_angle_x"] == 30.0
    assert f.values["eye_l_open"] == 0.0
    assert f.timestamp == 1.0


def test_bone_rot_params_accepted_and_clamped():
    name = bone_rot_param("arm_upper_l")
    assert name == "arm_upper_l_rot"
    f = make_param_frame({name: -500.0})
    assert f.values[name] == -90.0


def test_unknown_params_dropped_silently():
    f = make_param_frame({"totally_bogus": 1.0, "mouth_open": 0.5})
    assert "totally_bogus" not in f.values
    assert f.values["mouth_open"] == 0.5


def test_nan_dropped_like_unknown_param():
    name = bone_rot_param("arm_upper_l")
    f = make_param_frame({"head_angle_x": math.nan, name: math.nan, "mouth_open": 0.5})
    assert "head_angle_x" not in f.values
    assert name not in f.values
    assert f.values["mouth_open"] == 0.5


def test_rest_frame_is_defaults():
    f = rest_frame()
    assert f.values["eye_r_open"] == 1.0
    assert f.values["mouth_open"] == 0.0
    assert not any(k.endswith("_rot") for k in f.values)
