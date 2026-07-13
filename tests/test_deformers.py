from raig.compiler.deformers import build_deformers

CANVAS = (1024, 1024)

LAYER_SLOTS = {
    "hair_front": "hair_front", "hair_back": "hair_back", "face": "face",
    "iris_l": "iris_l", "eye_white_l": "eye_white_l",
    "iris_r": "iris_r", "eye_white_r": "eye_white_r",
    "mouth": "mouth", "torso": "torso", "arm_l": "arm_l", "arm_r": "arm_r",
    "leg_l": "leg_l", "leg_r": "leg_r",
}

SLOT_BBOXES = {
    "hair_front": (382.0, 120.0, 642.0, 260.0),
    "hair_back": (362.0, 90.0, 662.0, 410.0),
    "face": (392.0, 130.0, 632.0, 390.0),
    "iris_l": (452.0, 235.0, 472.0, 255.0),
    "eye_white_l": (436.0, 229.0, 488.0, 261.0),
    "iris_r": (552.0, 235.0, 572.0, 255.0),
    "eye_white_r": (536.0, 229.0, 588.0, 261.0),
    "mouth": (490.0, 305.0, 534.0, 325.0),
    "torso": (400.0, 380.0, 624.0, 660.0),
    "arm_l": (330.0, 400.0, 400.0, 640.0),
    "arm_r": (624.0, 400.0, 694.0, 640.0),
    "leg_l": (430.0, 640.0, 500.0, 980.0),
    "leg_r": (524.0, 640.0, 594.0, 980.0),
}


def by_kind_param(deformers):
    return {(d.kind, d.param): d for d in deformers}


def test_full_slot_set_produces_all_kinds():
    d = by_kind_param(build_deformers(LAYER_SLOTS, SLOT_BBOXES, CANVAS))
    assert ("eye_blink", "eye_l_open") in d
    assert ("eye_blink", "eye_r_open") in d
    assert ("mouth_open", "mouth_open") in d
    assert ("head_warp", "head_angle_x") in d
    assert ("body_shift", "body_angle_x") in d


def test_eye_blink_targets_and_center():
    d = by_kind_param(build_deformers(LAYER_SLOTS, SLOT_BBOXES, CANVAS))
    blink_l = d[("eye_blink", "eye_l_open")]
    assert set(blink_l.layer_names) == {"iris_l", "eye_white_l"}
    assert 229.0 <= blink_l.payload["center_y"] <= 261.0
    assert blink_l.payload["min_scale"] == 0.05


def test_head_warp_depth_parallax():
    d = by_kind_param(build_deformers(LAYER_SLOTS, SLOT_BBOXES, CANVAS))
    warp = d[("head_warp", "head_angle_x")]
    depth = warp.payload["depth"]
    assert depth["hair_front"] > depth["face"] > depth["hair_back"]
    assert "torso" not in depth
    assert set(warp.layer_names) >= {"face", "hair_front", "hair_back", "mouth"}


def test_body_shift_covers_body_layers():
    d = by_kind_param(build_deformers(LAYER_SLOTS, SLOT_BBOXES, CANVAS))
    shift = d[("body_shift", "body_angle_x")]
    assert set(shift.layer_names) == {"torso", "arm_l", "arm_r", "leg_l", "leg_r"}
    assert shift.payload["depth"]["arm_l"] > shift.payload["depth"]["torso"]


def test_missing_parts_skip_deformers():
    slots = {"face": "face", "torso": "torso"}
    bboxes = {"face": SLOT_BBOXES["face"], "torso": SLOT_BBOXES["torso"]}
    kinds = {d.kind for d in build_deformers(slots, bboxes, CANVAS)}
    assert "eye_blink" not in kinds
    assert "mouth_open" not in kinds
    assert "head_warp" in kinds and "body_shift" in kinds
