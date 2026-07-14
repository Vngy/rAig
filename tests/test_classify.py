import numpy as np
import pytest

from raig.compiler.classify import SLOTS, classify_layers, load_overrides
from raig.compiler.psd_ingest import LayerRecord


def make_record(name, group_path=(), center=(512, 512), size=(40, 40), z=0):
    w, h = size
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    offset = (center[0] - w // 2, center[1] - h // 2)
    return LayerRecord(
        name=name, group_path=tuple(group_path), z_index=z, offset=offset, rgba=rgba
    )


CANVAS = (1024, 1024)


def classify_one(record, overrides=None):
    return classify_layers([record], CANVAS, overrides)[0]


def test_name_match_english():
    c = classify_one(make_record("hair_front"))
    assert c.slot == "hair_front" and c.source == "name"


def test_name_match_japanese():
    assert classify_one(make_record("前髪")).slot == "hair_front"
    assert classify_one(make_record("口")).slot == "mouth"


def test_side_from_name_token():
    assert classify_one(make_record("left_iris")).slot == "iris_l"
    assert classify_one(make_record("iris_r")).slot == "iris_r"


def test_side_from_geometry_when_unmarked():
    c = classify_one(make_record("iris", center=(300, 245)))
    assert c.slot == "iris_l"
    c = classify_one(make_record("iris", center=(700, 245)))
    assert c.slot == "iris_r"


def test_hand_maps_to_arm_slot():
    assert classify_one(make_record("hand_r")).slot == "arm_r"


def test_inherit_from_group():
    c = classify_one(make_record("sleeve_frill", group_path=("body", "arm_l")))
    assert c.slot == "arm_l" and c.source == "inherit"


def test_geometry_fallback_zones():
    c = classify_one(make_record("sparkle", center=(512, 200)))
    assert c.slot == "head_misc" and c.source == "geometry"
    c = classify_one(make_record("sparkle", center=(512, 900)))
    assert c.slot == "misc" and c.source == "geometry"


def test_override_wins(tmp_path):
    ov = tmp_path / "overrides.toml"
    ov.write_text('[slots]\nsparkle = "hair_front"\n')
    overrides = load_overrides(ov)
    c = classify_one(make_record("sparkle"), overrides)
    assert c.slot == "hair_front" and c.source == "override"


def test_override_unknown_slot_rejected(tmp_path):
    ov = tmp_path / "overrides.toml"
    ov.write_text('[slots]\nsparkle = "wings"\n')
    with pytest.raises(ValueError, match="wings"):
        load_overrides(ov)


def test_name_variants_match_underscore_keyword():
    c = classify_one(make_record("hair front"))
    assert c.slot == "hair_front" and c.source == "name"
    assert classify_one(make_record("hair-front")).slot == "hair_front"
    assert classify_one(make_record("hair_front")).slot == "hair_front"


def test_bare_eye_keyword_matches_eye_white_with_side():
    assert classify_one(make_record("eye_L")).slot == "eye_white_l"
    assert classify_one(make_record("左目")).slot == "eye_white_l"


def test_eyebrow_matches_brow_not_eye_white():
    # "eyebrow" contains "eye" as a substring; the brow rule must win
    # (it precedes eye_white in _BASE_RULES priority order).
    c = classify_one(make_record("eyebrow_L"))
    assert c.slot == "brow_l" and c.source == "name"


def test_all_emitted_slots_are_canonical():
    records = [
        make_record("前髪"), make_record("iris", center=(300, 245)),
        make_record("mystery", center=(512, 900)),
    ]
    for c in classify_layers(records, CANVAS):
        assert c.slot in SLOTS
