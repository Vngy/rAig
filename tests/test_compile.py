import json
from pathlib import Path

import numpy as np
import pytest

from raig.cli import main
from raig.compiler.compile import compile_from_records, compile_psd, rig_summary
from raig.compiler.errors import CompileError
from raig.compiler.psd_ingest import LayerRecord
from raig.core.rig import load_rig

GOLDEN = Path(__file__).parent / "golden" / "mini_model_summary.json"


def solid_record(name, group_path, z, offset, size, color=(200, 200, 200)):
    w, h = size
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = color
    rgba[..., 3] = 255
    return LayerRecord(name=name, group_path=group_path, z_index=z,
                       offset=offset, rgba=rgba)


def test_compile_fixture_psd_structure(mini_psd_path):
    rig = compile_psd(mini_psd_path)
    assert rig.canvas_size == (1024, 1024)
    assert len(rig.layers) == 13
    assert [l.z_index for l in rig.layers] == sorted(l.z_index for l in rig.layers)
    bone_names = {b.name for b in rig.bones}
    assert bone_names == {
        "hips", "spine", "chest", "neck", "head",
        "arm_upper_l", "arm_lower_l", "hand_l",
        "arm_upper_r", "arm_lower_r", "hand_r",
        "leg_upper_l", "leg_lower_l", "leg_upper_r", "leg_lower_r",
    }
    kinds = {(d.kind, d.param) for d in rig.deformers}
    assert {("eye_blink", "eye_l_open"), ("eye_blink", "eye_r_open"),
            ("mouth_open", "mouth_open"), ("head_warp", "head_angle_x"),
            ("body_shift", "body_angle_x")} <= kinds
    by_name = {l.layer_name: l for l in rig.layers}
    assert by_name["iris_l"].clip_to == "eye_white_l"
    assert by_name["iris_r"].clip_to == "eye_white_r"
    assert by_name["face"].clip_to is None
    for layer in rig.layers:
        assert layer.vertices.shape[0] >= 3
        assert layer.triangles.shape[0] >= 1
        bound = layer.bone_indices[:, 0] >= 0
        assert bound.all(), f"{layer.layer_name} has unbound vertices"


def test_compile_without_torso_fails_loudly():
    records = [solid_record("face", ("head",), 0, (392, 130), (240, 260))]
    with pytest.raises(CompileError, match="torso"):
        compile_from_records(records, (1024, 1024))


def test_iris_clips_to_largest_eye_white():
    # lashes/eyeliner also classify to eye_white_*; the iris must clip to the
    # sclera (largest opaque footprint), not whichever comes first in z-order
    records = [
        solid_record("face", ("head",), 0, (392, 130), (240, 260)),
        solid_record("torso", ("body",), 1, (362, 390), (300, 400)),
        solid_record("eyeliner_L", ("head",), 2, (430, 200), (40, 6)),   # sliver, first
        solid_record("eye_white_L", ("head",), 3, (425, 195), (60, 40)),  # sclera
        solid_record("iris_L", ("head",), 4, (440, 205), (24, 24)),
    ]
    rig = compile_from_records(records, (1024, 1024))
    by_name = {l.layer_name: l for l in rig.layers}
    assert by_name["iris_L"].clip_to == "eye_white_L"


def test_override_exclude_drops_layer(capsys):
    records = [
        solid_record("face", ("head",), 0, (392, 130), (240, 260)),
        solid_record("torso", ("body",), 1, (362, 390), (300, 400)),
        solid_record("sparkle_effect", ("head",), 2, (392, 130), (240, 260)),
    ]
    rig = compile_from_records(
        records, (1024, 1024), overrides={"sparkle_effect": "exclude"}
    )
    names = {l.layer_name for l in rig.layers}
    assert "sparkle_effect" not in names
    assert {"face", "torso"} <= names
    assert "excluding layer 'sparkle_effect'" in capsys.readouterr().err


def test_empty_layer_skipped_with_warning(capsys):
    records = [
        solid_record("face", ("head",), 0, (392, 130), (240, 260)),
        solid_record("torso", ("body",), 1, (400, 380), (224, 280)),
        LayerRecord("empty_fx", ("body",), 2, (0, 0),
                    np.zeros((16, 16, 4), dtype=np.uint8)),
    ]
    rig = compile_from_records(records, (1024, 1024))
    assert {l.layer_name for l in rig.layers} == {"face", "torso"}
    assert "empty_fx" in capsys.readouterr().err


def test_golden_summary(mini_psd_path):
    rig = compile_psd(mini_psd_path)
    expected = json.loads(GOLDEN.read_text())
    assert rig_summary(rig) == expected


def test_cli_compile_writes_rig(mini_psd_path, tmp_path, capsys):
    out = tmp_path / "mini.raig"
    code = main(["compile", str(mini_psd_path), "-o", str(out)])
    assert code == 0
    assert out.exists()
    loaded = load_rig(out)
    assert len(loaded.layers) == 13
    report = capsys.readouterr().out
    assert "torso" in report and "head_warp" in report
