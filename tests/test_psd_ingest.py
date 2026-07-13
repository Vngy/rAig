import numpy as np

from raig.compiler.psd_ingest import load_layers


def test_loads_all_pixel_layers(mini_psd_path):
    records, canvas = load_layers(mini_psd_path)
    assert canvas == (1024, 1024)
    names = {r.name for r in records}
    assert names == {
        "hair_front", "iris_l", "eye_white_l", "iris_r", "eye_white_r",
        "mouth", "face", "hair_back",
        "arm_l", "arm_r", "torso", "leg_l", "leg_r",
    }


def test_group_paths(mini_psd_path):
    records, _ = load_layers(mini_psd_path)
    by_name = {r.name: r for r in records}
    assert by_name["face"].group_path == ("head",)
    assert by_name["torso"].group_path == ("body",)


def test_rgba_and_offset(mini_psd_path):
    records, _ = load_layers(mini_psd_path)
    by_name = {r.name: r for r in records}
    face = by_name["face"]
    assert face.rgba.dtype == np.uint8
    assert face.rgba.shape[2] == 4
    assert face.rgba[..., 3].max() == 255
    left, top = face.offset
    assert 380 <= left <= 400 and 120 <= top <= 140  # near (392, 130)


def test_z_order_is_draw_order(mini_psd_path):
    records, _ = load_layers(mini_psd_path)
    z = {r.name: r.z_index for r in records}
    # painter order semantics: bigger z draws on top
    assert z["iris_l"] > z["eye_white_l"] > z["face"] > z["hair_back"]
    assert z["hair_front"] > z["face"]
    assert z["torso"] > z["leg_l"]
    assert z["face"] > z["torso"]  # head group is above body group
    assert sorted(z.values()) == list(range(13))
