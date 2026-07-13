import numpy as np

from raig.core.rig import Bone, Deformer, LayerMesh, Rig, load_rig, save_rig


def tiny_rig() -> Rig:
    tex = np.zeros((4, 4, 4), dtype=np.uint8)
    tex[..., 0] = 255
    tex[..., 3] = 255
    layer = LayerMesh(
        layer_name="torso",
        part_slot="torso",
        z_index=0,
        texture=tex,
        offset=(10, 20),
        vertices=np.array([[10, 20], [14, 20], [12, 24]], dtype=np.float32),
        uvs=np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        bone_indices=np.array([[0, -1]] * 3, dtype=np.int32),
        bone_weights=np.array([[1.0, 0.0]] * 3, dtype=np.float32),
        clip_to=None,
    )
    bone = Bone(
        name="hips",
        parent=None,
        head=np.array([12.0, 24.0], dtype=np.float32),
        tail=np.array([12.0, 20.0], dtype=np.float32),
    )
    deformer = Deformer(
        kind="mouth_open",
        param="mouth_open",
        layer_names=["torso"],
        payload={"anchor_y": 20.0, "max_stretch": 0.8},
    )
    return Rig(canvas_size=(64, 64), layers=[layer], bones=[bone], deformers=[deformer])


def test_round_trip(tmp_path):
    rig = tiny_rig()
    path = tmp_path / "tiny.raig"
    save_rig(rig, path)
    loaded = load_rig(path)

    assert loaded.canvas_size == (64, 64)
    assert len(loaded.layers) == 1
    lay, orig = loaded.layers[0], rig.layers[0]
    assert lay.layer_name == "torso"
    assert lay.part_slot == "torso"
    assert lay.z_index == 0
    assert lay.offset == (10, 20)
    assert lay.clip_to is None
    np.testing.assert_array_equal(lay.texture, orig.texture)
    np.testing.assert_array_equal(lay.vertices, orig.vertices)
    np.testing.assert_array_equal(lay.uvs, orig.uvs)
    np.testing.assert_array_equal(lay.triangles, orig.triangles)
    np.testing.assert_array_equal(lay.bone_indices, orig.bone_indices)
    np.testing.assert_array_equal(lay.bone_weights, orig.bone_weights)

    assert loaded.bones[0].name == "hips"
    assert loaded.bones[0].parent is None
    np.testing.assert_array_equal(loaded.bones[0].head, rig.bones[0].head)

    d = loaded.deformers[0]
    assert d.kind == "mouth_open"
    assert d.param == "mouth_open"
    assert d.layer_names == ["torso"]
    assert d.payload["max_stretch"] == 0.8


def test_layers_sorted_by_z_on_load(tmp_path):
    rig = tiny_rig()
    l2 = LayerMesh(**{**rig.layers[0].__dict__, "layer_name": "front", "z_index": 5})
    l0 = rig.layers[0]
    rig_unsorted = Rig(
        canvas_size=rig.canvas_size,
        layers=[l2, l0],
        bones=rig.bones,
        deformers=[],
    )
    path = tmp_path / "z.raig"
    save_rig(rig_unsorted, path)
    loaded = load_rig(path)
    assert [l.z_index for l in loaded.layers] == [0, 5]
