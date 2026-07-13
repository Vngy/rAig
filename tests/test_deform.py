import numpy as np

from raig.core.params import make_param_frame, rest_frame
from raig.core.rig import Bone, Deformer, LayerMesh, Rig
from raig.render.deform import deform_rig, param_value


def make_layer(name, slot, verts, indices, weights, z=0):
    verts_arr = np.asarray(verts, np.float32)
    n = verts_arr.shape[0]
    tex = np.full((4, 4, 4), 255, dtype=np.uint8)
    return LayerMesh(
        layer_name=name, part_slot=slot, z_index=z, texture=tex, offset=(0, 0),
        vertices=verts_arr,
        uvs=np.zeros((n, 2), np.float32),
        triangles=np.zeros((max(n - 2, 1), 3), np.int32),
        bone_indices=np.asarray(indices, np.int32),
        bone_weights=np.asarray(weights, np.float32),
    )


def one_bone_rig():
    bone = Bone("arm_upper_l", None,
                np.array([0.0, 0.0], np.float32), np.array([0.0, 10.0], np.float32))
    layer = make_layer("arm", "arm_l",
                       [[0.0, 10.0], [50.0, 50.0]],
                       [[0, -1], [-1, -1]],
                       [[1.0, 0.0], [0.0, 0.0]])
    return Rig((100, 100), [layer], [bone], [])


def test_rest_frame_is_identity():
    rig = one_bone_rig()
    out = deform_rig(rig, rest_frame())
    np.testing.assert_allclose(out["arm"], rig.layers[0].vertices, atol=1e-4)


def test_single_bone_rotation_ccw():
    rig = one_bone_rig()
    out = deform_rig(rig, make_param_frame({"arm_upper_l_rot": 90.0}))
    np.testing.assert_allclose(out["arm"][0], [-10.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(out["arm"][1], [50.0, 50.0], atol=1e-6)  # unbound


def test_child_inherits_parent_rotation():
    parent = Bone("hips", None,
                  np.array([0.0, 0.0], np.float32), np.array([0.0, -10.0], np.float32))
    child = Bone("spine", "hips",
                 np.array([0.0, -10.0], np.float32), np.array([0.0, -20.0], np.float32))
    layer = make_layer("t", "torso", [[0.0, -20.0]], [[1, -1]], [[1.0, 0.0]])
    rig = Rig((100, 100), [layer], [parent, child], [])
    out = deform_rig(rig, make_param_frame({"hips_rot": 90.0}))
    np.testing.assert_allclose(out["t"][0], [20.0, 0.0], atol=1e-3)


def eye_rig(param="eye_l_open"):
    verts = [[10.0, 240.0], [10.0, 250.0], [10.0, 260.0]]
    layer = make_layer("eye_white_l", "eye_white_l", verts,
                       [[-1, -1]] * 3, [[0.0, 0.0]] * 3)
    d = Deformer("eye_blink", param, ["eye_white_l"],
                 {"center_y": 250.0, "min_scale": 0.05})
    return Rig((100, 300), [layer], [], [d])


def test_eye_blink_squashes_vertically():
    rig = eye_rig()
    closed = deform_rig(rig, make_param_frame({"eye_l_open": 0.0}))["eye_white_l"]
    span = closed[:, 1].max() - closed[:, 1].min()
    assert span <= 0.05 * 20.0 + 1e-3
    open_ = deform_rig(rig, make_param_frame({"eye_l_open": 1.0}))["eye_white_l"]
    np.testing.assert_allclose(open_[:, 1], [240.0, 250.0, 260.0], atol=1e-4)


def test_eye_blink_defaults_open_when_param_absent():
    rig = eye_rig()
    out = deform_rig(rig, make_param_frame({}))["eye_white_l"]
    np.testing.assert_allclose(out[:, 1], [240.0, 250.0, 260.0], atol=1e-4)


def test_mouth_stretches_down_from_anchor():
    layer = make_layer("mouth", "mouth", [[0.0, 305.0], [0.0, 325.0]],
                       [[-1, -1]] * 2, [[0.0, 0.0]] * 2)
    d = Deformer("mouth_open", "mouth_open", ["mouth"],
                 {"anchor_y": 305.0, "max_stretch": 0.8})
    rig = Rig((100, 400), [layer], [], [d])
    out = deform_rig(rig, make_param_frame({"mouth_open": 1.0}))["mouth"]
    np.testing.assert_allclose(out[0], [0.0, 305.0], atol=1e-4)  # anchor fixed
    np.testing.assert_allclose(out[1, 1], 305.0 + 20.0 * 1.8, atol=1e-3)


def test_head_warp_depth_parallax():
    common = dict(indices=[[-1, -1]], weights=[[0.0, 0.0]])
    hair = make_layer("hair_front", "hair_front", [[500.0, 200.0]], **common)
    face = make_layer("face", "face", [[500.0, 200.0]], **common)
    d = Deformer("head_warp", "head_angle_x", ["hair_front", "face"], {
        "center": [500.0, 200.0], "radius": 200.0,
        "shift_px_per_deg_x": 1.0, "shift_px_per_deg_y": 0.5,
        "roll_center": [500.0, 350.0],
        "depth": {"hair_front": 1.3, "face": 1.0},
    })
    rig = Rig((1000, 1000), [hair, face], [], [d])
    out = deform_rig(rig, make_param_frame({"head_angle_x": 30.0}))
    hair_dx = out["hair_front"][0, 0] - 500.0
    face_dx = out["face"][0, 0] - 500.0
    assert hair_dx > face_dx > 0.0
    np.testing.assert_allclose(hair_dx / face_dx, 1.3, atol=0.01)


def test_param_value_defaults():
    f = make_param_frame({})
    assert param_value(f, "eye_l_open") == 1.0
    assert param_value(f, "mouth_open") == 0.0
    assert param_value(f, "arm_upper_l_rot") == 0.0
