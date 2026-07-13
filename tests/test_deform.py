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


def test_two_bone_blend_midpoint():
    still = Bone("arm_upper_l", None,
                 np.array([0.0, 0.0], np.float32), np.array([0.0, 10.0], np.float32))
    turner = Bone("arm_upper_r", None,
                  np.array([0.0, 0.0], np.float32), np.array([0.0, 10.0], np.float32))
    layer = make_layer("blend", "arm_l", [[10.0, 0.0]], [[0, 1]], [[0.5, 0.5]])
    rig = Rig((100, 100), [layer], [still, turner], [])
    out = deform_rig(rig, make_param_frame({"arm_upper_r_rot": 90.0}))
    # still bone (rot 0, identity): vertex stays at (10, 0)
    # turner bone, 90deg CCW-canvas about its head (0,0): x'=-y=0, y'=x=10 -> (0, 10)
    # 0.5/0.5 LBS blend -> midpoint of (10, 0) and (0, 10) = (5, 5)
    np.testing.assert_allclose(out["blend"][0], [5.0, 5.0], atol=1e-3)


def two_bone_chain_rig():
    parent = Bone("hips", None,
                  np.array([0.0, 0.0], np.float32), np.array([0.0, -10.0], np.float32))
    child = Bone("spine", "hips",
                 np.array([0.0, -10.0], np.float32), np.array([0.0, -20.0], np.float32))
    layer = make_layer("t", "torso", [[0.0, -20.0]], [[1, -1]], [[1.0, 0.0]])
    return Rig((100, 100), [layer], [parent, child], [])


def test_child_inherits_parent_rotation():
    rig = two_bone_chain_rig()
    out = deform_rig(rig, make_param_frame({"hips_rot": 90.0}))
    np.testing.assert_allclose(out["t"][0], [20.0, 0.0], atol=1e-3)


def test_parent_and_child_rotation_compose():
    rig = two_bone_chain_rig()
    out = deform_rig(rig, make_param_frame({"hips_rot": 90.0, "spine_rot": 90.0}))
    # spine local first: rotate (0,-20) about spine's rest head (0,-10) by 90 CCW:
    #   rel (0,-10) -> (x'=-y=10, y'=x=0) -> (10, -10)
    # then hips world: rotate (10,-10) about (0,0) by 90 CCW:
    #   x'=-y=10, y'=x=10 -> (10, 10)
    np.testing.assert_allclose(out["t"][0], [10.0, 10.0], atol=1e-3)


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


def test_head_warp_pitch_shifts_vertically_with_falloff():
    common = dict(indices=[[-1, -1]] * 2, weights=[[0.0, 0.0]] * 2)
    face = make_layer("face", "face", [[500.0, 200.0], [700.0, 200.0]], **common)
    d = Deformer("head_warp", "head_angle_x", ["face"], {
        "center": [500.0, 200.0], "radius": 300.0,
        "shift_px_per_deg_x": 1.0, "shift_px_per_deg_y": 0.5,
        "roll_center": [500.0, 300.0],
        "depth": {"face": 1.0},
    })
    rig = Rig((1000, 1000), [face], [], [d])
    out = deform_rig(rig, make_param_frame({"head_angle_y": 20.0}))["face"]
    # vertex at center: dist 0 -> falloff cos(0) = 1
    #   y += 20 * 0.5 * depth 1.0 * 1 = 10 -> (500, 210); x untouched (head_angle_x=0)
    np.testing.assert_allclose(out[0], [500.0, 210.0], atol=1e-3)
    # vertex 200px out, radius 300: falloff = cos(pi/2 * 200/300) = cos 60deg = 0.5
    #   y += 20 * 0.5 * 1.0 * 0.5 = 5 -> (700, 205)
    np.testing.assert_allclose(out[1], [700.0, 205.0], atol=1e-3)


def test_head_warp_roll_rotates_scaled_by_falloff():
    common = dict(indices=[[-1, -1]] * 2, weights=[[0.0, 0.0]] * 2)
    face = make_layer("face", "face", [[500.0, 200.0], [500.0, 400.0]], **common)
    d = Deformer("head_warp", "head_angle_x", ["face"], {
        "center": [500.0, 200.0], "radius": 300.0,
        "shift_px_per_deg_x": 1.0, "shift_px_per_deg_y": 0.5,
        "roll_center": [500.0, 300.0],
        "depth": {"face": 1.0},
    })
    rig = Rig((1000, 1000), [face], [], [d])
    out = deform_rig(rig, make_param_frame({"head_angle_z": 30.0}))["face"]
    # vertex at center: falloff 1 -> full 30deg roll about roll_center (500, 300).
    #   rel (0, -100); x' = 100*sin30 = 50.0, y' = -100*cos30 = -86.602540
    #   -> (550.0, 213.397460)
    np.testing.assert_allclose(out[0], [550.0, 213.397460], atol=1e-3)
    # vertex at (500, 400): dist 200 from center, radius 300
    #   falloff = cos(pi/2 * 200/300) = cos 60deg = 0.5 -> effective roll 15deg
    #   rel (0, 100); x' = -100*sin15 = -25.881905, y' = 100*cos15 = 96.592583
    #   -> (474.118095, 396.592583)
    np.testing.assert_allclose(out[1], [474.118095, 396.592583], atol=1e-3)


def test_body_shift_translates_and_parallaxes():
    common = dict(indices=[[-1, -1]], weights=[[0.0, 0.0]])
    torso = make_layer("torso", "torso", [[100.0, 200.0]], **common)
    legs = make_layer("legs", "torso", [[100.0, 200.0]], **common)
    d = Deformer("body_shift", "body_angle_x", ["torso", "legs"], {
        "shift_px_per_deg_x": 2.0, "shift_px_per_deg_y": 1.0,
        "roll_center": [100.0, 300.0],
        "depth": {"torso": 1.0, "legs": 0.5},
    })
    rig = Rig((400, 400), [torso, legs], [], [d])
    out = deform_rig(
        rig, make_param_frame({"body_angle_x": 10.0, "body_angle_y": 10.0}))
    # torso (depth 1.0): x += 10*2.0*1.0 = 20, y += 10*1.0*1.0 = 10
    np.testing.assert_allclose(out["torso"][0], [120.0, 210.0], atol=1e-3)
    # legs (depth 0.5): x += 10*2.0*0.5 = 10, y += 10*1.0*0.5 = 5
    np.testing.assert_allclose(out["legs"][0], [110.0, 205.0], atol=1e-3)


def test_body_shift_roll_rotates_about_roll_center():
    torso = make_layer("torso", "torso", [[100.0, 200.0]],
                       [[-1, -1]], [[0.0, 0.0]])
    d = Deformer("body_shift", "body_angle_x", ["torso"], {
        "shift_px_per_deg_x": 2.0, "shift_px_per_deg_y": 1.0,
        "roll_center": [100.0, 300.0],
        "depth": {"torso": 1.0},
    })
    rig = Rig((400, 400), [torso], [], [d])
    out = deform_rig(rig, make_param_frame({"body_angle_z": 10.0}))["torso"]
    # no x/y shift (body_angle_x/y default 0); roll 10deg CCW about (100, 300).
    # rel (0, -100); x' = 100*sin10 = 17.364818, y' = -100*cos10 = -98.480775
    # -> (117.364818, 201.519225)
    np.testing.assert_allclose(out[0], [117.364818, 201.519225], atol=1e-3)


def test_param_value_defaults():
    f = make_param_frame({})
    assert param_value(f, "eye_l_open") == 1.0
    assert param_value(f, "mouth_open") == 0.0
    assert param_value(f, "arm_upper_l_rot") == 0.0
