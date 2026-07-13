import numpy as np
import pytest

from raig.compiler.errors import CompileError
from raig.compiler.skeleton import SLOT_TO_CHAIN, assign_weights, fit_skeleton
from raig.core.rig import Bone

CANVAS = (1024, 1024)

BBOXES = {
    "face": (392.0, 130.0, 632.0, 390.0),
    "torso": (400.0, 380.0, 624.0, 660.0),
    "arm_l": (330.0, 400.0, 400.0, 640.0),
    "leg_l": (430.0, 640.0, 500.0, 980.0),
}


def by_name(bones):
    return {b.name: b for b in bones}


def test_bone_set_matches_present_slots():
    bones = by_name(fit_skeleton(BBOXES, CANVAS))
    assert set(bones) == {
        "hips", "spine", "chest", "neck", "head",
        "arm_upper_l", "arm_lower_l", "hand_l",
        "leg_upper_l", "leg_lower_l",
    }
    assert bones["spine"].parent == "hips"
    assert bones["head"].parent == "neck"
    assert bones["hand_l"].parent == "arm_lower_l"


def test_spine_chain_geometry():
    bones = by_name(fit_skeleton(BBOXES, CANVAS))
    x0, y0, x1, y1 = BBOXES["torso"]
    hips = bones["hips"]
    assert x0 < hips.head[0] < x1 and y0 < hips.head[1] < y1
    # chain points upward (+y down): each tail is above its head
    for name in ("hips", "spine", "chest", "head"):
        assert bones[name].tail[1] < bones[name].head[1]
    # neck connects chest tail to bottom of head box
    np.testing.assert_allclose(bones["neck"].head, bones["chest"].tail)


def test_arm_chain_starts_near_torso():
    bones = by_name(fit_skeleton(BBOXES, CANVAS))
    upper = bones["arm_upper_l"]
    # arm bbox is tall: chain runs top->bottom, starting at the end nearer the chest
    assert upper.head[1] < upper.tail[1]
    assert abs(upper.head[0] - 365.0) < 1.0 and abs(upper.head[1] - 400.0) < 1.0
    # continuity
    np.testing.assert_allclose(bones["arm_lower_l"].head, upper.tail)
    np.testing.assert_allclose(bones["hand_l"].head, bones["arm_lower_l"].tail)


def test_missing_torso_raises():
    with pytest.raises(CompileError):
        fit_skeleton({"face": BBOXES["face"]}, CANVAS)


def test_missing_head_raises():
    with pytest.raises(CompileError):
        fit_skeleton({"torso": BBOXES["torso"]}, CANVAS)


def _two_bone_chain():
    return [
        Bone("a", None, np.array([0.0, 0.0], np.float32), np.array([0.0, 10.0], np.float32)),
        Bone("b", "a", np.array([0.0, 10.0], np.float32), np.array([0.0, 20.0], np.float32)),
    ]


def test_weights_sum_to_one_and_favor_near_bone():
    bones = _two_bone_chain()
    verts = np.array([[0.0, 5.0], [0.0, 15.0], [3.0, 10.0]], np.float32)
    idx, w = assign_weights(verts, ["a", "b"], bones)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert w[0, list(idx[0]).index(0)] > 0.95  # on bone a
    assert w[1, list(idx[1]).index(1)] > 0.95  # on bone b


def test_single_bone_chain_full_weight():
    bones = _two_bone_chain()
    idx, w = assign_weights(np.array([[5.0, 5.0]], np.float32), ["a"], bones)
    assert idx[0, 0] == 0 and w[0, 0] == 1.0 and w[0, 1] == 0.0


def test_empty_chain_unbound():
    idx, w = assign_weights(np.array([[1.0, 1.0]], np.float32), [], _two_bone_chain())
    assert (idx == -1).all() and (w == 0).all()


def test_slot_chain_table():
    assert SLOT_TO_CHAIN["torso"] == ["hips", "spine", "chest"]
    assert SLOT_TO_CHAIN["hair_front"] == ["head"]
    assert SLOT_TO_CHAIN["arm_r"] == ["arm_upper_r", "arm_lower_r", "hand_r"]
    assert SLOT_TO_CHAIN["leg_l"] == ["leg_upper_l", "leg_lower_l"]
