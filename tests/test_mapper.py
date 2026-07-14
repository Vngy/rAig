import numpy as np

from raig.tracking.bundle import LandmarkBundle
from raig.tracking.mapper import Mapper, angle_from_down

R_SHOULDER, R_ELBOW, R_WRIST = 12, 14, 16


def pose_array(joints: dict) -> np.ndarray:
    pose = np.zeros((33, 4))
    pose[:, 3] = 1.0  # everything visible by default
    for idx, (x, y) in joints.items():
        pose[idx, 0], pose[idx, 1] = x, y
    return pose


def yaw_matrix(deg):
    t = np.deg2rad(deg)
    m = np.eye(4)
    m[0, 0], m[0, 2] = np.cos(t), np.sin(t)
    m[2, 0], m[2, 2] = -np.sin(t), np.cos(t)
    return m


def test_angle_from_down():
    assert abs(angle_from_down(np.array([0.0, 1.0]))) < 1e-9  # straight down
    assert abs(angle_from_down(np.array([1.0, 0.0])) + 90.0) < 1e-9  # right = -90


def test_blink_is_mirrored():
    m = Mapper(smooth=False)
    f = m.map(LandmarkBundle(0.0, face_blendshapes={
        "eyeBlinkRight": 1.0, "eyeBlinkLeft": 0.0, "jawOpen": 0.0}))
    assert f.values["eye_l_open"] < 0.05  # user's RIGHT eye drives avatar _l
    assert f.values["eye_r_open"] > 0.95


def test_jaw_open_maps_to_mouth():
    m = Mapper(smooth=False)
    f = m.map(LandmarkBundle(0.0, face_blendshapes={"jawOpen": 0.7}))
    assert abs(f.values["mouth_open"] - 0.7) < 1e-6


def test_head_yaw_from_matrix():
    m = Mapper(smooth=False)
    f = m.map(LandmarkBundle(0.0, face_matrix=yaw_matrix(20.0)))
    assert abs(f.values["head_angle_x"] - 20.0) < 0.5


def test_arm_angle_from_pose():
    m = Mapper(smooth=False)
    # user's right arm: shoulder (0.3, 0.5), elbow to its right, wrist below elbow
    pose = pose_array({R_SHOULDER: (0.3, 0.5), R_ELBOW: (0.4, 0.5),
                       R_WRIST: (0.4, 0.6)})
    f = m.map(LandmarkBundle(0.0, pose=pose))
    assert abs(f.values["arm_upper_l_rot"] + 90.0) < 1e-3  # horizontal = -90
    assert abs(f.values["arm_lower_l_rot"] - 90.0) < 1e-3  # relative back to down


def test_low_visibility_limb_decays_to_rest():
    m = Mapper(smooth=False)
    pose = pose_array({R_SHOULDER: (0.3, 0.5), R_ELBOW: (0.4, 0.5),
                       R_WRIST: (0.4, 0.6)})
    f = m.map(LandmarkBundle(0.0, pose=pose))
    assert abs(f.values["arm_upper_l_rot"]) > 80.0
    hidden = pose.copy()
    hidden[[R_SHOULDER, R_ELBOW, R_WRIST], 3] = 0.1
    f = m.map(LandmarkBundle(1.0, pose=hidden))  # 1s later: fully decayed
    assert abs(f.values.get("arm_upper_l_rot", 0.0)) < 1.0


def test_same_timestamp_dropout_retains_value():
    # dt == 0 must mean full retention (exp(0) == 1), never a snap to default
    m = Mapper(smooth=False)
    f = m.map(LandmarkBundle(0.0, face_blendshapes={
        "eyeBlinkRight": 1.0, "eyeBlinkLeft": 1.0}))
    assert f.values["eye_l_open"] < 0.05
    f = m.map(LandmarkBundle(0.0))  # face lost, zero elapsed time
    assert f.values["eye_l_open"] < 0.05  # retained, not snapped to default 1.0


def test_hand_rot_uses_current_frame_arm_angles():
    m = Mapper(smooth=False)
    # user's right arm all along the 45-deg down-right diagonal:
    #   shoulder (0.3, 0.5) -> elbow (0.4, 0.6): angle_from_down((0.1, 0.1)) = -45
    #   elbow -> wrist (0.5, 0.7): same direction, so lower (relative) = 0
    pose = pose_array({R_SHOULDER: (0.3, 0.5), R_ELBOW: (0.4, 0.6),
                       R_WRIST: (0.5, 0.7)})
    # user's right hand (drives avatar hand_l): wrist landmark 0 at (0.5, 0.7),
    # middle_mcp landmark 9 straight below at (0.5, 0.75): absolute angle = 0
    hand = np.zeros((21, 3))
    hand[0, :2] = (0.5, 0.7)
    hand[9, :2] = (0.5, 0.75)
    f = m.map(LandmarkBundle(0.0, pose=pose, hand_r=hand))
    # relative to THIS frame's arm chain: 0 - lower(0) - upper(-45) = +45.
    # a stale read of last-frame arm angles (empty on first frame) would give 0.
    assert abs(f.values["hand_l_rot"] - 45.0) < 1e-3


def test_body_angle_z_from_shoulder_tilt():
    m = Mapper(smooth=False)
    # shoulder line: user's left shoulder (idx 11, screen-right) at (0.6, 0.52),
    # user's right shoulder (idx 12, screen-left) at (0.4, 0.5).
    # d = pos(11) - pos(12) = (0.2, 0.02); level shoulders give d = (+x, 0) -> 0.
    # body_angle_z = degrees(atan2(d.y, d.x)) = degrees(atan2(0.02, 0.2)) = +5.7106.
    # Sign check vs angle_from_down's positive-is-CCW canvas convention: the
    # canvas-CCW rotation R(t) = [[cos, -sin], [sin, cos]] (it takes down (0,1)
    # to left (-1,0) at t=+90, matching angle_from_down((-1,0)) = +90) applied
    # to the level shoulder vector (1, 0) gives (cos t, sin t), and
    # atan2(sin t, cos t) = t — so body_angle_z is exactly the canvas-CCW
    # rotation of the shoulder line. Consistent.
    # hips symmetric about the shoulder mid-x so body_angle_x = 0.
    pose = pose_array({11: (0.6, 0.52), 12: (0.4, 0.5),
                       23: (0.6, 0.9), 24: (0.4, 0.9)})
    f = m.map(LandmarkBundle(0.0, pose=pose))
    assert abs(f.values["body_angle_z"] - 5.7106) < 1e-3
    assert abs(f.values["body_angle_x"]) < 1e-6


def test_body_angle_x_from_lean():
    m = Mapper(smooth=False)
    # level shoulders (no z tilt): mid-x = (0.65 + 0.45) / 2 = 0.55
    # hips: mid-x = (0.6 + 0.4) / 2 = 0.5
    # brief: body_angle_x = (shoulder-mid - hip-mid horizontal lean) * 60
    #      = (0.55 - 0.5) * 60 = +3.0 (shoulders leaning toward screen +x)
    pose = pose_array({11: (0.65, 0.5), 12: (0.45, 0.5),
                       23: (0.6, 0.9), 24: (0.4, 0.9)})
    f = m.map(LandmarkBundle(0.0, pose=pose))
    assert abs(f.values["body_angle_x"] - 3.0) < 1e-6
    assert abs(f.values["body_angle_z"]) < 1e-6


def test_missing_face_decays_eyes_open():
    m = Mapper(smooth=False)
    m.map(LandmarkBundle(0.0, face_blendshapes={
        "eyeBlinkRight": 1.0, "eyeBlinkLeft": 1.0}))
    f = m.map(LandmarkBundle(1.0))  # face lost for 1s
    assert f.values["eye_l_open"] > 0.95  # back to default open


def test_calibration_hook_applied():
    class Cal:
        def apply(self, values):
            return {k: v - 5.0 if k == "head_angle_x" else v
                    for k, v in values.items()}

    m = Mapper(calibration=Cal(), smooth=False)
    f = m.map(LandmarkBundle(0.0, face_matrix=yaw_matrix(20.0)))
    assert abs(f.values["head_angle_x"] - 15.0) < 0.5
