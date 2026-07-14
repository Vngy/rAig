import numpy as np

from raig.tracking.bundle import LandmarkBundle


def bundle_from_results(
    timestamp_s: float, face_result, pose_result, hand_result
) -> LandmarkBundle:
    blendshapes = None
    if face_result.face_blendshapes:
        blendshapes = {
            c.category_name: c.score for c in face_result.face_blendshapes[0]
        }
    matrix = None
    if face_result.facial_transformation_matrixes:
        matrix = np.asarray(face_result.facial_transformation_matrixes[0],
                            dtype=np.float64)
    pose = None
    if pose_result.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility]
             for lm in pose_result.pose_landmarks[0]],
            dtype=np.float64,
        )
    hand_l = hand_r = None
    for handedness, lms in zip(hand_result.handedness, hand_result.hand_landmarks):
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float64)
        if handedness[0].category_name == "Left":
            hand_l = arr
        else:
            hand_r = arr
    return LandmarkBundle(
        timestamp=timestamp_s,
        face_blendshapes=blendshapes,
        face_matrix=matrix,
        pose=pose,
        hand_l=hand_l,
        hand_r=hand_r,
    )


class LandmarkExtractor:
    def __init__(self, model_paths: dict):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_paths["face"])),
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_paths["pose"])),
                running_mode=vision.RunningMode.VIDEO,
            )
        )
        self._hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_paths["hand"])),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )

    def extract(self, frame_rgb: np.ndarray, timestamp_ms: int) -> LandmarkBundle:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame_rgb
        )
        face = self._face.detect_for_video(image, timestamp_ms)
        pose = self._pose.detect_for_video(image, timestamp_ms)
        hand = self._hand.detect_for_video(image, timestamp_ms)
        return bundle_from_results(timestamp_ms / 1000.0, face, pose, hand)
