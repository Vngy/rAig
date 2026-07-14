from types import SimpleNamespace

import numpy as np
import pytest

from raig.tracking.landmarks import bundle_from_results
from raig.tracking.models import ensure_models, models_dir


def test_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RAIG_MODELS_DIR", str(tmp_path / "m"))
    assert models_dir() == tmp_path / "m"


def test_ensure_models_skips_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("RAIG_MODELS_DIR", str(tmp_path))
    for name in ("face_landmarker.task", "pose_landmarker_full.task",
                 "hand_landmarker.task"):
        (tmp_path / name).write_bytes(b"cached")
    calls = []
    paths = ensure_models(fetch=lambda url, dst: calls.append(url))
    assert calls == []
    assert set(paths) == {"face", "pose", "hand"}
    assert all(p.exists() for p in paths.values())


def test_ensure_models_downloads_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("RAIG_MODELS_DIR", str(tmp_path))

    def fake_fetch(url, dst):
        dst.write_bytes(b"model")

    paths = ensure_models(fetch=fake_fetch)
    assert all(p.read_bytes() == b"model" for p in paths.values())


def _category(name, score):
    return SimpleNamespace(category_name=name, score=score)


def _lm(x, y, z=0.0, visibility=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def test_bundle_from_results_full():
    face = SimpleNamespace(
        face_blendshapes=[[_category("jawOpen", 0.4), _category("eyeBlinkLeft", 0.1)]],
        facial_transformation_matrixes=[np.eye(4)],
    )
    pose = SimpleNamespace(pose_landmarks=[[_lm(0.1 * i, 0.2) for i in range(33)]])
    hand = SimpleNamespace(
        handedness=[[_category("Left", 0.99)]],
        hand_landmarks=[[_lm(0.5, 0.5) for _ in range(21)]],
    )
    b = bundle_from_results(1.5, face, pose, hand)
    assert b.timestamp == 1.5
    assert b.face_blendshapes == {"jawOpen": 0.4, "eyeBlinkLeft": 0.1}
    np.testing.assert_allclose(b.face_matrix, np.eye(4))
    assert b.pose.shape == (33, 4)
    assert abs(b.pose[3, 0] - 0.3) < 1e-9
    assert b.hand_l is not None and b.hand_l.shape == (21, 3)
    assert b.hand_r is None


def test_bundle_from_results_all_missing():
    empty_face = SimpleNamespace(face_blendshapes=[], facial_transformation_matrixes=[])
    empty_pose = SimpleNamespace(pose_landmarks=[])
    empty_hand = SimpleNamespace(handedness=[], hand_landmarks=[])
    b = bundle_from_results(0.0, empty_face, empty_pose, empty_hand)
    assert b.face_blendshapes is None and b.pose is None
    assert b.hand_l is None and b.hand_r is None


@pytest.mark.integration
def test_extractor_runs_on_synthetic_image():
    import numpy as np

    from raig.tracking.landmarks import LandmarkExtractor
    from raig.tracking.models import ensure_models

    extractor = LandmarkExtractor(ensure_models())
    frame = np.zeros((480, 640, 3), np.uint8)
    b = extractor.extract(frame, 0)  # no human: everything None, no crash
    assert b.pose is None or b.pose.shape == (33, 4)
