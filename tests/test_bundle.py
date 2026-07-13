import numpy as np

from raig.tracking.bundle import LandmarkBundle, load_replay, save_replay


def test_jsonl_round_trip(tmp_path):
    bundles = [
        LandmarkBundle(
            timestamp=0.0,
            face_blendshapes={"jawOpen": 0.5},
            face_matrix=np.eye(4),
            pose=np.zeros((33, 4)),
        ),
        LandmarkBundle(timestamp=1 / 60, hand_l=np.ones((21, 3))),
    ]
    path = tmp_path / "session.jsonl"
    save_replay(bundles, path)
    loaded = load_replay(path)
    assert len(loaded) == 2
    assert loaded[0].face_blendshapes == {"jawOpen": 0.5}
    np.testing.assert_allclose(loaded[0].face_matrix, np.eye(4))
    assert loaded[0].pose.shape == (33, 4)
    assert loaded[0].hand_l is None
    np.testing.assert_allclose(loaded[1].hand_l, np.ones((21, 3)))
    assert loaded[1].pose is None
