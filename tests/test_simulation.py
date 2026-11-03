import numpy as np
import cv2
import pytest
from fastfacetrack.clustering import cluster_faces, cluster_purity
from fastfacetrack.detectors import HaarDetector
from fastfacetrack.simulation import Identity, IdentityBank, SimulationConfig, simulate_video
from fastfacetrack.evaluation import evaluate_video, load_truth
from fastfacetrack.models import FaceClassifier
from fastfacetrack.video import video_info


def _bank():
    rng = np.random.default_rng(0)
    ids = []
    for k in range(3):
        patch = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
        ids.append(Identity(f"p{k}", patch, (20, 20, 80, 100), 60))
    return IdentityBank(ids)


def test_simulation_writes_video_and_matching_truth(tmp_path):
    cfg = SimulationConfig(width=320, height=240, n_frames=30, seed=3, face_width=(40, 60))
    truth = simulate_video(_bank(), "p0", ["p1"], tmp_path / "v.mp4", cfg, target_always_present=True,
                           others_always_present=True)
    info = video_info(tmp_path / "v.mp4")
    assert (info.width, info.height, info.n_frames) == (320, 240, 30)
    loaded = load_truth(tmp_path / "v.truth.json")
    assert len(loaded) == 30 == len(truth["frames"])
    for faces in loaded.values():
        for f in faces:
            x1, y1, x2, y2 = f["box"]
            assert 0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240


def test_simulation_is_deterministic(tmp_path):
    cfg = SimulationConfig(width=320, height=240, n_frames=15, seed=5, face_width=(40, 60))
    a = simulate_video(_bank(), "p0", ["p1"], tmp_path / "a.mp4", cfg)
    b = simulate_video(_bank(), "p0", ["p1"], tmp_path / "b.mp4", cfg)
    assert a["frames"] == b["frames"]


def test_evaluate_video_returns_expected_columns(tmp_path, toy_faces):
    cfg = SimulationConfig(width=320, height=240, n_frames=12, seed=1, face_width=(40, 60))
    simulate_video(_bank(), "p0", ["p1"], tmp_path / "v.mp4", cfg)
    x, y = toy_faces
    clf = FaceClassifier("tree").fit(x, y)
    summary, per_frame = evaluate_video(tmp_path / "v.mp4", HaarDetector(), {"tree": clf}, tmp_path / "v.truth.json")
    for col in ("detect_ms", "classify_ms", "fps", "det_f1", "frame_accuracy"):
        assert col in summary.columns
    assert len(per_frame) == 12


def test_clustering_separates_two_clear_groups(toy_faces):
    x, y = toy_faces
    labels, info = cluster_faces(x, 2, features="pixels")
    assert cluster_purity(labels, y) > 0.95
