"""Extract face crops from a video with a detector."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detectors import FaceDetector
from .features import FACE_SIZE, crop_face
from .video import iter_frames


@dataclass
class FaceSet:
    """Face crops with their provenance. ``faces`` is ``(n, S, S)`` uint8 grayscale."""

    faces: np.ndarray
    frame_index: np.ndarray
    boxes: np.ndarray            # (n, 4) x1, y1, x2, y2
    scores: np.ndarray
    source: str = ""
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.faces)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, faces=self.faces, frame_index=self.frame_index, boxes=self.boxes,
                            scores=self.scores, source=self.source)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FaceSet":
        z = np.load(path, allow_pickle=False)
        return cls(z["faces"], z["frame_index"], z["boxes"], z["scores"], str(z["source"]))

    def save_images(self, folder: str | Path) -> int:
        """Write every crop as a PNG (``face_000000.png`` ...)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for i, face in enumerate(self.faces):
            cv2.imwrite(str(folder / f"face_{i:06d}.png"), face)
        return len(self.faces)


def concat(sets: list[FaceSet]) -> FaceSet:
    return FaceSet(np.concatenate([s.faces for s in sets]), np.concatenate([s.frame_index for s in sets]),
                   np.concatenate([s.boxes for s in sets]), np.concatenate([s.scores for s in sets]),
                   ",".join(s.source for s in sets))


def extract_faces(video_path: str | Path, detector: FaceDetector, stride: int = 1, min_size: int = 24,
                  min_score: float = 0.0, size: int = FACE_SIZE, max_frames: int | None = None) -> FaceSet:
    """Run ``detector`` on the video and keep every face at least ``min_size`` pixels wide."""
    faces, idx, boxes, scores = [], [], [], []
    for i, frame in iter_frames(video_path, stride, max_frames):
        for det in detector.detect(frame):
            if min(det.width, det.height) < min_size or det.score < min_score:
                continue
            faces.append(crop_face(frame, det.box, size))
            idx.append(i)
            boxes.append(det.box)
            scores.append(det.score)
    n = len(faces)
    return FaceSet(np.stack(faces) if n else np.zeros((0, size, size), np.uint8),
                   np.array(idx, dtype=int), np.array(boxes, dtype=int).reshape(n, 4),
                   np.array(scores, dtype=float), str(video_path))
