"""Face-crop preprocessing and feature extraction."""
from __future__ import annotations

import cv2
import numpy as np

FACE_SIZE = 40  # side of the square grayscale crop used everywhere


def crop_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int], size: int = FACE_SIZE,
              margin: float = 0.15) -> np.ndarray:
    """Square crop around ``box`` (enlarged by ``margin``), grayscale, resized, histogram-equalised."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
    h, w = frame_bgr.shape[:2]
    xa, xb = int(max(0, cx - half)), int(min(w, cx + half))
    ya, yb = int(max(0, cy - half)), int(min(h, cy + half))
    patch = frame_bgr[ya:yb, xa:xb]
    if patch.size == 0:
        return np.zeros((size, size), np.uint8)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(gray)


def pixel_features(faces: np.ndarray) -> np.ndarray:
    """Flattened pixels scaled to [0, 1]; ``faces`` is ``(n, S, S)`` uint8."""
    return faces.reshape(len(faces), -1).astype(np.float32) / 255.0


def hog_features(faces: np.ndarray) -> np.ndarray:
    """Histogram-of-oriented-gradients descriptor of every crop."""
    size = faces.shape[1]
    hog = cv2.HOGDescriptor((size, size), (16, 16), (8, 8), (8, 8), 9)
    return np.stack([hog.compute(f).ravel() for f in faces]).astype(np.float32)


def extract_features(faces: np.ndarray, kind: str = "hog") -> np.ndarray:
    if kind == "hog":
        return hog_features(faces)
    if kind == "pixels":
        return pixel_features(faces)
    raise ValueError("kind must be 'hog' or 'pixels'")
