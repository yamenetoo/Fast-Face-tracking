"""Face detectors.

Two interchangeable detectors share one interface (``detect(frame) -> list[Detection]``):

* :class:`UltraLightDetector` - the 1 MB "Ultra-Light-Fast-Generic-Face-Detector" (RFB-320 / slim-320
  ONNX models) run through OpenCV's DNN module. The model file is downloaded on first use.
* :class:`HaarDetector` - OpenCV's bundled Haar cascade. Needs no download; useful as a fallback.
"""
from __future__ import annotations

import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ULTRALIGHT_BASE_URL = (
    "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/"
    "master/models/onnx/"
)
ULTRALIGHT_MODELS = {"rfb-320": "version-RFB-320.onnx", "slim-320": "version-slim-320.onnx"}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fastfacetrack"


@dataclass(frozen=True)
class Detection:
    """A detected face: pixel box ``(x1, y1, x2, y2)`` and a confidence in [0, 1]."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3) -> list[int]:
    """Greedy NMS. ``boxes`` is ``(n, 4)`` as ``x1, y1, x2, y2``; returns indices to keep."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = np.array([iou(tuple(boxes[i]), tuple(boxes[j])) for j in rest])
        order = rest[overlaps <= iou_threshold]
    return keep


class FaceDetector(ABC):
    """Common interface."""

    name: str = "detector"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Return the faces found in a BGR frame."""


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(dest)
    return dest


class UltraLightDetector(FaceDetector):
    """UltraLight face detector (ONNX) executed with ``cv2.dnn``.

    Parameters
    ----------
    model:
        ``"rfb-320"`` (more accurate) or ``"slim-320"`` (faster), or a path to an ``.onnx`` file.
    score_threshold, nms_iou:
        Confidence cut-off and NMS overlap limit.
    input_size:
        ``(width, height)`` the frame is resized to; the models were trained for 320x240.
    """

    def __init__(self, model: str = "rfb-320", score_threshold: float = 0.7, nms_iou: float = 0.3,
                 input_size: tuple[int, int] = (320, 240), cache_dir: Path | str = DEFAULT_CACHE_DIR):
        if model in ULTRALIGHT_MODELS:
            path = _download(ULTRALIGHT_BASE_URL + ULTRALIGHT_MODELS[model],
                             Path(cache_dir) / ULTRALIGHT_MODELS[model])
        else:
            path = Path(model)
        self.name = "ultralight-" + Path(path).stem.lower().replace("version-", "")
        self.score_threshold, self.nms_iou, self.input_size = score_threshold, nms_iou, input_size
        self._net = cv2.dnn.readNetFromONNX(str(path))

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, scalefactor=1 / 128.0, size=self.input_size,
                                     mean=(127, 127, 127), swapRB=True)
        self._net.setInput(blob)
        scores, boxes = self._net.forward(["scores", "boxes"])
        scores, boxes = scores[0, :, 1], boxes[0]
        mask = scores > self.score_threshold
        if not mask.any():
            return []
        scores, boxes = scores[mask], boxes[mask] * np.array([w, h, w, h])
        keep = non_max_suppression(boxes, scores, self.nms_iou)
        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                out.append(Detection(x1, y1, x2, y2, float(scores[i])))
        return out


class HaarDetector(FaceDetector):
    """OpenCV Haar-cascade frontal-face detector (no download needed)."""

    name = "haar"

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: int = 24):
        self._cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scale_factor, self.min_neighbors, self.min_size = scale_factor, min_neighbors, min_size

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        found = self._cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors,
                                               minSize=(self.min_size, self.min_size))
        return [Detection(int(x), int(y), int(x + w), int(y + h), 1.0) for x, y, w, h in found]


def make_detector(kind: str = "ultralight", **kwargs) -> FaceDetector:
    """Factory: ``"ultralight"`` (or ``"ultralight-slim"``) and ``"haar"``."""
    kind = kind.lower()
    if kind == "ultralight":
        return UltraLightDetector(**kwargs)
    if kind in ("ultralight-slim", "slim"):
        return UltraLightDetector(model="slim-320", **kwargs)
    if kind == "haar":
        return HaarDetector(**kwargs)
    raise ValueError(f"unknown detector {kind!r}; use 'ultralight', 'ultralight-slim' or 'haar'")
