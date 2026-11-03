"""Small helpers around ``cv2.VideoCapture`` / ``cv2.VideoWriter``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int


def video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(str(path), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def iter_frames(path: str | Path, stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_index, bgr_frame)`` for every ``stride``-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    index, yielded = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, frame
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        cap.release()


def write_video(frames: Iterator[np.ndarray] | list[np.ndarray], path: str | Path, fps: float = 15.0) -> Path:
    """Write BGR frames to an MP4 file (``mp4v`` codec)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for frame in frames:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
    if writer is None:
        raise ValueError("no frames to write")
    writer.release()
    return path
