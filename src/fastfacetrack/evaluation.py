"""Test trained classifiers on the frames of a video and record timing and target presence.

The detector runs once per frame; every classifier then scores the detected crops, so detection time
and classification time are reported separately.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .detectors import Detection, FaceDetector, iou
from .features import FACE_SIZE, crop_face
from .models import FaceClassifier
from .video import iter_frames


@dataclass
class FrameDetections:
    frame_index: int
    detections: list[Detection]
    crops: np.ndarray            # (m, S, S) uint8
    detect_ms: float


def load_truth(path: str | Path) -> dict[int, list[dict]]:
    """Ground truth written by the simulator: ``{frame_index: [{"id", "box", "target"}, ...]}``."""
    data = json.loads(Path(path).read_text())
    return {f["frame"]: f["faces"] for f in data["frames"]}


def run_detector(video_path: str | Path, detector: FaceDetector, stride: int = 1,
                 max_frames: int | None = None, size: int = FACE_SIZE) -> list[FrameDetections]:
    out = []
    for i, frame in iter_frames(video_path, stride, max_frames):
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        ms = (time.perf_counter() - t0) * 1000
        crops = np.stack([crop_face(frame, d.box, size) for d in dets]) if dets else np.zeros((0, size, size), np.uint8)
        out.append(FrameDetections(i, dets, crops, ms))
    return out


def _match(dets: list[Detection], truth_faces: list[dict], min_iou: float):
    """Assign every detection to the best ground-truth face (IoU >= min_iou) or ``None``."""
    assigned = []
    for d in dets:
        best, best_iou = None, min_iou
        for k, g in enumerate(truth_faces):
            v = iou(d.box, tuple(g["box"]))
            if v >= best_iou:
                best, best_iou = k, v
        assigned.append(best)
    return assigned


def score_classifier(frames: list[FrameDetections], clf: FaceClassifier, threshold: float = 0.5,
                     truth: dict[int, list[dict]] | None = None, min_iou: float = 0.4) -> pd.DataFrame:
    """One row per frame: timings, predicted/true target presence and detection-level TP/FP/FN."""
    rows = []
    for fd in frames:
        t0 = time.perf_counter()
        probs = clf.predict_proba(fd.crops)
        cls_ms = (time.perf_counter() - t0) * 1000
        pred_target = probs >= threshold
        row = {"model": clf.name, "frame": fd.frame_index, "n_faces": len(fd.detections),
               "detect_ms": fd.detect_ms, "classify_ms": cls_ms,
               "max_target_prob": float(probs.max()) if len(probs) else 0.0,
               "pred_present": bool(pred_target.any())}
        if truth is not None:
            gt = truth.get(fd.frame_index, [])
            assigned = _match(fd.detections, gt, min_iou)
            tp = fp = 0
            found_target = False
            for pred, k in zip(pred_target, assigned):
                if not pred:
                    continue
                if k is not None and gt[k]["target"]:
                    tp += 1
                    found_target = True
                else:
                    fp += 1
            n_target = sum(g["target"] for g in gt)
            row.update(true_present=n_target > 0, n_true_faces=len(gt),
                       n_matched=len({k for k in assigned if k is not None}), tp=tp, fp=fp,
                       fn=max(0, n_target - tp))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_frame: pd.DataFrame) -> dict:
    """Aggregate a :func:`score_classifier` table into one result row."""
    d = per_frame
    out = {"model": d["model"].iloc[0], "frames": len(d),
           "detect_ms": d["detect_ms"].mean(), "classify_ms": d["classify_ms"].mean()}
    out["total_ms"] = out["detect_ms"] + out["classify_ms"]
    out["fps"] = 1000.0 / out["total_ms"] if out["total_ms"] > 0 else float("inf")
    out["frames_target_predicted"] = int(d["pred_present"].sum())
    if "true_present" in d:
        tp, fp, fn = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        out.update(
            det_precision=prec, det_recall=rec,
            det_f1=2 * prec * rec / (prec + rec) if tp else 0.0,
            face_detection_recall=d["n_matched"].sum() / max(1, d["n_true_faces"].sum()),
            frame_accuracy=(d["pred_present"] == d["true_present"]).mean(),
            frames_target_true=int(d["true_present"].sum()))
        first_true = d.index[d["true_present"]]
        hit = d.index[d["true_present"] & d["pred_present"] & (d["tp"] > 0)]
        out["first_hit_delay_frames"] = (int(d.loc[hit[0], "frame"] - d.loc[first_true[0], "frame"])
                                         if len(hit) and len(first_true) else float("nan"))
    return out


def evaluate_video(video_path: str | Path, detector: FaceDetector, classifiers: dict[str, FaceClassifier],
                   truth_path: str | Path | None = None, threshold: float = 0.5, stride: int = 1,
                   max_frames: int | None = None, min_iou: float = 0.4):
    """Detect once, classify with every model. Returns ``(summary, per_frame)`` DataFrames."""
    frames = run_detector(video_path, detector, stride, max_frames)
    truth = load_truth(truth_path) if truth_path else None
    tables = [score_classifier(frames, c, threshold, truth, min_iou) for c in classifiers.values()]
    per_frame = pd.concat(tables, ignore_index=True)
    summary = pd.DataFrame([summarise(t) for t in tables])
    summary.insert(0, "detector", detector.name)
    return summary, per_frame


def combine_results(tables: list[pd.DataFrame], path: str | Path | None = None) -> pd.DataFrame:
    """Concatenate result tables (e.g. one per detector or video) and optionally write a CSV."""
    combined = pd.concat(tables, ignore_index=True)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
    return combined
