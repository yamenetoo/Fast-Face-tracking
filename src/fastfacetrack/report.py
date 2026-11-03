"""Figures for the experiment report (matplotlib is imported lazily)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def plot_model_comparison(summary: pd.DataFrame, path: str | Path) -> Path:
    """Detection-level F1 (left) and time per frame split into detection and classification (right)."""
    plt = _plt()
    g = summary.groupby("model", sort=False)
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True).fillna(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(mean.index, mean["det_f1"], yerr=std["det_f1"], color="tab:blue", capsize=3)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("F1 of target detections")
    ax[0].set_title("Accuracy (mean ± sd over test videos)")
    ax[1].bar(mean.index, mean["detect_ms"], label="face detector", color="tab:gray")
    ax[1].bar(mean.index, mean["classify_ms"], bottom=mean["detect_ms"], label="classifier", color="tab:orange")
    ax[1].set_ylabel("ms per frame")
    ax[1].set_title("Time per frame")
    ax[1].legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_timeline(per_frame: pd.DataFrame, model: str, path: str | Path, threshold: float = 0.5) -> Path:
    """Target probability per frame against the frames in which the target is truly on screen."""
    plt = _plt()
    d = per_frame[per_frame["model"] == model]
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(d["frame"], 0, d["true_present"].astype(float), step="mid", alpha=0.25, color="tab:green",
                    label="target on screen (truth)")
    ax.plot(d["frame"], d["max_target_prob"], lw=1.2, color="tab:blue", label=f"{model}: max target probability")
    ax.axhline(threshold, color="tab:red", ls="--", lw=0.8, label="threshold")
    ax.set_xlabel("frame")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def annotate_frames(video_path: str | Path, frames_detections, classifier, out_path: str | Path,
                    frame_indices: list[int], threshold: float = 0.5, columns: int = 3, scale: float = 0.5) -> Path:
    """Grid of frames: green box = classified as target, red = other (with probability)."""
    from .video import iter_frames
    wanted = set(frame_indices)
    by_index = {fd.frame_index: fd for fd in frames_detections}
    tiles = []
    for i, frame in iter_frames(video_path):
        if i not in wanted or i not in by_index:
            continue
        fd = by_index[i]
        probs = classifier.predict_proba(fd.crops)
        for det, p in zip(fd.detections, probs):
            colour = (0, 200, 0) if p >= threshold else (0, 0, 220)
            cv2.rectangle(frame, det.box[:2], det.box[2:], colour, 2)
            cv2.putText(frame, f"{p:.2f}", (det.x1, max(14, det.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.putText(frame, f"frame {i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(cv2.resize(frame, None, fx=scale, fy=scale))
    if not tiles:
        raise ValueError("none of the requested frames were found")
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + columns]) for r in range(0, len(tiles), columns)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path
