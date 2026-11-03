"""Command line interface: ``fastfacetrack <command> --help``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .detectors import make_detector
from .evaluation import combine_results, evaluate_video
from .extraction import FaceSet, extract_faces
from .models import MODEL_NAMES, FaceClassifier
from .training import make_dataset, train_models


def _extract(a):
    fs = extract_faces(a.video, make_detector(a.detector), a.stride, a.min_size)
    fs.save(a.out)
    if a.images:
        fs.save_images(a.images)
    print(f"{len(fs)} faces from {a.video} -> {a.out}")


def _cluster(a):
    from .clustering import cluster_faces, montage
    import cv2
    fs = FaceSet.load(a.faces)
    labels, info = cluster_faces(fs.faces, a.k, a.features)
    cv2.imwrite(a.out, montage(fs.faces, labels))
    print(f"silhouette {info['silhouette']:.3f}; montage -> {a.out} (one row per cluster)")
    np.save(Path(a.out).with_suffix(".labels.npy"), labels)


def _train(a):
    faces, y, groups = make_dataset([FaceSet.load(p) for p in a.target], [FaceSet.load(p) for p in a.other])
    clfs, report = train_models(faces, y, groups, tuple(a.models), a.features, a.seed, epochs=a.epochs)
    for c in clfs.values():
        c.save(a.out)
    print(report.to_string(index=False))
    report.to_csv(Path(a.out) / "training_report.csv", index=False)


def _test(a):
    clfs = {n: FaceClassifier.load(a.models_dir, n) for n in a.models}
    tables = []
    for d in a.detectors:
        summary, _ = evaluate_video(a.video, make_detector(d), clfs, a.truth, a.threshold, a.stride)
        tables.append(summary)
    result = combine_results(tables, a.out)
    print(result.round(3).to_string(index=False))


def _simulate(a):
    from .simulation import IdentityBank, SimulationConfig, simulate_video
    bank = IdentityBank.demo(make_detector(a.detector), a.cache)
    names = [i.name for i in bank.identities if i.source_width >= 60]
    cfg = SimulationConfig(n_frames=a.frames, seed=a.seed)
    simulate_video(bank, names[0], names[1:1 + a.others], a.out, cfg)
    print(f"video -> {a.out}; ground truth -> {Path(a.out).with_suffix('.truth.json')}; target = {names[0]}")


def _experiment(a):
    from .pipeline import ExperimentConfig, run_experiment
    cfg = ExperimentConfig(out_dir=a.out, photos_dir=a.photos, seed=a.seed, epochs=a.epochs,
                           n_test_videos=a.test_videos, test_frames=a.test_frames, train_frames=a.train_frames)
    run_experiment(cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fastfacetrack", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("extract", help="detect faces in a video and save the crops")
    s.add_argument("video"); s.add_argument("--out", default="faces.npz")
    s.add_argument("--detector", default="ultralight", choices=["ultralight", "ultralight-slim", "haar"])
    s.add_argument("--stride", type=int, default=1); s.add_argument("--min-size", type=int, default=24)
    s.add_argument("--images", help="also write every crop as a PNG into this folder")
    s.set_defaults(fn=_extract)

    s = sub.add_parser("cluster", help="k-means on extracted faces; writes a montage with one row per cluster")
    s.add_argument("faces"); s.add_argument("-k", type=int, required=True)
    s.add_argument("--features", default="hog", choices=["hog", "pixels"]); s.add_argument("--out", default="clusters.png")
    s.set_defaults(fn=_cluster)

    s = sub.add_parser("train", help="train classifiers: target faces vs other faces")
    s.add_argument("--target", nargs="+", required=True, help="face sets (.npz) of the target person")
    s.add_argument("--other", nargs="+", required=True, help="face sets (.npz) of everybody else")
    s.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    s.add_argument("--features", default="hog", choices=["hog", "pixels"])
    s.add_argument("--epochs", type=int, default=40); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="models")
    s.set_defaults(fn=_train)

    s = sub.add_parser("test", help="run detector + trained models on a video; writes a results CSV")
    s.add_argument("video"); s.add_argument("--models-dir", default="models")
    s.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    s.add_argument("--detectors", nargs="+", default=["ultralight"], choices=["ultralight", "ultralight-slim", "haar"])
    s.add_argument("--truth", help="ground-truth JSON (simulated videos)"); s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--stride", type=int, default=1); s.add_argument("--out", default="results.csv")
    s.set_defaults(fn=_test)

    s = sub.add_parser("simulate", help="write one synthetic test video with ground truth")
    s.add_argument("--out", default="simulated.mp4"); s.add_argument("--frames", type=int, default=300)
    s.add_argument("--others", type=int, default=3); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--detector", default="ultralight"); s.add_argument("--cache", default="demo_photos")
    s.set_defaults(fn=_simulate)

    s = sub.add_parser("experiment", help="full simulation: videos -> faces -> training -> test -> figures")
    s.add_argument("--out", default="results"); s.add_argument("--photos", help="folder with your own photos")
    s.add_argument("--seed", type=int, default=0); s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--train-frames", type=int, default=120); s.add_argument("--test-frames", type=int, default=400)
    s.add_argument("--test-videos", type=int, default=3)
    s.set_defaults(fn=_experiment)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
