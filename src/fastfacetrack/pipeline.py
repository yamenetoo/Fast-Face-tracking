"""End-to-end experiment: simulate -> extract -> (cluster) -> train -> test -> report."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .clustering import cluster_faces, cluster_purity, montage
from .detectors import make_detector
from .evaluation import combine_results, evaluate_video, run_detector, load_truth
from .extraction import extract_faces
from .models import MODEL_NAMES
from .report import annotate_frames, plot_model_comparison, plot_timeline
from .simulation import IdentityBank, SimulationConfig, simulate_video
from .training import make_dataset, train_models


@dataclass
class ExperimentConfig:
    out_dir: str = "results"
    photos_dir: str | None = None                 # folder with your own photos; None = public demo photos
    train_detector: str = "ultralight"            # detector used to build the training set
    test_detectors: tuple[str, ...] = ("ultralight", "ultralight-slim", "haar")
    models: tuple[str, ...] = MODEL_NAMES
    features: str = "hog"
    seed: int = 0
    min_identity_width: int = 60                  # use photo faces at least this wide as identities
    n_train_others: int = 5                       # people seen in training (besides the target)
    n_test_unseen: int = 3                        # people that appear only at test time
    train_frames: int = 120                       # frames per training clip (one clip per person)
    test_frames: int = 400
    n_test_videos: int = 3
    test_seen_others: int = 2
    test_unseen_others: int = 2
    threshold: float = 0.5
    epochs: int = 40
    width: int = 640
    height: int = 480


def run_experiment(cfg: ExperimentConfig | None = None, log=print) -> dict:
    """Run the whole simulation and write tables and figures into ``cfg.out_dir``."""
    cfg = cfg or ExperimentConfig()
    out = Path(cfg.out_dir)
    for sub in ("videos", "faces", "tables", "figures", "models"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t_start = time.perf_counter()

    # 1. identities ------------------------------------------------------------------------------
    detector = make_detector(cfg.train_detector)
    if cfg.photos_dir:
        photos = sorted(p for p in Path(cfg.photos_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        bank = IdentityBank.from_images(photos, detector, min_width=cfg.min_identity_width)
    else:
        bank = IdentityBank.demo(detector, cache_dir=out / "demo_photos", min_width=cfg.min_identity_width)
    need = 1 + cfg.n_train_others + cfg.n_test_unseen
    if len(bank) < need:
        raise ValueError(f"only {len(bank)} usable faces in the photos, {need} required")
    rng = np.random.default_rng(cfg.seed)
    names = [i.name for i in bank.identities]
    rng.shuffle(names)
    target, train_others = names[0], names[1:1 + cfg.n_train_others]
    unseen = names[1 + cfg.n_train_others:need]
    log(f"identities: {len(bank)} usable | target={target} | seen others={train_others} | unseen={unseen}")

    # 2-3. training clips (one person per clip) and face extraction -------------------------------
    sets, owner = {}, {}
    for k, name in enumerate([target] + train_others):
        video = out / "videos" / f"train_{name}.mp4"
        sim = SimulationConfig(width=cfg.width, height=cfg.height, n_frames=cfg.train_frames,
                               seed=cfg.seed * 1000 + k, presence_probability=1.0)
        simulate_video(bank, name, [], video, sim, target_always_present=True)
        fs = extract_faces(video, detector)
        fs.save(out / "faces" / f"{name}.npz")
        sets[name] = fs
        log(f"  clip {name}: {len(fs)} faces extracted")

    # 4. optional clustering check ------------------------------------------------------------------
    all_faces = np.concatenate([sets[n].faces for n in sets])
    truth_ids = np.concatenate([[n] * len(sets[n]) for n in sets])
    labels, info = cluster_faces(all_faces, len(sets), cfg.features, seed=cfg.seed)
    purity = cluster_purity(labels, truth_ids)
    cv2.imwrite(str(out / "figures" / "cluster_montage.png"), montage(all_faces, labels))
    log(f"clustering ({len(sets)} clusters): purity={purity:.3f}, silhouette={info['silhouette']:.3f}")

    # 5. training --------------------------------------------------------------------------------
    faces, y, groups = make_dataset([sets[target]], [sets[n] for n in train_others])
    classifiers, train_report = train_models(faces, y, groups, cfg.models, cfg.features, cfg.seed,
                                             epochs=cfg.epochs)
    train_report.to_csv(out / "tables" / "training_report.csv", index=False)
    for clf in classifiers.values():
        clf.save(out / "models")
    log("training done:\n" + train_report[["model", "train_seconds", "val_accuracy", "val_f1"]].to_string(index=False))

    # 6-7. test videos and evaluation ----------------------------------------------------------------
    summaries, frames_tables = [], []
    first_video = None
    for v in range(cfg.n_test_videos):
        r = np.random.default_rng(cfg.seed * 7919 + v)
        actors = [str(a) for a in r.choice(train_others, cfg.test_seen_others, replace=False)] + \
                 [str(a) for a in r.choice(unseen, cfg.test_unseen_others, replace=False)]
        video = out / "videos" / f"test_{v}.mp4"
        sim = SimulationConfig(width=cfg.width, height=cfg.height, n_frames=cfg.test_frames,
                               seed=cfg.seed * 1000 + 500 + v)
        simulate_video(bank, target, actors, video, sim)
        first_video = first_video or video
        for dname in cfg.test_detectors:
            det = make_detector(dname)
            summary, per_frame = evaluate_video(video, det, classifiers, video.with_suffix(".truth.json"),
                                                cfg.threshold)
            summary.insert(0, "video", v)
            per_frame.insert(0, "video", v)
            per_frame.insert(1, "detector", det.name)
            summaries.append(summary)
            frames_tables.append(per_frame)
        log(f"test video {v}: actors={actors}")
    summary = combine_results(summaries, out / "tables" / "test_summary.csv")
    per_frame = combine_results(frames_tables, out / "tables" / "per_frame.csv")

    aggregate = (summary.groupby(["detector", "model"], sort=False)
                 .agg(["mean", "std"], numeric_only=True).round(4))
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.reset_index().to_csv(out / "tables" / "test_aggregate.csv", index=False)

    # 8. figures ------------------------------------------------------------------------------------
    main = summary[summary["detector"] == detector.name]
    plot_model_comparison(main, out / "figures" / "model_comparison.png")
    best = main.groupby("model")["det_f1"].mean().idxmax()
    sub = per_frame[(per_frame["detector"] == detector.name) & (per_frame["video"] == 0)]
    plot_timeline(sub, best, out / "figures" / f"timeline_{best}.png", cfg.threshold)
    frames0 = run_detector(first_video, detector)
    present = [f for f, t in load_truth(first_video.with_suffix(".truth.json")).items() if any(x["target"] for x in t)]
    pick = [present[int(i)] for i in np.linspace(0, len(present) - 1, 6)] if present else [0]
    annotate_frames(first_video, frames0, classifiers[best], out / "figures" / "annotated_frames.png", pick, cfg.threshold)

    meta = {"best_model": best, "cluster_purity": purity, "cluster_silhouette": info["silhouette"],
            "identities": {"target": target, "seen_others": train_others, "unseen": unseen},
            "elapsed_seconds": round(time.perf_counter() - t_start, 1)}
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['elapsed_seconds']} s; best model by F1: {best}")
    return {"train_report": train_report, "summary": summary, "aggregate": aggregate, "per_frame": per_frame, "meta": meta}
