"""Held-out benchmark of SWIFT-MO on synthetic scenes: ablations, background-subtraction baselines, and
the sequential classifier against fixed-length decisions. Seeds used here were not used to develop the method."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .moving import MovingConfig, _match, simulate_scene, train_class_model
from .swift import SPRTClassifier, SwiftMO


def _clip(cfg, n, noise_seed, n_objects=0):
    return simulate_scene(MovingConfig(**{**asdict(cfg), "n_frames": n}), n_objects, noise_seed=noise_seed)[0]


def _score(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def run_swift_variant(cfg, model, name, alpha=0.05, warmup=30, n_cal=100, n_test=100, fixed_thr=None,
                      compensate=True, overlap=0.5):
    shape = (cfg.height, cfg.width)
    sw = SwiftMO(shape, SPRTClassifier(model), alpha=alpha, compensate=compensate, overlap=overlap)
    stream = _clip(cfg, warmup + n_cal + n_test, 100)
    if fixed_thr is None:
        sw.calibrate(stream[:warmup + n_cal], warmup)
    else:
        sw.bg.init(stream[:warmup]); sw.threshold = fixed_thr
        for f in stream[warmup:warmup + n_cal]:
            sw.process(f)
    fa = sum(len(sw.process(f)) > 0 for f in stream[warmup + n_cal:])
    frames, truths = simulate_scene(cfg)
    sw.timing = {k: 0.0 for k in sw.timing}
    tp = fp = fn = 0
    track_obs: dict[int, list] = {}
    for f, tr in zip(frames, truths):
        dets = sw.process(f)
        a, b, c, pairs = _match(dets, tr)
        tp += a; fp += b; fn += c
        for d, g in pairs:
            track_obs.setdefault(d["track"], []).append((g["cls"], d))
    p, r, f1 = _score(tp, fp, fn)
    t = sw.mean_timing()
    row = {"method": name, "precision": p, "recall": r, "f1": f1, "false_alarm_rate": fa / n_test,
           "ms_per_frame": sum(t.values()), "threshold": sw.threshold}
    return row, track_obs, sw


def class_outcomes(track_obs, names):
    """Per track: truth = majority class; accuracy of the argmax after 1 and 5 frames, and of the SPRT."""
    out = []
    for obs in track_obs.values():
        truth = max(set(o[0] for o in obs), key=[o[0] for o in obs].count)
        row = {"truth": truth}
        for n in (1, 5):
            row[f"acc_{n}"] = names[int(np.argmax(obs[n - 1][1]["llr"]))] == truth if len(obs) >= n else None
        dec = [o for o in obs if o[1]["label"] is not None]
        row["sprt_label"] = dec[0][1]["label"] if dec else None
        row["sprt_correct"] = (dec[0][1]["label"] == truth) if dec else None
        row["sprt_frames"] = dec[0][1]["decided_at"] if dec else None
        out.append(row)
    return out


def run_background_baseline(cfg, kind, warmup=130, n_test=100, min_area=150):
    mk = (lambda: cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)) if kind == "mog2" \
        else (lambda: cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0, detectShadows=False))
    bs = mk()
    kernel = np.ones((3, 3), np.uint8)

    def boxes(frame):
        m = cv2.morphologyEx(bs.apply(frame), cv2.MORPH_OPEN, kernel)
        m = cv2.dilate(m, kernel, iterations=2)
        n, _, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
        return [{"box": (int(x), int(y), int(x + w), int(y + h))} for x, y, w, h, a in st[1:] if a >= min_area]

    stream = _clip(cfg, warmup + n_test, 100)
    for f in stream[:warmup]:
        bs.apply(f)
    fa = sum(len(boxes(f)) > 0 for f in stream[warmup:])
    frames, truths = simulate_scene(cfg)
    tp = fp = fn = 0
    t0 = time.perf_counter()
    for f, tr in zip(frames, truths):
        a, b, c, _ = _match(boxes(f), tr)
        tp += a; fp += b; fn += c
    ms = (time.perf_counter() - t0) * 1e3 / len(frames)
    p, r, f1 = _score(tp, fp, fn)
    return {"method": kind.upper(), "precision": p, "recall": r, "f1": f1, "false_alarm_rate": fa / n_test,
            "ms_per_frame": ms, "threshold": None}


def run_benchmark(out_dir="results/benchmark", seeds=range(2000, 2010), log=print):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, cls_rows = [], []
    for seed in seeds:
        cfg = MovingConfig(seed=seed)
        model = train_class_model(cfg)
        for name, kw in [("SWIFT-MO (calibrated)", {}), ("fixed threshold Z>3", {"fixed_thr": 3.0}),
                         ("no illumination compensation", {"compensate": False}),
                         ("no overlap suppression", {"overlap": 1.01})]:
            row, obs, sw = run_swift_variant(cfg, model, name, **kw)
            rows.append({"seed": seed, **row})
        for kind in ("mog2", "knn"):
            rows.append({"seed": seed, **run_background_baseline(cfg, kind)})
        for sep in (1.0, 0.25, 0.1):
            c2 = MovingConfig(seed=seed, class_sep=sep)
            m2 = train_class_model(c2)
            _, obs, sw = run_swift_variant(c2, m2, "full")
            for r in class_outcomes(obs, m2.names):
                cls_rows.append({"seed": seed, "class_sep": sep, **r})
        log(f"seed {seed} done")
    df = pd.DataFrame(rows); cf = pd.DataFrame(cls_rows)
    df.to_csv(out / "detection_per_scene.csv", index=False); cf.to_csv(out / "classification_per_track.csv", index=False)
    agg = df.groupby("method", sort=False)[["precision", "recall", "f1", "false_alarm_rate", "ms_per_frame"]].agg(["mean", "std"]).round(3)
    agg.to_csv(out / "detection_summary.csv")
    cagg = cf.groupby("class_sep").agg(tracks=("truth", "size"), acc_1=("acc_1", "mean"), acc_5=("acc_5", "mean"),
                                        sprt_decided=("sprt_correct", lambda x: x.notna().mean()),
                                        sprt_acc=("sprt_correct", lambda x: x.dropna().astype(float).mean()),
                                        sprt_frames=("sprt_frames", "mean")).round(3)
    cagg.to_csv(out / "classification_summary.csv")
    (out / "seeds.json").write_text(json.dumps(list(seeds)))
    log(agg.to_string()); log(cagg.to_string())
    return agg, cagg


def _null_maxima(sw, frames, record=True):
    """Maxima of the scan statistic over object-free frames, updating the background as in deployment."""
    from .swift import scan
    out = []
    for f in frames:
        (z, _), _ = scan(sw.bg.distance(f), sw.sizes, sw.stride)
        out.append(z)
        sw.bg.update(f)
    return out if record else None


def run_calibration_study(out_dir="results/calibration", seeds=range(3000, 3030), alpha=0.05, m=100, n_test=200, log=print):
    """Does the order-statistic threshold hold on later frames of the same object-free stream?

    mode A: calibrate on the m frames that follow a 30-frame initialisation (as in the benchmark);
    modes B and C: discard a further 100 or 300 frames of burn-in first, so the background settles.
    The achieved false-alarm probability is the share of the test-frame maxima above the threshold.
    """
    from .swift import conformal_threshold
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows, maxima = [], {}
    for seed in seeds:
        cfg = MovingConfig(seed=seed, n_frames=30 + 300 + m + n_test)
        stream, _ = simulate_scene(cfg, 0)
        shape = (cfg.height, cfg.width)
        for mode, burn in (("A", 0), ("B", 100), ("C", 300)):
            sw = SwiftMO(shape, alpha=alpha)
            sw.bg.init(stream[:30])
            if burn:
                _null_maxima(sw, stream[30:30 + burn], record=False)
            a = 30 + burn
            cal = _null_maxima(sw, stream[a:a + m])
            c = conformal_threshold(cal, alpha)
            test = _null_maxima(sw, stream[a + m:a + m + n_test])
            rows.append({"seed": seed, "mode": mode, "threshold": c, "cal_median": float(np.median(cal)),
                         "test_median": float(np.median(test)), "test_max": float(np.max(test)),
                         "exceed": float(np.mean(np.array(test) > c))})
            if seed == seeds[0]:
                maxima[mode] = {"cal": cal, "test": test, "c": c}
        log(f"seed {seed}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration_study.csv", index=False)
    (out / "example_maxima.json").write_text(json.dumps(maxima))
    s = df.groupby("mode").agg(threshold=("threshold", "mean"), test_median=("test_median", "mean"), test_max=("test_max", "mean"),
                               exceed_mean=("exceed", "mean"), exceed_sd=("exceed", "std"), exceed_max=("exceed", "max")).round(4)
    s.to_csv(out / "calibration_summary.csv"); log(s.to_string())
    return df, s
