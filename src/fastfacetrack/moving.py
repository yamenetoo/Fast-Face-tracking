"""Synthetic moving-object scenes with exact ground truth, and the SWIFT-MO experiment on them."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .detectors import iou
from .swift import ClassModel, SPRTClassifier, SwiftMO

# three object classes with overlapping colour distributions (BGR means, per-channel sd)
_BASE = {"red": ((70, 90, 190), 22), "green": ((80, 170, 100), 22), "yellow": ((70, 190, 200), 22)}
CLASSES = _BASE


def class_params(sep: float = 1.0):
    """Class colour (mean, sd), with the means pulled towards their grand mean by ``sep``."""
    g = np.mean([m for m, _ in _BASE.values()], axis=0)
    return {k: (tuple(g + sep * (np.array(m) - g)), sd) for k, (m, sd) in _BASE.items()}


@dataclass
class MovingConfig:
    height: int = 240
    width: int = 320
    n_frames: int = 240
    n_objects: int = 3
    noise_sd: float = 5.0
    seed: int = 0
    min_size: int = 28
    max_size: int = 64
    speed: float = 4.0
    noise_seed: int | None = None     # None: same as seed
    class_sep: float = 1.0            # 1 = default colour separation; smaller = harder classes


def make_background(cfg: MovingConfig, rng):
    base = rng.uniform(80, 150, (cfg.height // 8 + 1, cfg.width // 8 + 1, 3)).astype(np.float32)
    import cv2
    bg = cv2.resize(base, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
    return bg


def render(cfg: MovingConfig, bg, objects, rng, t):
    frame = bg + 6 * np.sin(t / 40.0) + rng.normal(0, cfg.noise_sd, bg.shape).astype(np.float32)
    truth = []
    for o in objects:
        x0, y0 = int(o["x"]), int(o["y"])
        w, h = o["w"], o["h"]
        yy, xx = np.mgrid[0:h, 0:w]
        ell = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 <= 1.0
        mean, sd = class_params(cfg.class_sep)[o["cls"]]
        patch = rng.normal(mean, sd, (h, w, 3)).astype(np.float32)
        region = frame[y0:y0 + h, x0:x0 + w]
        region[ell] = patch[ell]
        truth.append({"box": (x0, y0, x0 + w, y0 + h), "cls": o["cls"], "id": o["id"]})
    return np.clip(frame, 0, 255).astype(np.uint8), truth


def step_objects(cfg: MovingConfig, objects):
    for o in objects:
        o["x"] += o["vx"]; o["y"] += o["vy"]
        if o["x"] < 0 or o["x"] + o["w"] >= cfg.width:
            o["vx"] *= -1; o["x"] = np.clip(o["x"], 0, cfg.width - o["w"] - 1)
        if o["y"] < 0 or o["y"] + o["h"] >= cfg.height:
            o["vy"] *= -1; o["y"] = np.clip(o["y"], 0, cfg.height - o["h"] - 1)


def new_objects(cfg: MovingConfig, rng, classes, n):
    objs = []
    for i in range(n):
        w = int(rng.integers(cfg.min_size, cfg.max_size)); h = int(rng.integers(cfg.min_size, cfg.max_size))
        ang = rng.uniform(0, 2 * np.pi)
        objs.append({"id": i, "cls": classes[i % len(classes)], "w": w, "h": h,
                     "x": rng.uniform(0, cfg.width - w - 1), "y": rng.uniform(0, cfg.height - h - 1),
                     "vx": cfg.speed * np.cos(ang), "vy": cfg.speed * np.sin(ang)})
    return objs


def simulate_scene(cfg: MovingConfig, n_objects: int | None = None, noise_seed: int | None = None):
    """Return (frames, truth). The scene (background, objects) comes from ``cfg.seed``; the pixel noise
    from ``noise_seed``. ``n_objects=0`` gives a background-only clip of the same scene."""
    rng = np.random.default_rng(cfg.seed)
    bg = make_background(cfg, rng)
    n = cfg.n_objects if n_objects is None else n_objects
    objects = new_objects(cfg, rng, list(CLASSES), n)
    nrng = np.random.default_rng(cfg.seed + 7919 if noise_seed is None else noise_seed)
    frames, truths = [], []
    for t in range(cfg.n_frames):
        f, tr = render(cfg, bg, objects, nrng, t)
        frames.append(f); truths.append(tr)
        step_objects(cfg, objects)
    return frames, truths


def train_class_model(cfg: MovingConfig, n=4000):
    rng = np.random.default_rng(cfg.seed + 1)
    return ClassModel.fit({k: rng.normal(m, sd, (n, 3)) for k, (m, sd) in class_params(cfg.class_sep).items()})


def _match(dets, truth, thr=0.5):
    used, tp, pairs = set(), 0, []
    for d in dets:
        best, bi = 0.0, None
        for j, g in enumerate(truth):
            v = iou(d["box"], g["box"])
            if j not in used and v > best:
                best, bi = v, j
        if bi is not None and best >= thr:
            used.add(bi); tp += 1; pairs.append((d, truth[bi]))
    return tp, len(dets) - tp, len(truth) - len(used), pairs


def run_moving_experiment(out_dir="results_moving", cfg: MovingConfig | None = None, alpha=0.05,
                          n_calibration=100, n_empty_clips=5, warmup=30, log=print) -> dict:
    """Calibrate on an object-free clip, check the false-alarm rate on fresh object-free clips,
    then detect, segment, track and classify three moving objects."""
    cfg = cfg or MovingConfig()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shape = (cfg.height, cfg.width)
    model = train_class_model(cfg)
    clip = lambda n, seed: simulate_scene(MovingConfig(**{**asdict(cfg), "n_frames": n}), 0, noise_seed=seed)[0]

    # 1-2. calibrate on the first part of an object-free stream, then test false alarms on its continuation
    #      (the conformal guarantee is for later frames of the same stationary scene), repeated over scenes
    fa = tested = 0
    thrs = []
    for k in range(n_empty_clips):
        sc = MovingConfig(**{**asdict(cfg), "seed": cfg.seed + 50 + k, "n_frames": warmup + n_calibration + 100})
        frames, _ = simulate_scene(sc, 0)
        t = SwiftMO(shape, SPRTClassifier(model), alpha=alpha)
        thrs.append(t.calibrate(frames[:warmup + n_calibration], warmup))
        for f in frames[warmup + n_calibration:]:
            fa += len(t.process(f)) > 0; tested += 1
    far = fa / tested
    log(f"false-alarm rate per object-free frame: {far:.3f} (target <= {alpha}; {n_empty_clips} scenes, {tested} frames)")

    # calibrated threshold for the moving-object scene (its own object-free clip)
    sw = SwiftMO(shape, SPRTClassifier(model), alpha=alpha)
    thr = sw.calibrate(clip(warmup + n_calibration, 100), warmup)
    log(f"threshold for the evaluation scene: c = {thr:.2f}")

    # 3. moving objects ---------------------------------------------------------------------------
    frames, truths = simulate_scene(cfg)
    scene = SwiftMO(shape, SPRTClassifier(model), alpha=alpha)
    scene.bg = sw.bg; scene.threshold = thr   # background and threshold from the object-free clip
    tp = fp = fn = 0
    votes: dict[int, list] = {}
    detections = []
    for f, tr in zip(frames, truths):
        dets = scene.process(f)
        detections.append(dets)
        a, b, c, pairs = _match(dets, tr)
        tp += a; fp += b; fn += c
        for d, g in pairs:
            votes.setdefault(d["track"], []).append((g["cls"], d["label"], d["decided_at"]))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    correct = total = 0; delays = []
    for v in votes.values():
        truth_cls = max(set(x[0] for x in v), key=[x[0] for x in v].count)
        decided = [x for x in v if x[1] is not None]
        if decided:
            total += 1; correct += decided[0][1] == truth_cls; delays.append(decided[0][2])
    timing = scene.mean_timing()
    nwin = int(sum(len(range(0, shape[0] - h + 1, scene.stride)) * len(range(0, shape[1] - w + 1, scene.stride))
                   for h, w in scene.sizes))
    summary = {"alpha": alpha, "threshold": round(thr, 3), "empty_scenes": n_empty_clips, "empty_frames_tested": tested, "empty_false_alarm_rate": round(far, 4),
               "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
               "tracks_decided": total, "tracks_correct": correct,
               "sprt_mean_frames_to_decision": float(np.mean(delays)) if delays else None,
               **{k: round(v, 3) for k, v in timing.items()}, "total_ms_per_frame": round(sum(timing.values()), 3),
               "windows_per_frame": nwin, "frame_size": [cfg.width, cfg.height]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))
    return {"summary": summary, "frames": frames, "truths": truths, "detections": detections, "scene": scene}


def annotate_moving(frame, truth, dets):
    """Draw truth (green) and detections (red, with the SPRT label once decided) on a copy of ``frame``."""
    import cv2
    img = frame.copy()
    for g in truth:
        x0, y0, x1, y1 = g["box"]; cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), 1)
    for d in dets:
        x0, y0, x1, y1 = d["box"]; cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, d["label"] or "?", (x0, max(y0 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img
