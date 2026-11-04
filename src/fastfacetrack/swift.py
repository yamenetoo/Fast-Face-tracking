"""SWIFT-MO: fast statistical detection, segmentation and classification of moving objects.

Every sub-window of a frame is scored in O(1) from integral images of sufficient statistics:

* ``motion``: sum of per-pixel squared Mahalanobis distances to an adaptive Gaussian background,
  standardised into a scan statistic ``Z(W)``;
* ``class``: Gaussian log-likelihood of the window under a pre-trained cluster (``window_loglik``).

The detection threshold is a finite-sample, distribution-free quantile of the maximum of ``Z`` over
background-only frames (see ``calibrate``). Classes are decided over time with Wald's SPRT.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

from .detectors import iou


# --------------------------------------------------------------------------------------------------
# integral images and O(1) window statistics
# --------------------------------------------------------------------------------------------------
def _box_sum(I, y0, x0, y1, x1):
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]


class IntegralStats:
    """Integral images of x and x x^T for a (H, W, d) feature image."""

    def __init__(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=np.float64)
        self.d = feat.shape[2]
        chans = [feat[..., i] for i in range(self.d)]
        chans += [feat[..., i] * feat[..., j] for i in range(self.d) for j in range(i, self.d)]
        self.I = [cv2.integral(c, sdepth=cv2.CV_64F) for c in chans]

    def window(self, y0, x0, y1, x1):
        """Return ``n, S1 (d,), S2 (d, d)`` of the window [y0:y1, x0:x1]."""
        d = self.d
        s1 = np.array([_box_sum(self.I[i], y0, x0, y1, x1) for i in range(d)])
        s2 = np.zeros((d, d))
        k = d
        for i in range(d):
            for j in range(i, d):
                s2[i, j] = s2[j, i] = _box_sum(self.I[k], y0, x0, y1, x1)
                k += 1
        return (y1 - y0) * (x1 - x0), s1, s2


def gaussian_loglik_from_stats(n, s1, s2, mu, cov) -> float:
    """Log-likelihood of ``n`` iid N(mu, cov) pixels given only their sufficient statistics."""
    d = len(mu)
    prec = np.linalg.inv(cov)
    q = s2 - np.outer(s1, mu) - np.outer(mu, s1) + n * np.outer(mu, mu)
    return float(-0.5 * (n * (d * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1]) + np.trace(prec @ q)))


def window_loglik(stats: IntegralStats, box, mu, cov) -> float:
    x0, y0, x1, y1 = box
    n, s1, s2 = stats.window(y0, x0, y1, x1)
    return gaussian_loglik_from_stats(n, s1, s2, mu, cov)


# --------------------------------------------------------------------------------------------------
# adaptive background and the motion scan statistic
# --------------------------------------------------------------------------------------------------
class Background:
    """Per-pixel Gaussian background with exponential forgetting (diagonal covariance)."""

    def __init__(self, rate: float = 0.02, min_var: float = 4.0):
        self.rate, self.min_var = rate, min_var
        self.mean = self.var = None

    def init(self, frames: Sequence[np.ndarray]):
        a = np.stack([f.astype(np.float32) for f in frames])
        self.mean = a.mean(0)
        self.var = np.maximum(a.var(0), self.min_var)
        return self

    def distance(self, frame: np.ndarray, cap: float = 60.0) -> np.ndarray:
        """Squared Mahalanobis distance per pixel, summed over channels (chi2 with d dof)."""
        diff = frame.astype(np.float32) - self.mean
        diff = diff - np.median(diff[::4, ::4].reshape(-1, diff.shape[2]), axis=0)   # global illumination shift
        return np.minimum((diff * diff / self.var).sum(-1), cap).astype(np.float64)

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None):
        f = frame.astype(np.float32)
        r = np.full(f.shape[:2], self.rate, np.float32)
        if mask is not None:
            r[mask] = self.rate * 0.05          # barely learn the pixels of detected objects
        r = r[..., None]
        diff = f - self.mean
        self.mean += r * diff
        self.var = np.maximum(self.var + r * (diff * diff - self.var), self.min_var)


def make_sizes(shape, min_size=16, max_frac=0.6, scale=1.3, aspects=(1.0, 0.6, 1.6)):
    H, W = shape[:2]
    out, s = [], float(min_size)
    while s <= max_frac * min(H, W):
        for a in aspects:
            h, w = int(round(s * a ** 0.5)), int(round(s / a ** 0.5))
            if h < H and w < W:
                out.append((h, w))
        s *= scale
    return sorted(set(out))


def scan(dmap: np.ndarray, sizes, stride=4, dof=3, min_z=None):
    """Z(W) for every window; return the best one (max_z, box) and, if ``min_z``, all windows above it."""
    I = cv2.integral(dmap, sdepth=cv2.CV_64F)
    H, W = dmap.shape
    best, cand = (-np.inf, None), []
    for h, w in sizes:
        ys = np.arange(0, H - h + 1, stride)
        xs = np.arange(0, W - w + 1, stride)
        if len(ys) == 0 or len(xs) == 0:
            continue
        S = I[np.ix_(ys + h, xs + w)] - I[np.ix_(ys, xs + w)] - I[np.ix_(ys + h, xs)] + I[np.ix_(ys, xs)]
        n = h * w
        Z = (S - n * dof) / np.sqrt(2.0 * n * dof)
        k = np.unravel_index(np.argmax(Z), Z.shape)
        if Z[k] > best[0]:
            best = (float(Z[k]), (int(xs[k[1]]), int(ys[k[0]]), int(xs[k[1]] + w), int(ys[k[0]] + h)))
        if min_z is not None:
            Zf = Z.astype(np.float32)
            local = Zf >= cv2.dilate(Zf, np.ones((max(1, h // stride), max(1, w // stride)), np.uint8))
            for iy, ix in zip(*np.nonzero(local & (Z > min_z))):          # local maxima over a window-sized area
                cand.append((float(Z[iy, ix]), (int(xs[ix]), int(ys[iy]), int(xs[ix] + w), int(ys[iy] + h))))
    return best, cand


def suppress_overlaps(cands, thr=0.5):
    """Greedy suppression by intersection over the smaller area: drops sub-windows of a better window
    and windows that merely swallow it. ``cands`` = [(z, (x0, y0, x1, y1))], any order."""
    kept = []
    for z, b in sorted(cands, key=lambda c: -c[0]):
        a_b = (b[2] - b[0]) * (b[3] - b[1])
        ok = True
        for _, k in kept:
            ix = max(0, min(b[2], k[2]) - max(b[0], k[0])); iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            a_k = (k[2] - k[0]) * (k[3] - k[1])
            if ix * iy / min(a_b, a_k) >= thr:
                ok = False
                break
        if ok:
            kept.append((z, b))
    return kept


def conformal_threshold(maxima: Iterable[float], alpha: float = 0.05) -> float:
    """Order statistic ceil((m+1)(1-alpha)) of m exchangeable null maxima: P(M_new > c) <= alpha."""
    m = np.sort(np.asarray(list(maxima), dtype=float))
    k = int(np.ceil((len(m) + 1) * (1 - alpha)))
    if k > len(m):
        raise ValueError(f"need at least {int(np.ceil(1 / alpha)) - 1} calibration frames for alpha={alpha}")
    return float(m[k - 1])


# --------------------------------------------------------------------------------------------------
# classification with SPRT and simple tracking
# --------------------------------------------------------------------------------------------------
@dataclass
class ClassModel:
    names: list
    means: np.ndarray
    covs: np.ndarray

    @classmethod
    def fit(cls, pixels_by_class: dict):
        names = list(pixels_by_class)
        means = np.stack([pixels_by_class[n].mean(0) for n in names])
        covs = np.stack([np.cov(pixels_by_class[n].T) + 1e-3 * np.eye(pixels_by_class[n].shape[1]) for n in names])
        return cls(names, means, covs)

    def loglik(self, pixels: np.ndarray, n_eff: float = 20.0) -> np.ndarray:
        """Per-class log-likelihood of a pixel set, scaled to ``n_eff`` effective independent pixels."""
        n, d = pixels.shape
        s1 = pixels.sum(0)
        s2 = pixels.T @ pixels
        ll = np.array([gaussian_loglik_from_stats(n, s1, s2, m, c) for m, c in zip(self.means, self.covs)])
        return ll * min(1.0, n_eff / n)


@dataclass
class Track:
    box: tuple
    llr: np.ndarray                     # accumulated log-likelihood per class
    age: int = 0
    missed: int = 0
    decided: int | None = None
    decided_at: int | None = None
    history: list = field(default_factory=list)


class SPRTClassifier:
    """Wald SPRT of class k against all others, run per track on accumulated log-likelihoods."""

    def __init__(self, model: ClassModel, alpha=0.01, beta=0.01):
        self.model = model
        self.upper = np.log((1 - beta) / alpha)

    def step(self, track: Track, pixels: np.ndarray):
        ll = self.model.loglik(pixels)
        track.llr = track.llr + ll
        if track.decided is None:
            k = int(np.argmax(track.llr))
            others = np.delete(track.llr, k)
            lam = track.llr[k] - np.logaddexp.reduce(others)    # log-odds of k against the rest
            if lam >= self.upper:
                track.decided, track.decided_at = k, track.age
        return ll


class SwiftMO:
    """Frame-by-frame detector, segmenter and classifier of moving objects."""

    def __init__(self, shape, classifier: SPRTClassifier | None = None, alpha=0.05, stride=4,
                 sizes=None, rate=0.02, overlap=0.5, max_missed=5):
        self.shape = shape
        self.sizes = sizes or make_sizes(shape)
        self.stride, self.alpha, self.overlap, self.max_missed = stride, alpha, overlap, max_missed
        self.bg = Background(rate)
        self.cls = classifier
        self.threshold = None
        self.tracks: list[Track] = []
        self.timing = {"background_ms": 0.0, "scan_ms": 0.0, "classify_ms": 0.0, "frames": 0}

    # -- calibration on frames that contain no moving object --------------------------------------
    def calibrate(self, background_frames: Sequence[np.ndarray], warmup=30):
        self.bg.init(background_frames[:warmup])
        maxima = []
        for f in background_frames[warmup:]:
            (z, _), _ = scan(self.bg.distance(f), self.sizes, self.stride)
            maxima.append(z)
            self.bg.update(f)
        self.threshold = conformal_threshold(maxima, self.alpha)
        self.null_maxima = np.array(maxima)
        return self.threshold

    def process(self, frame: np.ndarray):
        t0 = time.perf_counter()
        dmap = self.bg.distance(frame)
        t1 = time.perf_counter()
        _, cand = scan(dmap, self.sizes, self.stride, min_z=self.threshold)
        boxes = []
        if cand:
            boxes = suppress_overlaps(cand, self.overlap)
        t2 = time.perf_counter()
        out, used = [], set()
        fg = np.zeros(frame.shape[:2], bool)
        for z, box in boxes:
            x0, y0, x1, y1 = box
            mask = dmap[y0:y1, x0:x1] > 9.0
            fg[y0:y1, x0:x1] |= mask
            ti = max(range(len(self.tracks)), key=lambda i: iou(box, self.tracks[i].box) if i not in used else -1,
                     default=None)
            if ti is not None and ti not in used and iou(box, self.tracks[ti].box) > 0.2:
                tr = self.tracks[ti]
            else:
                tr = Track(box, np.zeros(len(self.cls.model.names)) if self.cls else np.zeros(1))
                self.tracks.append(tr)
                ti = len(self.tracks) - 1
            used.add(ti)
            tr.box, tr.missed, tr.age = box, 0, tr.age + 1
            if self.cls is not None and mask.sum() >= 20:
                pix = frame[y0:y1, x0:x1][mask].astype(np.float64)
                self.cls.step(tr, pix)
            out.append({"box": box, "z": z, "track": ti, "label": None if tr.decided is None else self.cls.model.names[tr.decided],
                        "decided_at": tr.decided_at, "age": tr.age, "mask_pixels": int(mask.sum())})
        for i, tr in enumerate(self.tracks):
            if i not in used:
                tr.missed += 1
        self.bg.update(frame, fg)
        t3 = time.perf_counter()
        self.timing["background_ms"] += (t1 - t0) * 1e3
        self.timing["scan_ms"] += (t2 - t1) * 1e3
        self.timing["classify_ms"] += (t3 - t2) * 1e3
        self.timing["frames"] += 1
        return out

    def mean_timing(self):
        n = max(self.timing["frames"], 1)
        return {k: v / n for k, v in self.timing.items() if k != "frames"}
