"""Synthetic videos with known ground truth.

Real face photos are pasted onto a moving, noisy background and animated (translation with bouncing,
scale and rotation changes, lighting changes, blur, people entering and leaving the scene). The output is
a normal MP4 file plus a JSON file with the exact face boxes and identities of every frame, so the whole
pipeline (detect -> classify -> report) can be tested on a computer without any private footage.

Identity source: every face a detector finds in a set of photos becomes one "person". The demo bank
downloads three public group photos from the UltraLight detector repository; use
``IdentityBank.from_images`` for your own photos.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detectors import FaceDetector
from .video import write_video

DEMO_PHOTO_BASE = ("https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/"
                   "master/imgs/")
DEMO_PHOTOS = ("1.jpg", "2.jpg", "3.jpg")


@dataclass
class Identity:
    """One person: a face patch (face plus margin) and where the face sits inside the patch."""

    name: str
    patch: np.ndarray            # BGR
    face_box: tuple[int, int, int, int]
    source_width: int            # width of the original face in pixels (a quality indicator)


class IdentityBank:
    def __init__(self, identities: list[Identity]):
        self.identities = identities

    def __len__(self) -> int:
        return len(self.identities)

    def by_name(self, name: str) -> Identity:
        return next(i for i in self.identities if i.name == name)

    @classmethod
    def from_images(cls, paths: list[str | Path], detector: FaceDetector, min_width: int = 20,
                    margin: float = 0.4, prefix: str = "person") -> "IdentityBank":
        """Every detected face (at least ``min_width`` pixels wide) in the photos becomes an identity."""
        found: list[Identity] = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            h, w = image.shape[:2]
            for det in sorted(detector.detect(image), key=lambda d: (d.y1 // 40, d.x1)):
                if det.width < min_width:
                    continue
                m = int(margin * max(det.width, det.height))
                xa, ya, xb, yb = max(0, det.x1 - m), max(0, det.y1 - m), min(w, det.x2 + m), min(h, det.y2 + m)
                found.append(Identity(f"{prefix}_{len(found):03d}", image[ya:yb, xa:xb].copy(),
                                      (det.x1 - xa, det.y1 - ya, det.x2 - xa, det.y2 - ya), det.width))
        return cls(found)

    @classmethod
    def demo(cls, detector: FaceDetector, cache_dir: str | Path = "demo_photos", min_width: int = 20) -> "IdentityBank":
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for name in DEMO_PHOTOS:
            dest = cache_dir / name
            if not dest.exists():
                urllib.request.urlretrieve(DEMO_PHOTO_BASE + name, dest)  # noqa: S310 - fixed https URL
            paths.append(dest)
        return cls.from_images(paths, detector, min_width=min_width)


@dataclass
class SimulationConfig:
    width: int = 640
    height: int = 480
    fps: float = 15.0
    n_frames: int = 300
    seed: int = 0
    face_width: tuple[float, float] = (70.0, 110.0)   # displayed width of the face box in pixels
    speed: tuple[float, float] = (1.5, 4.0)           # pixels per frame
    rotation_deg: float = 12.0
    scale_variation: float = 0.18
    brightness_variation: float = 0.30
    blur_probability: float = 0.15                    # share of frames with motion blur
    presence_probability: float = 0.6                 # share of frames in which an actor is on screen
    noise_sigma: float = 4.0


@dataclass
class _Actor:
    identity: Identity
    is_target: bool
    base_width: float
    pos: np.ndarray
    vel: np.ndarray
    phase: np.ndarray
    present: np.ndarray                                # boolean per frame


def _background(cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Large, smooth, coloured texture that is scrolled to imitate camera motion."""
    h, w = cfg.height * 2, cfg.width * 2
    base = rng.integers(60, 190, size=3)
    layers = []
    for k, sigma in enumerate((60, 25, 8, 2)):
        noise = rng.standard_normal((h, w, 3)).astype(np.float32)
        layers.append(cv2.GaussianBlur(noise, (0, 0), sigma) * (sigma ** 0.5) * (26 / (k + 1)))
    img = base[None, None, :].astype(np.float32) + sum(layers)
    return np.clip(img, 0, 255).astype(np.uint8)


def _presence(cfg: SimulationConfig, rng: np.random.Generator, force_always: bool = False) -> np.ndarray:
    """Alternating on/off segments whose on-share is about ``presence_probability``."""
    if force_always:
        return np.ones(cfg.n_frames, bool)
    on = np.zeros(cfg.n_frames, bool)
    t, state = int(rng.integers(0, 30)), rng.random() < cfg.presence_probability
    while t < cfg.n_frames:
        length = int(rng.integers(25, 70))
        on[t:t + length] = state
        t += length
        state = not state if rng.random() < 0.85 else state
    return on


def simulate_video(bank: IdentityBank, target: str | None, others: list[str], out_path: str | Path,
                   cfg: SimulationConfig | None = None, target_always_present: bool = False,
                   others_always_present: bool = False) -> dict:
    """Render a video and write ``<out_path>.truth.json`` next to it. Returns the truth dictionary.

    ``target`` may be ``None`` (a video with only other people). ``*_always_present`` keep the actors on
    screen for the whole video (useful for building training clips).
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    actors = []
    names = ([target] if target else []) + list(others)
    for name in names:
        is_target = name == target
        always = target_always_present if is_target else others_always_present
        actors.append(_Actor(
            bank.by_name(name), is_target, float(rng.uniform(*cfg.face_width)),
            rng.uniform([cfg.width * 0.2, cfg.height * 0.25], [cfg.width * 0.8, cfg.height * 0.75]),
            rng.choice([-1, 1], 2) * rng.uniform(*cfg.speed, 2),
            rng.uniform(0, 2 * np.pi, 4), _presence(cfg, rng, always)))

    background = _background(cfg, rng)
    drift = rng.uniform(-1.2, 1.2, 2)
    frames_out, truth_frames = [], []
    margin = 0.25

    for t in range(cfg.n_frames):
        ox = int((cfg.width / 2 + drift[0] * t) % cfg.width)
        oy = int((cfg.height / 2 + drift[1] * t) % cfg.height)
        frame = background[oy:oy + cfg.height, ox:ox + cfg.width].astype(np.float32).copy()
        drawn = []                       # (actor, alpha mask, box) in drawing order
        for a in actors:
            a.pos += a.vel
            for axis, limit in ((0, cfg.width), (1, cfg.height)):
                if a.pos[axis] < limit * margin or a.pos[axis] > limit * (1 - margin):
                    a.vel[axis] *= -1
                    a.pos[axis] = np.clip(a.pos[axis], limit * margin, limit * (1 - margin))
            if not a.present[t]:
                continue
            scale = (a.base_width / a.identity.source_width) * (1 + cfg.scale_variation * np.sin(0.05 * t + a.phase[0]))
            angle = cfg.rotation_deg * np.sin(0.04 * t + a.phase[1])
            gain = 1 + cfg.brightness_variation * np.sin(0.03 * t + a.phase[2])
            patch = a.identity.patch.astype(np.float32)
            ph, pw = patch.shape[:2]
            fx1, fy1, fx2, fy2 = a.identity.face_box
            centre = np.array([(fx1 + fx2) / 2, (fy1 + fy2) / 2])
            m = cv2.getRotationMatrix2D(tuple(centre), angle, scale)
            m[:, 2] += a.pos - centre
            warped = cv2.warpAffine(patch, m, (cfg.width, cfg.height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            alpha = np.zeros((ph, pw), np.float32)
            cv2.ellipse(alpha, (int(centre[0]), int(centre[1])), (int(0.8 * (fx2 - fx1)), int(0.95 * (fy2 - fy1))),
                        0, 0, 360, 1.0, -1)       # soft elliptical window around the face
            alpha = cv2.GaussianBlur(alpha, (0, 0), max(2.0, 0.08 * (fx2 - fx1)))
            alpha = cv2.warpAffine(alpha, m, (cfg.width, cfg.height))[..., None]
            frame = frame * (1 - alpha) + np.clip(warped * gain, 0, 255) * alpha
            corners = np.array([[fx1, fy1, 1], [fx2, fy1, 1], [fx2, fy2, 1], [fx1, fy2, 1]]) @ m.T
            x1, y1 = corners.min(0)
            x2, y2 = corners.max(0)
            inside = (min(x2, cfg.width) - max(x1, 0)) * (min(y2, cfg.height) - max(y1, 0))
            if inside >= 0.8 * (x2 - x1) * (y2 - y1) > 0:
                drawn.append((a, alpha[..., 0] > 0.5,
                              [int(max(0, x1)), int(max(0, y1)), int(min(cfg.width, x2)), int(min(cfg.height, y2))]))
        faces = []
        for k, (a, _, box) in enumerate(drawn):   # a face hidden by a person drawn later is not ground truth
            covered = np.zeros((cfg.height, cfg.width), bool)
            for _, later_mask, _ in drawn[k + 1:]:
                covered |= later_mask
            region = covered[box[1]:box[3], box[0]:box[2]]
            if region.size and region.mean() < 0.2:
                faces.append({"id": a.identity.name, "target": a.is_target, "box": box})
        img = np.clip(frame + rng.normal(0, cfg.noise_sigma, frame.shape), 0, 255).astype(np.uint8)
        if rng.random() < cfg.blur_probability:
            k = int(rng.choice([3, 5, 7]))
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            img = cv2.filter2D(img, -1, kernel)
        frames_out.append(img)
        truth_frames.append({"frame": t, "faces": faces})

    out_path = Path(out_path)
    write_video(frames_out, out_path, cfg.fps)
    truth = {"fps": cfg.fps, "size": [cfg.width, cfg.height], "target": target, "others": list(others),
             "seed": cfg.seed, "frames": truth_frames}
    out_path.with_suffix(".truth.json").write_text(json.dumps(truth))
    return truth
