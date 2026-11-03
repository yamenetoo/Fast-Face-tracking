"""Build a labelled dataset from extracted faces and train several classifiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .extraction import FaceSet
from .models import MODEL_NAMES, FaceClassifier


def make_dataset(target_sets: list[FaceSet], other_sets: list[FaceSet], block: int = 30):
    """Stack crops (label 1 = target, 0 = other) and return ``(faces, y, groups)``.

    ``groups`` assigns every crop to a block of ``block`` consecutive frames of its video, so that a
    train/validation split never puts neighbouring (near-identical) frames on both sides.
    """
    faces, labels, groups = [], [], []
    for label, sets in ((1, target_sets), (0, other_sets)):
        for s in sets:
            faces.append(s.faces)
            labels.append(np.full(len(s), label))
            groups += [f"{label}:{s.source}:{i // block}" for i in s.frame_index]
    return np.concatenate(faces), np.concatenate(labels), np.array(groups)


def grouped_split(y: np.ndarray, groups: np.ndarray, val_fraction: float = 0.2, seed: int = 0):
    """Hold out whole groups (frame blocks) for validation, separately for each class so that both
    classes are present on both sides. Returns boolean ``(train_mask, val_mask)``."""
    rng = np.random.default_rng(seed)
    val_groups = []
    for label in np.unique(y):
        unique = np.unique(groups[y == label])
        rng.shuffle(unique)
        val_groups += list(unique[:max(1, int(round(len(unique) * val_fraction)))])
    val_mask = np.isin(groups, val_groups)
    return ~val_mask, val_mask


def train_models(faces: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                 names: tuple[str, ...] = MODEL_NAMES, features: str = "hog", seed: int = 0,
                 val_fraction: float = 0.2, epochs: int = 40, batch_size: int = 64, verbose: int = 0):
    """Train the requested models on the training part and report validation metrics.

    Returns ``(classifiers, report)``: a dict name -> fitted :class:`FaceClassifier` and a DataFrame.
    """
    if groups is None:
        groups = np.arange(len(y)).astype(str)
    train, val = grouped_split(y, groups, val_fraction, seed)
    classifiers, rows = {}, []
    for name in names:
        clf = FaceClassifier(name, features=features, seed=seed)
        clf.fit(faces[train], y[train], epochs=epochs, batch_size=batch_size, verbose=verbose)
        pred = clf.predict(faces[val])
        p, r, f, _ = precision_recall_fscore_support(y[val], pred, average="binary", zero_division=0)
        rows.append({"model": name, "kind": clf.kind, "parameters": clf.n_parameters,
                     "train_seconds": round(clf.train_seconds, 2), "n_train": int(train.sum()),
                     "n_val": int(val.sum()), "val_accuracy": accuracy_score(y[val], pred),
                     "val_precision": p, "val_recall": r, "val_f1": f})
        classifiers[name] = clf
    return classifiers, pd.DataFrame(rows)
