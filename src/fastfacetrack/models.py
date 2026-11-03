"""Face classifiers: three classical models (logistic regression, random forest, decision tree) and two
neural networks (ANN, CNN). All expose the same :class:`FaceClassifier` interface and output the
probability that a crop shows the **target** person."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import FACE_SIZE, extract_features

CLASSICAL = ("logreg", "rf", "tree")
DEEP = ("ann", "cnn")
MODEL_NAMES = CLASSICAL + DEEP


def _classical_estimator(name: str, seed: int):
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=seed)),
                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed)
    if name == "tree":
        return DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=seed)
    raise ValueError(name)


def _keras_model(name: str, size: int, seed: int):
    import tensorflow as tf  # imported lazily: the classical models work without TensorFlow
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.keras.utils.set_random_seed(seed)
    if name == "ann":
        model = keras.Sequential([
            layers.Input(shape=(size, size)), layers.Flatten(), layers.Rescaling(1 / 255.0),
            layers.Dense(128, activation="relu"), layers.Dropout(0.3),
            layers.Dense(64, activation="relu"), layers.Dense(1, activation="sigmoid")], name="ann")
    elif name == "cnn":
        model = keras.Sequential([
            layers.Input(shape=(size, size, 1)), layers.Rescaling(1 / 255.0),
            layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"), layers.GlobalAveragePooling2D(),
            layers.Dense(32, activation="relu"), layers.Dropout(0.3), layers.Dense(1, activation="sigmoid")],
            name="cnn")
    else:
        raise ValueError(name)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


class FaceClassifier:
    """Binary classifier ``target`` (1) vs ``other`` (0) on grayscale face crops."""

    def __init__(self, name: str, features: str = "hog", seed: int = 0, size: int = FACE_SIZE):
        if name not in MODEL_NAMES:
            raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
        self.name, self.features, self.seed, self.size = name, features, seed, size
        self.kind = "classical" if name in CLASSICAL else "deep"
        self.model = _classical_estimator(name, seed) if self.kind == "classical" else None
        self.history: dict | None = None
        self.train_seconds: float = float("nan")

    # -- training ----------------------------------------------------------------------------
    def fit(self, faces: np.ndarray, y: np.ndarray, epochs: int = 40, batch_size: int = 64,
            validation_split: float = 0.15, verbose: int = 0) -> "FaceClassifier":
        start = time.perf_counter()
        y = np.asarray(y).astype(int)
        if self.kind == "classical":
            self.model.fit(extract_features(faces, self.features), y)
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1          # threads only slow down one-frame-at-a-time inference
        else:
            from tensorflow import keras
            self.model = _keras_model(self.name, self.size, self.seed)
            x = faces[..., None].astype("float32") if self.name == "cnn" else faces.astype("float32")
            counts = np.bincount(y, minlength=2)
            weights = {c: len(y) / (2 * max(1, counts[c])) for c in (0, 1)}
            stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
            hist = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                                  class_weight=weights, callbacks=[stop], verbose=verbose)
            self.history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        self.train_seconds = time.perf_counter() - start
        return self

    # -- inference ---------------------------------------------------------------------------
    def predict_proba(self, faces: np.ndarray) -> np.ndarray:
        """Probability of the target class for each crop, shape ``(n,)``."""
        if len(faces) == 0:
            return np.zeros(0)
        if self.kind == "classical":
            return self.model.predict_proba(extract_features(faces, self.features))[:, 1]
        x = faces[..., None].astype("float32") if self.name == "cnn" else faces.astype("float32")
        if len(x) <= 256:                       # direct call avoids the per-call overhead of Model.predict
            return np.asarray(self.model(x, training=False)).ravel()
        return self.model.predict(x, verbose=0).ravel()

    def predict(self, faces: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(faces) >= threshold).astype(int)

    @property
    def n_parameters(self) -> int | None:
        return int(self.model.count_params()) if self.kind == "deep" and self.model is not None else None

    # -- persistence -------------------------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"name": self.name, "features": self.features, "seed": self.seed, "size": self.size}
        (folder / f"{self.name}.json").write_text(json.dumps(meta))
        if self.kind == "classical":
            joblib.dump(self.model, folder / f"{self.name}.joblib")
        else:
            self.model.save(folder / f"{self.name}.keras")
        return folder

    @classmethod
    def load(cls, folder: str | Path, name: str) -> "FaceClassifier":
        folder = Path(folder)
        meta = json.loads((folder / f"{name}.json").read_text())
        clf = cls(meta["name"], meta["features"], meta["seed"], meta["size"])
        if clf.kind == "classical":
            clf.model = joblib.load(folder / f"{name}.joblib")
        else:
            from tensorflow import keras
            clf.model = keras.models.load_model(folder / f"{name}.keras")
        return clf
