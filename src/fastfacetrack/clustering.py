"""Optional clustering of extracted faces (to separate people without labels)."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .features import extract_features


def cluster_faces(faces: np.ndarray, n_clusters: int, features: str = "hog", n_components: int = 30,
                  seed: int = 0) -> tuple[np.ndarray, dict]:
    """K-means on PCA-reduced features. Returns ``(labels, info)`` with the silhouette score."""
    x = extract_features(faces, features)
    x = PCA(n_components=min(n_components, x.shape[0] - 1, x.shape[1]), random_state=seed).fit_transform(x)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x)
    sil = float(silhouette_score(x, km.labels_)) if 1 < n_clusters < len(x) else float("nan")
    return km.labels_, {"silhouette": sil, "inertia": float(km.inertia_)}


def cluster_purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of samples that belong to the majority true class of their cluster."""
    labels, truth = np.asarray(labels), np.asarray(truth)
    hit = 0
    for c in np.unique(labels):
        _, counts = np.unique(truth[labels == c], return_counts=True)
        hit += counts.max()
    return hit / len(labels)


def montage(faces: np.ndarray, labels: np.ndarray, per_cluster: int = 12, scale: int = 2) -> np.ndarray:
    """One row of sample crops per cluster, for a quick visual check (grayscale image)."""
    rows = []
    size = faces.shape[1] * scale
    for c in np.unique(labels):
        members = faces[labels == c][:per_cluster]
        tiles = [cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST) for f in members]
        tiles += [np.zeros((size, size), np.uint8)] * (per_cluster - len(tiles))
        rows.append(np.hstack(tiles))
    return np.vstack(rows)
