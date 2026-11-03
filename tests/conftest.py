import numpy as np
import pytest


@pytest.fixture(scope="session")
def toy_faces():
    """Two synthetic 'people': smooth blobs with different structure, 40x40 grayscale."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:40, 0:40]
    def person(cx, cy, n):
        out = []
        for _ in range(n):
            img = 255 * np.exp(-(((xx - cx) / 9) ** 2 + ((yy - cy) / 14) ** 2)) + rng.normal(0, 12, (40, 40))
            out.append(np.clip(img, 0, 255).astype(np.uint8))
        return np.stack(out)
    a, b = person(14, 20, 80), person(27, 20, 80)
    return np.concatenate([a, b]), np.array([1] * 80 + [0] * 80)
