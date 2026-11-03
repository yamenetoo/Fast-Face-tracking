import numpy as np
import pytest
from fastfacetrack.models import FaceClassifier
from fastfacetrack.training import grouped_split, make_dataset, train_models
from fastfacetrack.extraction import FaceSet


@pytest.mark.parametrize("name", ["logreg", "rf", "tree"])
def test_classical_models_learn_toy_task(toy_faces, name):
    x, y = toy_faces
    clf = FaceClassifier(name, seed=0).fit(x, y)
    assert (clf.predict(x) == y).mean() > 0.95
    p = clf.predict_proba(x[:5])
    assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()


def test_empty_batch_returns_empty(toy_faces):
    x, y = toy_faces
    assert FaceClassifier("tree").fit(x, y).predict_proba(x[:0]).shape == (0,)


def test_save_and_load_roundtrip(toy_faces, tmp_path):
    x, y = toy_faces
    clf = FaceClassifier("logreg").fit(x, y)
    clf.save(tmp_path)
    again = FaceClassifier.load(tmp_path, "logreg")
    assert np.allclose(clf.predict_proba(x[:10]), again.predict_proba(x[:10]))


def test_unknown_model_name():
    with pytest.raises(ValueError):
        FaceClassifier("svm")


def test_grouped_split_keeps_both_classes_and_blocks_together():
    y = np.array([1] * 60 + [0] * 60)
    groups = np.array([f"{l}:{i // 10}" for l, i in zip(y, list(range(60)) * 2)])
    train, val = grouped_split(y, groups, 0.2, seed=1)
    assert set(y[val]) == {0, 1} and set(y[train]) == {0, 1}
    assert not set(groups[train]) & set(groups[val])


def test_train_models_report(toy_faces):
    x, y = toy_faces
    sets_t = FaceSet(x[y == 1], np.arange(80), np.zeros((80, 4), int), np.ones(80), "t")
    sets_o = FaceSet(x[y == 0], np.arange(80), np.zeros((80, 4), int), np.ones(80), "o")
    faces, labels, groups = make_dataset([sets_t], [sets_o], block=20)
    _, report = train_models(faces, labels, groups, ("logreg", "tree"))
    assert list(report["model"]) == ["logreg", "tree"] and (report["val_accuracy"] > 0.9).all()


def test_cnn_and_ann_train_and_predict(toy_faces):
    pytest.importorskip("tensorflow")
    x, y = toy_faces
    for name in ("ann", "cnn"):
        clf = FaceClassifier(name).fit(x, y, epochs=3)
        assert clf.predict_proba(x[:4]).shape == (4,) and clf.n_parameters > 0
