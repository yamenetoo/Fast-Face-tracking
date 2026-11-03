import numpy as np
from fastfacetrack.detectors import Detection, HaarDetector, iou, make_detector, non_max_suppression
import pytest


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    assert sorted(non_max_suppression(boxes, np.array([0.9, 0.8, 0.7]), 0.3)) == [0, 2]


def test_detection_properties():
    d = Detection(2, 3, 12, 23, 0.9)
    assert (d.width, d.height, d.box) == (10, 20, (2, 3, 12, 23))


def test_haar_on_blank_frame_finds_nothing():
    assert HaarDetector().detect(np.full((240, 320, 3), 127, np.uint8)) == []


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_detector("nope")
