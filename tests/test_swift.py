import numpy as np
import pytest

from fastfacetrack.moving import MovingConfig, run_moving_experiment
from fastfacetrack.swift import (IntegralStats, conformal_threshold, gaussian_loglik_from_stats, make_sizes, scan,
                                 suppress_overlaps, window_loglik)


def test_window_loglik_matches_direct():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(60, 80, 3))
    mu = rng.normal(size=3)
    A = rng.normal(size=(3, 3)); cov = A @ A.T + np.eye(3)
    box = (15, 10, 52, 41)
    fast = window_loglik(IntegralStats(img), box, mu, cov)
    x = img[10:41, 15:52].reshape(-1, 3) - mu
    direct = -0.5 * np.sum(3 * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + np.einsum("ni,ij,nj->n", x, np.linalg.inv(cov), x))
    assert fast == pytest.approx(direct, rel=1e-10)


def test_conformal_threshold_order_statistic():
    m = np.arange(1, 20)                       # 19 values, alpha = 0.05 -> the maximum
    assert conformal_threshold(m, 0.05) == 19
    with pytest.raises(ValueError):
        conformal_threshold(m[:5], 0.05)


def test_scan_finds_bright_block():
    d = np.random.default_rng(1).chisquare(3, (120, 160))
    d[40:80, 60:110] += 20
    (z, box), _ = scan(d, make_sizes(d.shape), stride=2)
    x0, y0, x1, y1 = box
    assert abs(x0 - 60) <= 6 and abs(y0 - 40) <= 6 and abs(x1 - 110) <= 8 and abs(y1 - 80) <= 8


def test_suppress_overlaps_keeps_best_and_separate():
    kept = suppress_overlaps([(9, (0, 0, 40, 40)), (8, (5, 5, 20, 20)), (7, (0, 0, 90, 90)), (6, (100, 100, 140, 140))])
    assert [b for _, b in kept] == [(0, 0, 40, 40), (100, 100, 140, 140)]


def test_moving_experiment_runs_and_detects(tmp_path):
    cfg = MovingConfig(n_frames=80)
    res = run_moving_experiment(tmp_path, cfg, n_calibration=40, n_empty_clips=2, log=lambda *a: None)
    s = res["summary"]
    assert s["recall"] > 0.5 and s["precision"] > 0.5
    assert s["empty_false_alarm_rate"] <= 0.1
