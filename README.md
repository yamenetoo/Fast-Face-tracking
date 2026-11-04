<p align="center">
  <img src="assets/logo.svg" width="140" alt="fastfacetrack logo">
</p>

# fastfacetrack

Fast face detection, extraction, target-face classification and evaluation on video, with a built-in simulator and a one-click Colab notebook.

The package turns the original *Fast-Face-tracking* pipeline into an installable Python library with a command line interface:

1. **Face extraction.** Detect faces in a video with the 1 MB **UltraLight** detector (OpenCV Haar cascade as a fallback) and save the crops.
2. **Clustering (optional).** Group the extracted faces with k-means to separate people without labelling.
3. **Machine-learning models.** Logistic regression, random forest and decision tree on HOG features.
4. **Deep-learning models.** An ANN and a CNN (TensorFlow/Keras) on the raw crops.
5. **Test on video.** Detect on every frame, classify each face, and record the detection time, the classification time and whether the target is present.
6. **Combine the results** of detectors, models and videos into one table / CSV.
7. **Simulator.** Synthetic videos with exact ground truth, so everything above can be run and measured without private footage.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamenetoo/Fast-Face-tracking/blob/main/notebooks/colab_simulation.ipynb)

---

## Contents

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Command line](#3-command-line)
4. [Python API](#4-python-api)
5. [How it works](#5-how-it-works)
6. [Simulation results](#6-simulation-results)
7. [Limitations](#7-limitations)
8. [Repository layout](#8-repository-layout)
9. [Tests](#9-tests)

---

## 1. Install

```bash
pip install git+https://github.com/yamenetoo/Fast-Face-tracking.git          # classical models + detectors
pip install "fastfacetrack[deep] @ git+https://github.com/yamenetoo/Fast-Face-tracking.git"   # + ANN and CNN
```

or from a clone: `pip install -e ".[dev]"`. Requires Python 3.9+. Dependencies: numpy, pandas, scikit-learn, opencv-python-headless, joblib, matplotlib (and TensorFlow for the ANN and CNN; the three classical models work without it).

The UltraLight ONNX model (about 1.2 MB) is downloaded from the [original detector repository](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) on first use and cached in `~/.cache/fastfacetrack`.

## 2. Quick start

**Simulation in Colab** (no data needed): open the badge above and choose Runtime → Run all (about 10 minutes on CPU).

**Simulation from the terminal**

```bash
fastfacetrack experiment --out results          # about 6 minutes on 2 CPU cores
```

**Your own videos**

```python
import fastfacetrack as fft

det    = fft.make_detector("ultralight")                 # or "ultralight-slim", "haar"
target = fft.extract_faces("target_person.mp4", det)     # video that shows only the person to find
others = fft.extract_faces("other_people.mp4", det)      # everybody else
X, y, groups = fft.make_dataset([target], [others])
models, report = fft.train_models(X, y, groups)          # logreg, rf, tree, ann, cnn
print(report)

summary, per_frame = fft.evaluate_video("test.mp4", det, models)
print(summary[["model", "detect_ms", "classify_ms", "fps", "frames_target_predicted"]])
```

If you have a single video with several people and no labels, cluster first and pick the target's cluster:

```bash
fastfacetrack extract meeting.mp4 --out faces.npz
fastfacetrack cluster faces.npz -k 4 --out clusters.png    # one row of sample faces per cluster
```

## 3. Command line

| Command | Purpose |
|---|---|
| `fastfacetrack extract VIDEO --out faces.npz [--detector ultralight\|ultralight-slim\|haar] [--stride N] [--images DIR]` | detect and save face crops |
| `fastfacetrack cluster faces.npz -k K --out clusters.png` | k-means on the crops, montage with one row per cluster |
| `fastfacetrack train --target T.npz ... --other O.npz ... --out models/` | train the five classifiers (`--models` selects a subset) |
| `fastfacetrack test VIDEO --models-dir models/ [--truth FILE.truth.json] --detectors ultralight haar --out results.csv` | detect + classify every frame, write timings and presence |
| `fastfacetrack simulate --out sim.mp4` | one synthetic video plus `sim.truth.json` |
| `fastfacetrack experiment --out results/` | the complete simulation with tables and figures |

## 4. Python API

| Module | Contents |
|---|---|
| `detectors` | `UltraLightDetector`, `HaarDetector`, `make_detector`, `Detection`, `iou`, `non_max_suppression` |
| `video` | `iter_frames`, `video_info`, `write_video` |
| `extraction` | `extract_faces`, `FaceSet` (save/load `.npz`, save PNGs) |
| `features` | `crop_face` (square crop, grayscale, histogram equalisation, 40 × 40), `hog_features`, `pixel_features` |
| `clustering` | `cluster_faces`, `cluster_purity`, `montage` |
| `models` | `FaceClassifier` (`logreg`, `rf`, `tree`, `ann`, `cnn`; `fit`, `predict_proba`, `save`, `load`) |
| `training` | `make_dataset`, `grouped_split`, `train_models` |
| `evaluation` | `run_detector`, `score_classifier`, `evaluate_video`, `combine_results`, `load_truth` |
| `simulation` | `IdentityBank`, `SimulationConfig`, `simulate_video` |
| `report` | `plot_model_comparison`, `plot_timeline`, `annotate_frames` |
| `pipeline` | `ExperimentConfig`, `run_experiment` |

### Recorded results

`evaluate_video` returns a summary table (one row per detector × model) and a per-frame table.

| Column | Meaning |
|---|---|
| `detect_ms`, `classify_ms`, `total_ms`, `fps` | mean time per frame spent in the detector, in the classifier, and the resulting throughput |
| `frames_target_predicted` | frames in which the target is predicted present |
| `det_precision`, `det_recall`, `det_f1` | detection-level scores for the target (needs ground truth) |
| `frame_accuracy` | share of frames with the right "target present / absent" answer |
| `face_detection_recall` | share of ground-truth faces the detector found at all |
| `first_hit_delay_frames` | frames between the target first appearing and the first correct detection |

## 5. How it works

```
video ──▶ detector (UltraLight / Haar) ──▶ boxes ──▶ crop (40×40, gray, equalised)
                                                         │
                       ┌─────────────────────────────────┤
                       ▼                                 ▼
          HOG ─▶ logreg / rf / tree            raw pixels ─▶ ANN / CNN
                       └──────────────┬──────────────────┘
                                      ▼
                    P(target) per face ─▶ presence per frame, timings, CSV
```

* **Training labels.** Crops from video of the target are labelled 1, crops of everybody else 0. Class weights balance the two.
* **Validation split.** Whole blocks of 30 consecutive frames are held out per class, so near-identical neighbouring frames never sit on both sides of the split.
* **Evaluation.** The detector runs once per frame and every classifier scores the same crops, so detection and classification times are reported separately. With ground truth, each detection is matched to a true face at IoU ≥ 0.4.

### The simulator

`simulate_video` renders a normal MP4 and a JSON file with the exact box and identity of every face in every frame. People are real photo faces pasted onto a drifting noisy background, with motion (bouncing), scale, rotation, lighting, motion blur, sensor noise, people entering and leaving, and overlaps. A face covered by more than 20 % by another person is not counted as ground truth. The demo identities come from three public group photos of the UltraLight repository (downloaded at run time, not stored here); use `IdentityBank.from_images` for your own photos.

## 6. Simulation results

Settings (`results/config.json`): 9 usable identities from the demo photos; 1 target, 5 people seen in training, 2 people seen only at test time per video; training clips of 120 frames per person; 3 test videos of 400 frames (640 × 480, 15 fps) with the target on screen in about 25 % of the frames; threshold 0.5; seed 0. Timings were measured on 2 CPU cores.

**Clustering.** K-means on HOG features separates the 6 training people perfectly (purity 1.00, silhouette 0.42; `results/figures/cluster_montage.png`).

**Training (validation = held-out frame blocks).**

| Model | Parameters | Train seconds | Validation accuracy | Validation F1 |
|---|---:|---:|---:|---:|
| logreg | n/a | 0.1 | 1.000 | 1.000 |
| rf | n/a | 0.9 | 1.000 | 1.000 |
| tree | n/a | 0.1 | 0.993 | 0.984 |
| ann | 213,249 | 10.0 | 1.000 | 1.000 |
| cnn | 25,409 | 15.5 | 0.987 | 0.968 |

**Test videos, UltraLight RFB-320 detector (mean over 3 videos).**

| Model | Precision | Recall | F1 | Frame accuracy | Detect ms | Classify ms | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | 0.832 | 1.000 | 0.907 ± 0.049 | 0.960 | 8.8 | 0.6 | 107 |
| rf | 0.886 | 0.990 | **0.934 ± 0.059** | 0.975 | 8.8 | 13.8 | 44 |
| tree | 0.497 | 0.913 | 0.637 ± 0.088 | 0.830 | 8.8 | 0.2 | 111 |
| ann | 0.834 | 0.925 | 0.871 ± 0.055 | 0.945 | 8.8 | 4.5 | 75 |
| cnn | 0.452 | 0.884 | 0.572 ± 0.234 | 0.757 | 8.8 | 10.8 | 51 |

**Detectors** (same classifiers, same videos):

| Detector | Face detection recall | Detection time per frame |
|---|---:|---:|
| UltraLight RFB-320 | 0.999 | 8.8 ms |
| UltraLight slim-320 | 1.000 | 8.3 ms |
| Haar cascade | 0.923 | 58.0 ms |

What the run shows:

* UltraLight is about 6.5 times faster than the Haar cascade on 640 × 480 frames and misses fewer faces; the slim model is slightly faster than RFB and no less accurate here.
* HOG with logistic regression is the best accuracy-per-millisecond choice (F1 0.91 at 0.6 ms); the random forest has the highest F1 (0.93) but its classification time exceeds the detector's.
* The decision tree and the CNN are weak on unseen footage. Both reach a high validation score and then produce many false alarms on people who never appeared in training (precision 0.50 and 0.45), and the CNN's F1 varies a lot between videos (± 0.23).
* On the first test video the target is found in the first frame it appears for logreg, rf, ann and cnn, and after 12 frames for the tree.

Tables and figures are in [`results/`](results): `tables/test_aggregate.csv` (mean and sd per detector and model), `tables/test_summary.csv` (every video), `tables/training_report.csv`, and `figures/` (model comparison, probability timeline, annotated frames, cluster montage).

## SWIFT-MO: fast statistical detection, segmentation and classification of moving objects

`fastfacetrack.swift` adds a second engine that needs no trained detector. For each frame:

1. an adaptive per-pixel Gaussian background gives the squared Mahalanobis distance of every pixel;
2. integral images turn **every sub-window** of **every size** into an O(1) scan statistic `Z(W)`; the same integral images give the Gaussian log-likelihood of any window under a class model from its sufficient statistics (`window_loglik`);
3. the alarm threshold is an order statistic of the maximum of `Z` over object-free frames, a finite-sample distribution-free bound on the false-alarm rate per frame under frame exchangeability (`conformal_threshold`);
4. detected windows are tracked, and the class of each track is decided by Wald's sequential probability ratio test on the accumulated class log-likelihoods.

```python
res = fft.run_moving_experiment("results/moving")      # about 1 minute on CPU, 320 x 240 frames
```

First result (one synthetic scene, 240 frames, three objects of three colour classes, seed 0; `results/moving/summary.json`):

| Quantity | Value |
|---|---:|
| Windows scored per frame | 84,550 |
| Time per frame: background / scan / classification | 1.0 / 2.4 / 1.7 ms (5.1 ms total, 2 CPU cores) |
| Detection precision / recall / F1 (IoU ≥ 0.5) | 0.90 / 0.84 / 0.87 |
| False-alarm rate on object-free frames (target ≤ 0.05; 5 scenes, 500 frames) | 0.000 |
| Tracks classified correctly by the SPRT | 7 of 7 (decided after 1 frame) |

How to read it: this is a proof of concept on a synthetic scene whose objects have strongly different colours and whose noise is independent between pixels, which is what the model assumes. A 0.000 false-alarm rate with target 0.05 means the threshold is conservative here; it is not yet checked on real video, with camera motion, shadows or correlated noise. The SPRT decides after one frame because the classes are far apart; its error guarantee assumes independent frames. Seven tracks for three objects show track fragmentation that the simple association step does not repair. The full write-up of the method, claims and experiment plan is in the contribution document.

## 7. Limitations

* **One photo per person.** Every simulated person is a single photo animated by the simulator. Training and test videos therefore show the same photo under different motion, lighting, blur and background. The results measure re-identification under those variations, **not** generalisation to new photos of the same person, and the near-perfect validation scores are optimistic for the same reason.
* **Simulated, not recorded.** Real video adds expressions, 3-D head rotation, occlusion by objects and compression artefacts that the simulator only approximates.
* **Small classifiers.** The models separate a few known people. They are not biometric-grade identity recognition and should not be used to identify people without their consent.
* **Timings** depend on the machine (2 CPU cores here, Colab will differ) and include Python overhead.
* **Label noise.** If a "target" video contains other faces or false detections, they are labelled as the target. Use the clustering step or clean clips.
* The demo photos and the UltraLight model belong to the original detector repository; check its licence before redistributing them.

## 8. Repository layout

```
Fast-Face-tracking/
├── src/fastfacetrack/        the package (detectors, video, extraction, features, clustering,
│                             models, training, evaluation, simulation, report, pipeline, cli)
├── tests/                    pytest suite (23 tests)
├── notebooks/colab_simulation.ipynb   the Colab notebook: install, run, show results (not pre-executed)
├── examples/run_simulation.py
├── results/                  tables, figures and settings of the simulation reported above
├── pyproject.toml  LICENSE  README.md
```

## 9. Tests

```bash
pip install -e ".[dev]"
pytest            # 23 tests, about 45 seconds
```

The tests cover box geometry and NMS, the five classifiers on a toy task, model save/load, the grouped split, clustering, the simulator (valid boxes, determinism, video size) and the evaluation columns.

## Author

Mohamad Yamen Al-Mohamad, Department of Statistics, University of Tabriz. MIT licence.
