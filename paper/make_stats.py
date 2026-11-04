"""Statistical tables for the article: paired tests, calibration study, SPRT prediction.
Run from the repository root:  PYTHONPATH=src python paper/make_stats.py"""
import itertools, json
import numpy as np, pandas as pd
from scipy import stats
from fastfacetrack.moving import class_params

P = "paper/"
d = pd.read_csv("results/benchmark/detection_per_scene.csv")

def paired(a, b, col):
    x = d[d.method == a].set_index("seed")[col]; y = d[d.method == b].set_index("seed")[col]
    z = (x - y).dropna(); n = len(z); m = z.mean(); se = z.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1)
    return m, m - t * se, m + t * se, stats.wilcoxon(z).pvalue
rows = []
for lab, col, f in (("Precision", "precision", "%.3f"), ("Recall", "recall", "%.3f"), ("F$_1$", "f1", "%.3f"), ("ms per frame", "ms_per_frame", "%.2f")):
    m, lo, hi, p = paired("SWIFT-MO (calibrated)", "MOG2", col)
    rows.append(f"{lab} & {f%m} & [{f%lo},\\,{f%hi}] & {p:.3f} \\\\")
open(P + "tab_paired.tex", "w").write("\\begin{tabular}{lccc}\\toprule\nMetric & Mean difference & 95\\% $t$-interval & Wilcoxon $p$ \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# calibration study
c = pd.read_csv("results/calibration/calibration_study.csv")
a, m, test_n = .05, 100, 200
k = int(np.ceil((m + 1) * (1 - a))); beta = stats.beta(m + 1 - k, k)
sd_pred = np.sqrt(beta.var() + beta.mean() * (1 - beta.mean()) / test_n)
names = {"A": "A: 0 frames", "B": "B: 100 frames", "C": "C: 300 frames"}
rows = []
for mode, g in c.groupby("mode"):
    e = g.exceed; n = len(e); se = e.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
    rows.append(f"{names[mode]} & {g.threshold.mean():.2f} & {g.test_median.mean():.2f} & {g.test_max.mean():.2f} & "
                f"{e.mean():.3f} [{e.mean()-t*se:.3f},\\,{e.mean()+t*se:.3f}] & {e.std(ddof=1):.3f} & {e.max():.3f} \\\\")
rows.append(f"Prediction under exchangeability & -- & -- & -- & {beta.mean():.3f} & {sd_pred:.3f} & -- \\\\")
open(P + "tab_cal.tex", "w").write("\\begin{tabular}{lcccccc}\\toprule\nBurn-in before calibration & Threshold $c$ & Median $M$ (test) & Max $M$ (test) & Exceedance, mean [95\\% CI] & sd & max \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print(open(P + "tab_cal.tex").read())

# SPRT prediction
cls = pd.read_csv("results/benchmark/classification_per_track.csv")
print(cls.columns.tolist())
sd = 22.0; B = np.log(.99 / .01); neff = 20
out = []
for sep in (1, .25, .1):
    Q = class_params(sep)
    kl = [0.5 * np.sum((np.array(Q[u][0]) - np.array(Q[v][0])) ** 2) / sd ** 2 for u, v in itertools.combinations(Q, 2)]
    g = cls[np.isclose(cls.class_sep, sep)]
    obs = g.sprt_frames.dropna().mean() if "sprt_frames" in g else float("nan")
    out.append((sep, min(kl), np.mean(kl), neff * min(kl), B / (neff * min(kl)), obs))
rows = [f"{s:g} & {kmin:.3f} & {kmean:.3f} & {pf:.1f} & {max(1,asn):.2f} & {obs:.2f} \\\\" for s, kmin, kmean, pf, asn, obs in out]
open(P + "tab_asn.tex", "w").write("\\begin{tabular}{lccccc}\\toprule\nSeparation & KL (closest pair) & KL (mean pair) & Drift per frame & Predicted $\\mathrm{E}[N]$ & Observed \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print(open(P + "tab_asn.tex").read(), open(P + "tab_paired.tex").read())
