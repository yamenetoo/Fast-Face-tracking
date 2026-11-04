"""Build numbers.tex and the LaTeX tables of the paper from the CSV files in ../results."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent


def ms(x, s, d=3):
    return f"{x:.{d}f}$\\,\\pm\\,${s:.{d}f}"


def detection_table(csv_dir, label):
    d = pd.read_csv(R / csv_dir / "detection_per_scene.csv")
    g = d.groupby("method", sort=False)
    rows = []
    for name, x in g:
        rows.append(f"{name} & {ms(x.precision.mean(), x.precision.std())} & {ms(x.recall.mean(), x.recall.std())} & "
                    f"{ms(x.f1.mean(), x.f1.std())} & {x.false_alarm_rate.mean():.3f} & {ms(x.ms_per_frame.mean(), x.ms_per_frame.std(), 1)} \\\\")
    head = ("\\begin{tabular}{lccccc}\\toprule\nMethod & Precision & Recall & F$_1$ & False-alarm rate & ms/frame \\\\\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n", d


def class_table(csv_dir):
    c = pd.read_csv(R / csv_dir / "classification_per_track.csv")
    rows = []
    for sep, x in c.groupby("class_sep"):
        dec = x.sprt_correct.notna()
        rows.append(f"{sep:g} & {len(x)} & {x.acc_1.astype(float).mean():.3f} & {x.acc_5.dropna().astype(float).mean():.3f} & "
                    f"{dec.mean():.3f} & {x.sprt_correct.dropna().astype(float).mean():.3f} & {x.sprt_frames.mean():.2f} \\\\")
    head = ("\\begin{tabular}{rrccccc}\\toprule\nClass separation & Tracks & Acc.\\ after 1 frame & Acc.\\ after 5 frames & "
            "SPRT decided & SPRT accuracy & SPRT frames \\\\\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n", c


tab, d = detection_table("benchmark", "v2")
(OUT / "tab_detection.tex").write_text(tab)
ctab, c = class_table("benchmark")
(OUT / "tab_class.tex").write_text(ctab)
if (R / "benchmark_v1").exists():
    t1, d1 = detection_table("benchmark_v1", "v1"); (OUT / "tab_detection_v1.tex").write_text(t1)
    ct1, _ = class_table("benchmark_v1"); (OUT / "tab_class_v1.tex").write_text(ct1)

def get(df, m, col): return df[df.method == m][col]
full = "SWIFT-MO (calibrated)"
s = json.loads((R / "moving" / "summary.json").read_text())
N = {
    "NwinPerFrame": f"{s['windows_per_frame']:,}".replace(",", "{,}"),
    "msSwift": f"{get(d, full, 'ms_per_frame').mean():.1f}",
    "fSwift": f"{get(d, full, 'f1').mean():.2f}",
    "fSwiftSd": f"{get(d, full, 'f1').std():.2f}",
    "fMog": f"{get(d, 'MOG2', 'f1').mean():.2f}",
    "msMog": f"{get(d, 'MOG2', 'ms_per_frame').mean():.1f}",
    "msKnn": f"{get(d, 'KNN', 'ms_per_frame').mean():.1f}",
    "faNoComp": f"{get(d, 'no illumination compensation', 'false_alarm_rate').mean():.2f}",
    "nSeeds": str(d.seed.nunique()),
    "seedLow": str(int(d.seed.min())), "seedHigh": str(int(d.seed.max())),
}
(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
print((OUT / "numbers.tex").read_text()); print(tab); print(ctab)
