"""Calibration and validation-chosen thresholds from saved predictions (docs/secondary_analyses_protocol.md, A-B).

Usage:
    python -X utf8 -m src.analyze_calibration_thresholds

No model is run. For every completed PTB-XL run with saved predictions: Brier score and ECE (10 equal-width
bins) per label on validation and test; per-label thresholds chosen on validation only (0.01-0.99, maximum
F1, ties closest to 0.5) and applied unchanged to test. 0.5 stays the primary rule.
Writes results/secondary/calibration_thresholds/.
"""
from __future__ import annotations

import csv

import numpy as np

from src.common import ROOT, read_json, write_csv, write_json

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
OUT = ROOT / "results/secondary/calibration_thresholds"
FAMILIES = ("ptbxl_v2", "ptbxl_v2_fixedgain", "ptbxl_signal", "ptbxl_cnn_lstm", "ptbxl_spatiotemporal", "ptbxl_hybrid_vit")
GRID = np.round(np.arange(1, 100) / 100, 2)
FIGURE_RUNS = {  # seed-42 reliability diagrams, in the categorical palette's fixed order
    "PCA + LDA": "ptbxl_v2/ptbxl_v2_lda_wide_seed42", "CNN-only": "ptbxl_v2/ptbxl_v2_cnn_pool_seed42",
    "CNN-LSTM": "ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42",
    "Temporal conv": "ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42",
    "Hybrid ViT": "ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42", "Signal ResNet (ref.)": "ptbxl_signal/ptbxl_v2_signal_resnet1d_seed42"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def predictions(directory, split):
    with (directory / f"{split}_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([[int(r[f"truth_{n}"]) for n in CLASS_NAMES] for r in rows])
    p = np.asarray([[float(r[f"probability_{n}"]) for n in CLASS_NAMES] for r in rows])
    return y, p


def ece(y, p, bins=10):
    """Expected calibration error of one binary label: count-weighted |mean p - prevalence| over equal-width bins."""
    index = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(np.sum(index == b) / len(p) * abs(p[index == b].mean() - y[index == b].mean())
                     for b in range(bins) if np.any(index == b)))


def f1_sens_spec(y, decision):
    tp, fp = np.sum((decision == 1) & (y == 1)), np.sum((decision == 1) & (y == 0))
    fn, tn = np.sum((decision == 0) & (y == 1)), np.sum((decision == 0) & (y == 0))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return float(f1), float(tp / (tp + fn)) if tp + fn else float("nan"), float(tn / (tn + fp)) if tn + fp else float("nan")


def choose_threshold(y, p):
    """Validation threshold maximizing F1 over GRID; ties go to the value closest to 0.5."""
    scores = [f1_sens_spec(y, (p >= t).astype(int))[0] for t in GRID]
    best = max(scores)
    return float(min((t for t, s in zip(GRID, scores) if s == best), key=lambda t: (abs(t - 0.5), t)))


def runs():
    for family in FAMILIES:
        for run_file in sorted((ROOT / "results" / family).glob("*/run.json")):
            if read_json(run_file).get("status") == "completed" and (run_file.parent / "test_predictions.csv").exists():
                yield run_file.parent


def plot(reliability):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.0), sharey=True, constrained_layout=True)
    for j, (axis, label) in enumerate(zip(axes, CLASS_NAMES)):
        axis.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, linestyle="--", zorder=0)
        for (name, curves), color, marker in zip(reliability.items(), COLORS, MARKERS):
            x, yv = curves[j]
            axis.plot(x, yv, color=color, linewidth=2, marker=marker, markersize=4, label=name)
        axis.set_title(label, color="#0b0b0b")
        axis.set_xlabel("Mean predicted probability")
        axis.set_xlim(0, 1), axis.set_ylim(0, 1)
        axis.grid(alpha=.15)
    axes[0].set_ylabel("Observed frequency")
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Reliability on the test fold, seed 42 (10 equal-width bins; dashed = perfect calibration)", color="#0b0b0b")
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"reliability_seed42_test.{suffix}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration, thresholds, summary = [], [], []
    for directory in runs():
        run_id, family = directory.name, directory.parent.name
        (yv, pv), (yt, pt) = predictions(directory, "validation"), predictions(directory, "test")
        row = {"family": family, "run_id": run_id}
        for split, y, p in (("validation", yv, pv), ("test", yt, pt)):
            for j, label in enumerate(CLASS_NAMES):
                calibration.append({"family": family, "run_id": run_id, "split": split, "label": label,
                                    "brier": float(np.mean((p[:, j] - y[:, j]) ** 2)), "ece_10bin": ece(y[:, j], p[:, j]),
                                    "mean_probability": float(p[:, j].mean()), "prevalence": float(y[:, j].mean())})
            mine = [c for c in calibration if c["run_id"] == run_id and c["split"] == split]
            row[f"{split}_macro_brier"] = float(np.mean([c["brier"] for c in mine]))
            row[f"{split}_macro_ece"] = float(np.mean([c["ece_10bin"] for c in mine]))
        f1_fixed, f1_chosen = [], []
        for j, label in enumerate(CLASS_NAMES):
            t = choose_threshold(yv[:, j], pv[:, j])
            a, b = f1_sens_spec(yt[:, j], (pt[:, j] >= 0.5).astype(int)), f1_sens_spec(yt[:, j], (pt[:, j] >= t).astype(int))
            f1_fixed.append(a[0]), f1_chosen.append(b[0])
            thresholds.append({"family": family, "run_id": run_id, "label": label, "validation_threshold": t,
                               "test_f1_at_0.5": a[0], "test_sensitivity_at_0.5": a[1], "test_specificity_at_0.5": a[2],
                               "test_f1_at_validation_threshold": b[0], "test_sensitivity_at_validation_threshold": b[1],
                               "test_specificity_at_validation_threshold": b[2]})
        row.update(test_macro_f1_at_0_5=float(np.mean(f1_fixed)), test_macro_f1_at_validation_thresholds=float(np.mean(f1_chosen)))
        summary.append(row)
        print(f"{run_id}: test ECE {row['test_macro_ece']:.4f}, macro F1 0.5 -> chosen {row['test_macro_f1_at_0_5']:.4f} -> "
              f"{row['test_macro_f1_at_validation_thresholds']:.4f}", flush=True)
    write_csv(OUT / "calibration_per_label.csv", calibration)
    write_csv(OUT / "thresholds_per_label.csv", thresholds)
    write_csv(OUT / "summary.csv", summary)
    reliability = {}
    for name, relative in FIGURE_RUNS.items():
        y, p = predictions(ROOT / "results" / relative, "test")
        curves = []
        for j in range(len(CLASS_NAMES)):
            index = np.minimum((p[:, j] * 10).astype(int), 9)
            filled = [b for b in range(10) if np.sum(index == b) >= 10]  # bins with at least 10 ECGs
            curves.append(([float(p[index == b, j].mean()) for b in filled], [float(y[index == b, j].mean()) for b in filled]))
        reliability[name] = curves
    write_json(OUT / "reliability_seed42_test.json", {"bins": 10, "min_ecgs_per_plotted_bin": 10, "curves": reliability})
    plot(reliability)
    lines = ["# Calibration and validation-chosen thresholds", "",
             "Generated by `python -m src.analyze_calibration_thresholds` from saved predictions (protocol: "
             "docs/secondary_analyses_protocol.md). 0.5 remains the primary threshold; test results are exploratory.", "",
             "| Run | Val macro ECE | Test macro ECE | Test macro Brier | Test macro F1 at 0.5 | Test macro F1 at validation thresholds |",
             "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['run_id']} | {r['validation_macro_ece']:.4f} | {r['test_macro_ece']:.4f} | {r['test_macro_brier']:.4f} | "
              f"{r['test_macro_f1_at_0_5']:.4f} | {r['test_macro_f1_at_validation_thresholds']:.4f} |" for r in summary]
    lines += ["", "![Reliability, seed 42, test](reliability_seed42_test.png)", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
