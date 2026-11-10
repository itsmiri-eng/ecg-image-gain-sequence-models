"""Publication figures for the IEEE manuscript, from saved results only (no model is run).

Usage:
    python -X utf8 -m src.make_paper_figures

Writes PDF (vector) and PNG (300 dpi) to results/publication/figures_20261110/, plus
results/fixed_gain/retrained_comparison/hierarchical_test.csv: the PTB-XL fixed − adaptive comparison with the
same hierarchical seed × patient bootstrap as the external H2 (seeds 42-46 primary, 43-46 GPU-only sensitivity).
Colours: the validated reference categorical palette (dataviz skill), slots in fixed order.
"""
from __future__ import annotations

import csv
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from src.common import ROOT, write_csv
from src.compare_heads_hierarchical import load as load_predictions
from src.external_test import CLASS_NAMES, auroc, hierarchical_difference, macro_auroc

OUT = ROOT / "results/publication/figures_20261110"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dc", "#ffffff"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # reference palette slots 1-6
MARKERS = ["o", "s", "^", "D", "v", "P"]
HEADS = ["cnn_pool", "cnn_pool_wide", "cnn_lstm", "temporal_conv", "hybrid_vit"]
NAMES = {"cnn_pool": "CNN-only", "cnn_pool_wide": "CNN-only (capacity-matched)", "cnn_lstm": "CNN-LSTM",
         "temporal_conv": "CNN + temporal conv.", "hybrid_vit": "CNN + small ViT"}
V1 = {"cnn_lstm": "ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42",
      "temporal_conv": "ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42",
      "hybrid_vit": "ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42"}
DATASET_COLORS = {"PTB-XL test": SERIES[0], "Georgia (USA)": SERIES[1], "Chapman-Shaoxing + Ningbo (China)": SERIES[2]}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def style(axis, xgrid=True):
    axis.spines[["top", "right"]].set_visible(False)
    if xgrid:
        axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)


def run_dir(head, seed, fixed=False):
    if fixed:
        return ROOT / "results/ptbxl_v2_fixedgain" / f"ptbxl_v2_fg_{head}_seed{seed}"
    if seed == 42 and head in V1:
        return ROOT / "results" / V1[head]
    return ROOT / "results/ptbxl_v2" / f"ptbxl_v2_{head}_seed{seed}"


# ---------------------------------------------------------------- figure 1: pages
def figure_pages():
    with (ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv").open(newline="") as handle:
        manifest = {int(r["ecg_id"]): r for r in csv.DictReader(handle)}
    previews = sorted((ROOT / "data/processed/ptbxl_lead_images_fixedgain2p5_v1/preview_train_pages").glob("ecg_*.png"))
    chosen = next(p for p in previews if float(manifest[int(p.stem[4:])]["gain_mm_per_mv"]) != 2.5)
    ecg = int(chosen.stem[4:])
    adaptive = cv2.imread(str(ROOT / "data/derived/ptbxl_images_v1" / manifest[ecg]["image_path"]), cv2.IMREAD_GRAYSCALE)
    fixed = cv2.imread(str(chosen), cv2.IMREAD_GRAYSCALE)
    crop = (slice(80, 80 + 3 * 240), slice(0, 860))  # three lead rows (I, II, III), pulse and the first ~3.1 s
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)
    for axis, image, title in ((axes[0], adaptive, f"(a) Adaptive gain (this ECG: {float(manifest[ecg]['gain_mm_per_mv']):g} mm/mV)"),
                               (axes[1], fixed, "(b) Fixed gain (2.5 mm/mV for every ECG)")):
        axis.imshow(image[crop], cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        axis.set_title(title, loc="left", color=INK)
        axis.set_xticks([]), axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(GRID)
    save(fig, "fig1_rendered_pages")
    return ecg


# ---------------------------------------------------------------- figure 2: architectures
def figure_architecture():
    fig, axis = plt.subplots(figsize=(7.16, 3.1))
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    axis.set_xlim(0, 100), axis.set_ylim(0, 46), axis.axis("off")

    def box(x, y, w, h, title, body, edge=MUTED, face="#f6f6f3"):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", linewidth=0.8,
                                      edgecolor=edge, facecolor=face))
        axis.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=6.3, weight="bold", color=INK)
        axis.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=5.4, color=MUTED, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        axis.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7,
                                                                        shrinkA=0, shrinkB=0))

    box(3, 33, 24, 12, "Rendered ECG page", "12 lead rows, 10 s each\n12 x 120 x 1120 ink image\ntraining-only normalisation")
    box(38, 33, 24, 12, "Shared CNN encoder", "depthwise + pointwise conv.\nchannels 12-8-16-24, GroupNorm\n1,220 parameters")
    box(73, 33, 24, 12, "Column feature sequence", "adaptive pooling to 24 x 4 x 70\n70 columns x 96 features\n(includes the calibration pulse)")
    arrow(27.4, 39, 37.6, 39), arrow(62.4, 39, 72.6, 39)
    heads = [("CNN-only", "column mean\nlinear 96-5\n1,705 parameters"),
             ("Capacity-matched", "per-column MLP\n96-96-64, no mixing\n17,065 parameters"),
             ("CNN-LSTM", "1 LSTM layer\nhidden 32, mean\n18,025 parameters"),
             ("Temporal conv.", "2 causal residual blocks\nkernel 3, dilation 1, 2\n17,161 parameters"),
             ("Small ViT", "2 layers, 4 heads\ndim 32, learned pos.\n23,881 parameters")]
    width, gap = 18.4, 1.5
    axis.plot([85, 85], [32.6, 30], color=MUTED, lw=0.7)
    axis.plot([0.6 + width / 2, 0.6 + 4 * (width + gap) + width / 2], [30, 30], color=MUTED, lw=0.7)
    for i, (title, body) in enumerate(heads):
        x = 0.6 + i * (width + gap)
        box(x, 13.5, width, 12.5, title, body, edge=SERIES[i], face=SURFACE)
        arrow(x + width / 2, 30, x + width / 2, 26.4)
        arrow(x + width / 2, 13.1, x + width / 2, 9.4)
    box(0.6, 1, 98.8, 8, "Five independent sigmoid outputs: NORM, MI, STTC, CD, HYP",
        "unweighted BCE; checkpoint chosen on validation; fixed threshold 0.5; one test inference per run")
    save(fig, "fig2_architectures")


# ---------------------------------------------------------------- figure 3: head comparison
def ptbxl_seed_scores():
    scores = {}
    with (ROOT / "results/comparisons/ptbxl_heads_hierarchical/per_seed.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["partition"] == "test":
                key = next(k for k, v in {**NAMES, "cnn_pool_wide": "CNN-only, capacity-matched",
                                          "temporal_conv": "CNN + temporal conv", "hybrid_vit": "CNN + small ViT"}.items()
                           if v == r["model"])
                scores.setdefault(key, []).append(float(r["macro_auroc"]))
    for key, relative in (("pca_lda", "ptbxl_v2/ptbxl_v2_lda_wide_seed42"), ("resnet1d_wang", None)):
        paths = [ROOT / "results" / relative] if relative else sorted((ROOT / "results/ptbxl_signal").glob("*seed4*"))
        scores[key] = [json.loads((p / "run.json").read_text())["metrics"]["test"]["macro_roc_auc"] for p in paths]
    return scores


def external_seed_scores(dataset):
    scores = {}
    with (ROOT / "results/external" / dataset / "per_run.csv").open(newline="") as handle:
        for r in csv.DictReader(handle):
            if r["page"] in ("adaptive", "signal") and not r["run"].startswith("ptbxl_v2_lda_seed"):
                key = r["head"] if r["head"] != "pca_svm" else None
                if key:
                    scores.setdefault(key, []).append(float(r["external_macro_auroc"]))
    return scores


def figure_heads():
    panels = [("PTB-XL test fold (5 labels)", ptbxl_seed_scores()),
              ("Georgia, USA (STTC, CD, HYP)", external_seed_scores("georgia")),
              ("China (MI, STTC, CD, HYP)", external_seed_scores("china"))]
    order = ["pca_lda", *HEADS, "resnet1d_wang"]
    labels = {**NAMES, "pca_lda": "PCA + LDA", "resnet1d_wang": "1D ResNet on signals (ref.)"}
    colors = {**{h: SERIES[i] for i, h in enumerate(HEADS)}, "pca_lda": MUTED, "resnet1d_wang": MUTED}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), sharey=True, constrained_layout=True)
    for axis, (title, scores) in zip(axes, panels):
        for y, key in enumerate(order):
            values = scores[key]
            marker = "o" if key in HEADS else ("s" if key == "pca_lda" else "P")
            axis.scatter(values, [y] * len(values), s=11, color=colors[key], alpha=0.45, linewidths=0, marker=marker, zorder=2)
            axis.scatter([np.mean(values)], [y], s=34, color=colors[key], edgecolors=SURFACE, linewidths=0.8, marker="D", zorder=3)
            axis.text(np.mean(values), y - 0.22, f"{np.mean(values):.3f}", ha="center", va="bottom", fontsize=5.6, color=INK)
        axis.set_title(title, color=INK)
        axis.set_xlabel("Macro AUROC")
        style(axis)
    axes[0].set_yticks(range(len(order)), [labels[k] for k in order])
    axes[0].set_ylim(-0.6, len(order) - 0.3)
    axes[0].invert_yaxis()
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=MUTED, alpha=0.45, linestyle="", markersize=3.5, label="one seed"),
                        plt.Line2D([], [], marker="D", color=MUTED, linestyle="", markersize=4.5, label="mean over seeds")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "fig3_head_comparison")


# ---------------------------------------------------------------- figure 4: fixed vs adaptive
def ptbxl_fixed_vs_adaptive():
    rows, metrics = [], {"macro AUROC": lambda y, p: macro_auroc(y, p, CLASS_NAMES), "HYP AUROC": lambda y, p: auroc(y[:, 4], p[:, 4])}
    for head in ("cnn_pool", "cnn_lstm", "temporal_conv", "hybrid_vit"):
        adaptive, fixed, reference = {}, {}, None
        for seed in (42, 43, 44, 45, 46):
            ids, patients, y, pa = load_predictions(run_dir(head, seed), "test")
            ids_f, _, y_f, pf = load_predictions(run_dir(head, seed, fixed=True), "test")
            if ids != ids_f or not np.array_equal(y, y_f) or (reference is not None and ids != reference[0]):
                raise ValueError(f"{head} seed {seed}: test records not aligned")
            reference = (ids, patients, y)
            adaptive[seed], fixed[seed] = pa, pf
        groups = {}
        for i, patient in enumerate(reference[1]):
            groups.setdefault(patient, []).append(i)
        clusters = [np.asarray(v) for v in groups.values()]
        for seed_set, seeds in (("42-46", [42, 43, 44, 45, 46]), ("43-46 GPU only", [43, 44, 45, 46])):
            for name, metric in metrics.items():
                result = hierarchical_difference(reference[2], adaptive, fixed, metric, seeds, clusters=clusters)
                rows.append({"architecture": head, "seed_set": seed_set, "metric": name,
                             **{k: v for k, v in result.items() if k != "seeds"}})
                print(f"PTB-XL fixed-adaptive {head} {seed_set} {name}: {result['difference']:+.4f} "
                      f"[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}]", flush=True)
    write_csv(ROOT / "results/fixed_gain/retrained_comparison/hierarchical_test.csv", rows)
    return rows


def figure_fixed_gain(ptbxl_rows):
    external = {}
    for dataset, name in (("georgia", "Georgia (USA)"), ("china", "Chapman-Shaoxing + Ningbo (China)")):
        with (ROOT / "results/external" / dataset / "hypotheses.csv").open(newline="") as handle:
            for r in csv.DictReader(handle):
                if r["family"] == "H2":
                    external[(name, r["comparison"].split(":")[0], r["metric"])] = r
    archs = ["cnn_pool", "temporal_conv", "cnn_lstm", "hybrid_vit"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True, constrained_layout=True)
    for axis, metric in zip(axes, ("macro AUROC", "HYP AUROC")):
        axis.axvline(0, color=MUTED, linewidth=0.7, zorder=1)
        for y, head in enumerate(archs):
            for k, (dataset, color) in enumerate(DATASET_COLORS.items()):
                if dataset == "PTB-XL test":
                    r = next(r for r in ptbxl_rows if r["architecture"] == head and r["metric"] == metric and r["seed_set"] == "42-46")
                else:
                    r = external[(dataset, head, metric)]
                d, lo, hi = float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"])
                yy = y + (k - 1) * 0.24
                axis.plot([lo, hi], [yy, yy], color=color, linewidth=1.4, solid_capstyle="round", zorder=2)
                axis.scatter([d], [yy], s=20, color=color, marker=MARKERS[k], edgecolors=SURFACE, linewidths=0.6, zorder=3)
        axis.set_title(f"Fixed minus adaptive gain: {metric}", color=INK)
        axis.set_xlabel(f"Difference in {metric} (mean over seeds, 95% CI)")
        axis.set_ylim(len(archs) - 0.5, -0.5)
        style(axis)
    axes[0].set_yticks(range(len(archs)), [NAMES[h] for h in archs])
    fig.legend(handles=[plt.Line2D([], [], color=c, marker=MARKERS[k], linewidth=1.4, markersize=4, label=n)
                        for k, (n, c) in enumerate(DATASET_COLORS.items())],
               loc="outside lower center", ncol=3, frameon=False)
    save(fig, "fig4_fixed_vs_adaptive_gain")


# ---------------------------------------------------------------- figure 5: calibration pulse
def figure_pulse():
    with (ROOT / "results/secondary/saliency/summary.csv").open(newline="") as handle:
        saliency = [r for r in csv.DictReader(handle) if r["logit"] == "HYP" and r["ecgs"] == "HYP-positive"]
    with (ROOT / "results/secondary/pulse_swap/decomposition_posthoc.csv").open(newline="") as handle:
        decomposition = {r["model"]: r for r in csv.DictReader(handle)}
    models = ["CNN-only", "Temporal conv", "CNN-LSTM", "Hybrid ViT"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), constrained_layout=True)
    axis = axes[0]
    for k, (condition, color) in enumerate((("adaptive gain", SERIES[0]), ("fixed gain", SERIES[1]))):
        for y, model in enumerate(models):
            r = next(r for r in saliency if r["model"] == model and r["condition"] == condition)
            m, lo, hi = (100 * float(r[c]) for c in ("pulse_share_mean", "pulse_share_ci_lower", "pulse_share_ci_upper"))
            yy = y + (k - 0.5) * 0.3
            axis.barh(yy, m, height=0.28, color=color, edgecolor=SURFACE, linewidth=1, zorder=2,
                      label=condition.capitalize() if y == 0 else None)
            axis.plot([lo, hi], [yy, yy], color=INK, linewidth=0.6, zorder=3)
    axis.axvline(3.6, color=MUTED, linestyle="--", linewidth=0.7)
    axis.text(3.9, -0.62, "pulse area 3.6%", fontsize=5.8, color=MUTED, va="center")
    axis.set_yticks(range(len(models)), models), axis.set_ylim(len(models) - 0.45, -0.8)
    axis.set_xlabel("Share of HYP attribution on the calibration pulse (%)")
    axis.set_title("(a) Integrated gradients, HYP-positive validation ECGs", color=INK, loc="left")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    style(axis)
    axis = axes[1]
    parts = (("consistent_rerender_hyp_change", "Gain prior (waveform and pulse redrawn together)", SERIES[2]),
             ("approx_calibration_reading", "Calibration reading (pulse-only swap minus redraw)", SERIES[4]))
    for k, (column, label, color) in enumerate(parts):
        values = [float(decomposition[m][column]) for m in models]
        axis.barh(np.arange(len(models)) + (k - 0.5) * 0.3, values, height=0.28, color=color, edgecolor=SURFACE,
                  linewidth=1, label=label, zorder=2)
    axis.axvline(0, color=MUTED, linewidth=0.7)
    axis.set_yticks(range(len(models)), models), axis.invert_yaxis()
    axis.set_xlabel("Mean change in HYP probability (pages at 5-10 mm/mV)")
    axis.set_title("(b) Post-hoc decomposition of the pulse-swap effect", color=INK, loc="left")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=5.8)
    style(axis)
    save(fig, "fig5_calibration_pulse")


def main():
    ecg = figure_pages()
    figure_architecture()
    figure_heads()
    rows = ptbxl_fixed_vs_adaptive()
    figure_fixed_gain(rows)
    figure_pulse()
    print(f"figures written to {OUT} (page example: PTB-XL ecg_id {ecg})")


if __name__ == "__main__":
    main()
