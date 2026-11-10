"""LaTeX tables for the IEEE manuscript, generated from saved results only (no transcription by hand).

Usage:
    python -X utf8 -m src.make_paper_tables

Writes paper/ieee_manuscript/tables/*.tex. Run src.make_paper_figures first (it writes the PTB-XL hierarchical
fixed-vs-adaptive file used by tab_fixed_gain).
"""
from __future__ import annotations

import csv
import json

import numpy as np

from src.common import ROOT
from src.external_test import holm

OUT = ROOT / "paper/ieee_manuscript/tables"
LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
HEADS = ("cnn_pool", "cnn_pool_wide", "cnn_lstm", "temporal_conv", "hybrid_vit")
NAMES = {"cnn_pool": "CNN-only", "cnn_pool_wide": "CNN-only, capacity-matched", "cnn_lstm": "CNN-LSTM",
         "temporal_conv": "CNN + temporal conv.", "hybrid_vit": "CNN + small ViT"}
V1 = {"cnn_lstm": "ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42",
      "temporal_conv": "ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42",
      "hybrid_vit": "ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42"}
SEEDS = (42, 43, 44, 45, 46)


def run_path(head, seed, fixed=False):
    if fixed:
        return ROOT / "results/ptbxl_v2_fixedgain" / f"ptbxl_v2_fg_{head}_seed{seed}"
    if seed == 42 and head in V1:
        return ROOT / "results" / V1[head]
    return ROOT / "results/ptbxl_v2" / f"ptbxl_v2_{head}_seed{seed}"


def test_metrics(path):
    m = json.loads((path / "run.json").read_text())["metrics"]["test"]
    return [m["macro_roc_auc"], m["macro_average_precision"], m["macro_f1"]] + [m["per_label"][l]["roc_auc"] for l in LABELS]


def fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def fmt_diff(d, lo, hi):
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("-", "$-$")


def table_ptbxl():
    rows = []
    for fixed in (False, True):
        for head in HEADS:
            if fixed and head == "cnn_pool_wide":
                continue
            a = np.asarray([test_metrics(run_path(head, s, fixed)) for s in SEEDS])
            mean, sd = a.mean(0), a.std(0, ddof=1)
            rows.append((("Fixed" if fixed else "Adaptive"), NAMES[head], f"{mean[0]:.3f} $\\pm$ {sd[0]:.3f}",
                         f"{mean[1]:.3f}", f"{mean[2]:.3f}", *[f"{v:.3f}" for v in mean[3:]]))
    for relative, name in (("ptbxl_v2/ptbxl_v2_lda_wide_seed42", "PCA + LDA"), ("ptbxl_v2/ptbxl_v2_svm_seed42", "PCA + linear SVM"),
                           ("ptbxl_v2_fixedgain/ptbxl_v2_fg_lda_seed42", "PCA + LDA")):
        v = test_metrics(ROOT / "results" / relative)
        rows.append(("Fixed" if "fixedgain" in relative else "Adaptive", name + "$^\\dagger$", f"{v[0]:.3f}", f"{v[1]:.3f}",
                     f"{v[2]:.3f}", *[f"{x:.3f}" for x in v[3:]]))
    sig = np.asarray([test_metrics(p) for p in sorted((ROOT / "results/ptbxl_signal").glob("*seed4*"))])
    rows.append(("Signal", "1D ResNet (reference)$^\\ddagger$", f"{sig[:, 0].mean():.3f} $\\pm$ {sig[:, 0].std(ddof=1):.3f}",
                 f"{sig[:, 1].mean():.3f}", f"{sig[:, 2].mean():.3f}", *[f"{x:.3f}" for x in sig.mean(0)[3:]]))
    body = []
    previous = None
    for r in rows:
        if previous is not None and r[0] != previous:
            body.append(r"\midrule")
        body.append(" & ".join(r) + r" \\")
        previous = r[0]
    (OUT / "tab_ptbxl.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def pairwise(path, partition=None, seed_set=None, family=None):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if partition:
        rows = [r for r in rows if r["partition"] == partition and r["seed_set"] == seed_set]
    if family:
        rows = [r for r in rows if r["family"] == family]
    return rows


def table_pairs():
    ptbxl = pairwise(ROOT / "results/comparisons/ptbxl_heads_hierarchical/pairwise.csv", "test", "42-46 (primary)")
    georgia = pairwise(ROOT / "results/external/georgia/hypotheses.csv", family="H1")
    china = pairwise(ROOT / "results/external/china/hypotheses.csv", family="H1")
    long = {"CNN-only": "cnn_pool", "CNN-only, capacity-matched": "cnn_pool_wide", "CNN-LSTM": "cnn_lstm",
            "CNN + temporal conv": "temporal_conv", "CNN + small ViT": "hybrid_vit"}
    body = []
    for r in ptbxl:
        b, a = r["comparison"].split(" minus ")
        key = f"{long[b]} - {long[a]}"
        cells = [f"{NAMES[long[b]]} $-$ {NAMES[long[a]]}"]
        for row in (r, next(x for x in georgia if x["comparison"] == key), next(x for x in china if x["comparison"] == key)):
            cells += [fmt_diff(float(row["difference"]), float(row["ci_lower"]), float(row["ci_upper"])), fmt_p(float(row["p_holm"]))]
        body.append(" & ".join(cells) + r" \\")
    (OUT / "tab_pairs.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def table_fixed_gain():
    with (ROOT / "results/fixed_gain/retrained_comparison/hierarchical_test.csv").open(newline="") as handle:
        ptbxl = [r for r in csv.DictReader(handle) if r["seed_set"] == "42-46"]
    for r, adjusted in zip(ptbxl, holm([float(r["p_bootstrap"]) for r in ptbxl])):
        r["p_holm"] = adjusted
    external = {d: pairwise(ROOT / f"results/external/{d}/hypotheses.csv", family="H2") for d in ("georgia", "china")}
    body = []
    for head in ("cnn_pool", "temporal_conv", "cnn_lstm", "hybrid_vit"):
        for metric in ("macro AUROC", "HYP AUROC"):
            cells = [NAMES[head] if metric == "macro AUROC" else "", metric.replace(" AUROC", "")]
            rows = [next(r for r in ptbxl if r["architecture"] == head and r["metric"] == metric)]
            rows += [next(r for r in external[d] if r["comparison"].startswith(head + ":") and r["metric"] == metric)
                     for d in ("georgia", "china")]
            for row in rows:
                cells += [fmt_diff(float(row["difference"]), float(row["ci_lower"]), float(row["ci_upper"])), fmt_p(float(row["p_holm"]))]
            body.append(" & ".join(cells) + r" \\")
        if head != "hybrid_vit":
            body.append(r"\addlinespace[2pt]")
    (OUT / "tab_fixed_gain.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def table_external():
    data = {}
    for d in ("georgia", "china"):
        with (ROOT / f"results/external/{d}/per_run.csv").open(newline="") as handle:
            for r in csv.DictReader(handle):
                if r["run"] == "ptbxl_v2_lda_seed42":
                    continue  # the narrow-grid LDA; the table reports the wide-grid fit, as in the PTB-XL table
                data.setdefault((r["head"], r["page"]), {}).setdefault(d, []).append(
                    (float(r["external_macro_auroc"]), float(r["external_minus_ptbxl"])))
    order = [(h, "adaptive") for h in HEADS] + [(h, "fixed") for h in ("cnn_pool", "cnn_lstm", "temporal_conv", "hybrid_vit")]
    order += [("pca_lda", "adaptive"), ("pca_svm", "adaptive"), ("pca_lda", "fixed"), ("resnet1d_wang", "signal")]
    names = {**NAMES, "pca_lda": "PCA + LDA", "pca_svm": "PCA + linear SVM", "resnet1d_wang": "1D ResNet (signal)"}
    body, previous = [], None
    for head, page in order:
        if previous is not None and page != previous:
            body.append(r"\midrule")
        cells = [page.capitalize(), names[head]]
        for d in ("georgia", "china"):
            v = np.asarray(data[(head, page)][d])
            sd = f" $\\pm$ {v[:, 0].std(ddof=1):.3f}" if len(v) > 1 else ""
            cells += [f"{v[:, 0].mean():.3f}{sd}", f"{v[:, 1].mean():+.3f}".replace("-", "$-$")]
        body.append(" & ".join(cells) + r" \\")
        previous = page
    (OUT / "tab_external.tex").write_text("\n".join(body) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_ptbxl()
    table_pairs()
    table_fixed_gain()
    table_external()
    for path in sorted(OUT.glob("*.tex")):
        print(f"== {path.name}\n{path.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
