"""Q2 of docs/ptbxl_fixed_gain_protocol.md: fixed-gain-trained vs adaptive-gain-trained models.

Usage:
    python -m src.compare_fixed_gain

Pairs each completed fixed-gain run with the adaptive-gain run of the same
architecture and seed, using saved predictions only. Validation is primary;
test differences are exploratory. Strata use each ECG's original (v1) page gain.
Seed 42 pairs a Mac-CPU adaptive run with a GPU fixed-gain run; seeds 43-46 are
GPU on both sides (the matched-hardware check recommended in PROJECT_STATUS).
Paired patient bootstraps cover macro AUROC and, per protocol, HYP AUROC overall
and within the original 5-10 and 2.5 mm/mV strata.
Writes results/fixed_gain/retrained_comparison/.
"""
from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import roc_auc_score

from src.common import ROOT, write_csv
from src.compare_ptbxl_models import _patient_groups, _sample_record_indices, paired_patient_bootstrap
from src.evaluation.multilabel import evaluate_multilabel

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
OUT = ROOT / "results/fixed_gain/retrained_comparison"
PAIRS = {  # name: (adaptive-gain run, fixed-gain run), seed 42
    "PCA + LDA": ("ptbxl_v2/ptbxl_v2_lda_wide_seed42", "ptbxl_v2_fixedgain/ptbxl_v2_fg_lda_seed42"),
    "CNN (pooled)": ("ptbxl_v2/ptbxl_v2_cnn_pool_seed42", "ptbxl_v2_fixedgain/ptbxl_v2_fg_cnn_pool_seed42"),
    "CNN-LSTM": ("ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42", "ptbxl_v2_fixedgain/ptbxl_v2_fg_cnn_lstm_seed42"),
    "CNN + temporal conv": ("ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42",
                            "ptbxl_v2_fixedgain/ptbxl_v2_fg_temporal_conv_seed42"),
    "CNN + small ViT": ("ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42", "ptbxl_v2_fixedgain/ptbxl_v2_fg_hybrid_vit_seed42"),
}
KEYS = {"CNN (pooled)": "cnn_pool", "CNN-LSTM": "cnn_lstm", "CNN + temporal conv": "temporal_conv", "CNN + small ViT": "hybrid_vit"}
STRATA = ("all", "v1 gain 5 or 10", "v1 gain 2.5")


def paired_label_auroc(y, old, new, patients, n_resamples=500, seed=2024):
    """One label's AUROC, new minus old, with the same patient-cluster resampling as paired_patient_bootstrap."""
    groups, rng = _patient_groups(patients, len(y)), np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        indices, _ = _sample_record_indices(groups, rng)
        if 0 < y[indices].sum() < len(indices):
            differences.append(roc_auc_score(y[indices], new[indices]) - roc_auc_score(y[indices], old[indices]))
    lower, upper = [float(x) for x in np.quantile(differences, [.025, .975], method="linear")]
    a, b = float(roc_auc_score(y, old)), float(roc_auc_score(y, new))
    return {"old_point_estimate": a, "new_point_estimate": b, "difference": b - a, "ci_lower": lower, "ci_upper": upper}


def seed_pairs():
    """(model, seed, hardware, adaptive run, fixed run) for every seed that has both runs."""
    for name, (adaptive, fixed) in PAIRS.items():
        yield name, 42, "Mac CPU vs PC GPU" if name in KEYS else "CPU vs CPU", adaptive, fixed
    for seed in (43, 44, 45, 46):  # 45/46 added by docs/protocol_amendment_2026-11-03.md
        for name, key in KEYS.items():
            yield (name, seed, "PC GPU vs PC GPU", f"ptbxl_v2/ptbxl_v2_{key}_seed{seed}",
                   f"ptbxl_v2_fixedgain/ptbxl_v2_fg_{key}_seed{seed}")


def load(run, split):
    with (ROOT / "results" / run / f"{split}_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return ([r["ecg_id"] for r in rows], [r["patient_id"] for r in rows],
            np.asarray([[int(r[f"truth_{n}"]) for n in CLASS_NAMES] for r in rows]),
            np.asarray([[float(r[f"probability_{n}"]) for n in CLASS_NAMES] for r in rows]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv").open() as handle:
        v1_gain = {r["ecg_id"]: float(r["gain_mm_per_mv"]) for r in csv.DictReader(handle)}
    summary, per_label = [], []
    for name, seed, hardware, adaptive, fixed in seed_pairs():
        if not ((ROOT / "results" / adaptive / "run.json").exists() and (ROOT / "results" / fixed / "run.json").exists()):
            continue
        for split in ("validation", "test"):
            ids, patients, truth, p_adaptive = load(adaptive, split)
            ids_f, _, truth_f, p_fixed = load(fixed, split)
            if ids != ids_f or not np.array_equal(truth, truth_f):
                raise ValueError(f"{name} seed {seed}: partitions not aligned")
            gains = np.asarray([v1_gain[i] for i in ids])
            masks = {"all": np.ones(len(ids), bool), "v1 gain 5 or 10": gains >= 5, "v1 gain 2.5": gains == 2.5}
            for metric, stratum in [("macro AUROC", "all")] + [("HYP AUROC", s) for s in STRATA]:
                mask = masks[stratum]
                kept = [p for p, m in zip(patients, mask) if m]
                if metric == "macro AUROC":
                    item = paired_patient_bootstrap(truth, p_adaptive, p_fixed, patients, CLASS_NAMES,
                                                    n_resamples=500, seed=2024)["metrics"]["macro_roc_auc"]
                else:
                    item = paired_label_auroc(truth[mask, 4], p_adaptive[mask, 4], p_fixed[mask, 4], kept)
                summary.append({"model": name, "seed": seed, "hardware": hardware, "partition": split, "metric": metric,
                                "stratum": stratum, "ecgs": int(mask.sum()), "adaptive": item["old_point_estimate"],
                                "fixed": item["new_point_estimate"], "difference_fixed_minus_adaptive": item["difference"],
                                "ci_lower": item["ci_lower"], "ci_upper": item["ci_upper"]})
            for stratum, mask in masks.items():
                a = evaluate_multilabel(truth[mask], p_adaptive[mask], CLASS_NAMES)["per_label"]
                f = evaluate_multilabel(truth[mask], p_fixed[mask], CLASS_NAMES)["per_label"]
                for label in CLASS_NAMES:
                    per_label.append({"model": name, "seed": seed, "partition": split, "stratum": stratum, "label": label,
                                      "positives": a[label]["support"], "auroc_adaptive": a[label]["roc_auc"],
                                      "auroc_fixed": f[label]["roc_auc"], "sensitivity_adaptive": a[label]["recall"],
                                      "sensitivity_fixed": f[label]["recall"]})
        print(f"{name} seed {seed}: done", flush=True)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "per_label.csv", per_label)

    def row(s):
        return (f"| {s['model']} | {s['seed']} | {s['hardware']} | {s['adaptive']:.4f} | {s['fixed']:.4f} | "
                f"{s['difference_fixed_minus_adaptive']:+.4f} [{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}] |")

    lines = ["# Q2: fixed-gain-trained vs adaptive-gain-trained models (seeds 42-46)", "",
             "Generated by `python -m src.compare_fixed_gain` from saved predictions. Validation is primary; test is "
             "exploratory (fold 10 observed before). Paired patient bootstrap, 500 resamples, per seed. Seed 42 networks "
             "pair a Mac-CPU adaptive run with a GPU fixed-gain run; seeds 43-46 are GPU on both sides.", ""]
    for split in ("validation", "test"):
        for metric, stratum in [("macro AUROC", "all")] + [("HYP AUROC", s) for s in STRATA]:
            chosen = [s for s in summary if s["partition"] == split and s["metric"] == metric and s["stratum"] == stratum]
            where = "all ECGs" if stratum == "all" else f"ECGs originally at {stratum[8:]} mm/mV"
            lines += [f"## {split.capitalize()}: {metric}, {where}", "",
                      "| Model | Seed | Hardware | Adaptive | Fixed | Fixed − adaptive [95% CI] |", "|---|---:|---|---:|---:|---|"]
            lines += [row(s) for s in chosen] + [""]
        matched = [s for s in summary if s["partition"] == split and s["metric"] == "macro AUROC" and s["seed"] in (43, 44, 45, 46)]
        lines += [f"Mean fixed − adaptive macro AUROC over the matched GPU seeds 43-46 ({split}): " + "; ".join(
            f"{name} {np.mean([s['difference_fixed_minus_adaptive'] for s in matched if s['model'] == name]):+.4f}"
            for name in KEYS), ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
