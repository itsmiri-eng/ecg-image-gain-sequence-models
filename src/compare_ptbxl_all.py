"""Master PTB-XL comparison over every completed five-label run (v1 and v2 protocols).

Usage:
    python -m src.compare_ptbxl_all

Reads saved predictions only; no model is trained or re-run. Writes
results/comparisons/ptbxl_master/: master table, per-label table, seed summary,
paired patient-bootstrap differences and a summary figure.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common import ROOT, read_json, write_csv, write_json
from src.compare_ptbxl_models import paired_patient_bootstrap
from src.evaluation.multilabel import evaluate_multilabel

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
OUT = ROOT / "results/comparisons/ptbxl_master"
V1_RUNS = {
    "results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42": "cnn_lstm",
    "results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42": "temporal_conv",
    "results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42": "hybrid_vit",
}
ORDER = ["pca_svm", "pca_lda", "pca_lda_wide", "cnn_pool", "cnn_pool_wide", "cnn_lstm", "temporal_conv", "hybrid_vit"]
LABELS = {"pca_svm": "PCA + linear SVM", "pca_lda": "PCA + LDA (grid ≤256)","pca_lda_wide": "PCA + LDA (grid ≤1024)", "cnn_pool": "CNN (pooled)",
          "cnn_pool_wide": "CNN (pooled, capacity-matched)", "cnn_lstm": "CNN-LSTM",
          "temporal_conv": "CNN + temporal conv", "hybrid_vit": "CNN + small ViT"}


def load_predictions(directory, split="test"):
    with (directory / f"{split}_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    truth = np.asarray([[int(r[f"truth_{n}"]) for n in CLASS_NAMES] for r in rows])
    probability = np.asarray([[float(r[f"probability_{n}"]) for n in CLASS_NAMES] for r in rows])
    return [r["ecg_id"] for r in rows], [r["patient_id"] for r in rows], truth, probability


def discover():
    runs = []
    for relative, model in V1_RUNS.items():
        directory = ROOT / relative
        if (directory / "run.json").exists() and read_json(directory / "run.json").get("status") == "completed":
            runs.append({"directory": directory, "model": model, "protocol": "v1", "seed": 42})
    for run_file in sorted((ROOT / "results/ptbxl_v2").glob("*/run.json")):
        run = read_json(run_file)
        if run.get("status") == "completed":
            model = "pca_lda_wide" if run_file.parent.name.startswith("ptbxl_v2_lda_wide") else run["model"]
            runs.append({"directory": run_file.parent, "model": model, "protocol": "v2", "seed": run["seed"]})
    return runs


def describe(run):
    result = read_json(run["directory"] / "run.json")
    training = result.get("training", {})
    params = training.get("num_parameters")
    if params is None:  # PCA + LDA/SVM: a fitted linear projection, not trained weights; report its size instead
        params = (f"PCA {result['pca_components']} + 5 SVM (C={result['C']})" if run["model"] == "pca_svm"
                  else f"PCA {result['pca_components']} + 5 LDA")
    _, _, truth, probability = load_predictions(run["directory"])
    test = evaluate_multilabel(truth, probability, CLASS_NAMES)
    validation = read_json(run["directory"] / "validation_metrics.json")
    train = read_json(run["directory"] / "train_metrics.json")
    ci = read_json(run["directory"] / "test_patient_bootstrap.json")["metrics"]["macro_roc_auc"]
    return {
        "model": LABELS[run["model"]], "model_key": run["model"], "protocol": run["protocol"], "seed": run["seed"],
        "run_id": run["directory"].name, "test_macro_auroc": test["macro_roc_auc"],
        "test_macro_auroc_ci": f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]",
        "test_macro_ap": test["macro_average_precision"], "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"], "test_micro_f1": test["micro_f1"],
        "test_exact_match": test["subset_accuracy"], "test_label_accuracy": test["label_accuracy"],
        "test_mean_label_balanced_accuracy": float(np.mean([test["per_label"][n]["balanced_accuracy"] for n in CLASS_NAMES])),
        "validation_macro_auroc": validation["macro_roc_auc"], "train_macro_auroc": train["macro_roc_auc"],
        "train_minus_validation_auroc": train["macro_roc_auc"] - validation["macro_roc_auc"],
        "parameters": params, "epochs_run": training.get("epochs_run", ""), "selected_epoch": training.get("best_epoch", ""),
        "runtime_seconds": result.get("total_runtime_seconds", ""),
    }, test


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = discover()
    rows, per_label, probabilities = [], [], {}
    reference_ids = None
    for run in runs:
        row, test = describe(run)
        ids, patients, truth, probability = load_predictions(run["directory"])
        if reference_ids is None:
            reference_ids, reference_patients, reference_truth = ids, patients, truth
        if ids != reference_ids or not np.array_equal(truth, reference_truth):
            raise ValueError(f"{row['run_id']} does not use the identical ordered test cohort")
        probabilities[row["run_id"]] = probability
        rows.append(row)
        for name in CLASS_NAMES:
            m = test["per_label"][name]
            per_label.append({"model": row["model"], "run_id": row["run_id"], "seed": row["seed"], "label": name,
                              "support": m["support"], "recall_sensitivity": m["recall"], "specificity": m["specificity"],
                              "precision": m["precision"], "f1": m["f1"], "roc_auc": m["roc_auc"],
                              "average_precision": m["average_precision"], "false_negatives": m["fn"]})
    rows.sort(key=lambda r: (ORDER.index(r["model_key"]), r["protocol"], r["seed"]))
    write_csv(OUT / "master_table.csv", rows)
    write_csv(OUT / "per_label.csv", per_label)

    # Seed summary per architecture: v2 protocol only (every seed shares the identical trainer), and all
    # seeds 42/43/44, where seed 42 of the sequence models is the v1 run. Pooling is supported by
    # experiments/reviews/2026-11-01/trainer_equivalence_check.json (v1 and v2 trainers give bit-identical
    # weights on real training data); seed 42 also ran on the Mac CPU and seeds 43/44 on the PC GPU.
    def summarize(keep):
        by_model = defaultdict(list)
        for row in rows:
            if keep(row):
                by_model[row["model_key"]].append(row)
        result = []
        for key in ORDER:
            group = by_model.get(key, [])
            if not group:
                continue
            summary = {"model": LABELS[key], "n_seeds": len(group), "seeds": " ".join(str(r["seed"]) for r in group)}
            for metric in ("test_macro_auroc", "test_macro_ap", "test_macro_f1", "validation_macro_auroc"):
                values = np.asarray([r[metric] for r in group])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else ""
            result.append(summary)
        return result

    seed_summary = summarize(lambda row: row["protocol"] == "v2")
    pooled_summary = summarize(lambda row: True)
    if seed_summary:
        write_csv(OUT / "seed_summary.csv", seed_summary)
        write_csv(OUT / "seed_summary_all_seeds.csv", pooled_summary)

    # Paired test differences on the identical patients, each seed-42 model against the CNN-only baseline
    # and against the previous step in the progression. Exploratory; fold 10 has been observed before.
    seed42 = {}
    for row in rows:
        if row["seed"] == 42 and (row["model_key"] not in seed42 or row["protocol"] == "v1"):
            seed42[row["model_key"]] = row["run_id"]
    # The validation-tuned wide-grid LDA is the statistical reference when present.
    present = [key for key in ORDER if key in seed42 and not (key == "pca_lda" and "pca_lda_wide" in seed42)]
    pairs = [(present[i - 1], present[i]) for i in range(1, len(present))]
    pairs += [("cnn_pool", key) for key in present if key not in ("cnn_pool", "pca_svm", "pca_lda", "pca_lda_wide") and ("cnn_pool", key) not in pairs]
    differences = []
    for old, new in pairs:
        if old not in seed42:
            continue
        result = paired_patient_bootstrap(reference_truth, probabilities[seed42[old]], probabilities[seed42[new]],
                                          reference_patients, CLASS_NAMES, n_resamples=500, seed=2024)
        for metric in ("macro_roc_auc", "macro_average_precision", "macro_f1"):
            item = result["metrics"][metric]
            differences.append({"comparison": f"{LABELS[new]} minus {LABELS[old]}", "old_run": seed42[old],
                                "new_run": seed42[new], "metric": metric, "difference": item["difference"],
                                "ci_lower": item["ci_lower"], "ci_upper": item["ci_upper"],
                                "interpretation": item["interpretation"]})
    if differences:
        write_csv(OUT / "paired_differences_seed42.csv", differences)

    lines = ["# PTB-XL master comparison (five overlapping labels, official patient folds)", "",
             "Generated by `python -m src.compare_ptbxl_all` from saved predictions; no model was re-run. "
             "All runs share the identical 2,158-ECG / 1,877-patient test fold (fold 10), training-only normalization, "
             "threshold 0.5 and validation-only selection. Fold 10 was already observed before the later models were "
             "specified, so every comparison is exploratory.", "",
             "| Model | Protocol | Seed | Macro AUROC [95% patient CI] | Macro AP | Macro F1 | Weighted F1 | Exact match | "
             "Label accuracy | Mean label bal. acc. | Val macro AUROC | Params | Epochs (sel.) | Runtime (s) |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        runtime = f"{r['runtime_seconds']:.0f}" if isinstance(r["runtime_seconds"], float) else ""
        params = f"{r['parameters']:,}" if isinstance(r["parameters"], int) else r["parameters"]
        lines.append(f"| {r['model']} | {r['protocol']} | {r['seed']} | {r['test_macro_auroc']:.4f} {r['test_macro_auroc_ci']} | "
                     f"{r['test_macro_ap']:.4f} | {r['test_macro_f1']:.4f} | {r['test_weighted_f1']:.4f} | "
                     f"{r['test_exact_match']:.4f} | {r['test_label_accuracy']:.4f} | "
                     f"{r['test_mean_label_balanced_accuracy']:.4f} | {r['validation_macro_auroc']:.4f} | "
                     f"{params} | {r['epochs_run']} ({r['selected_epoch']}) | {runtime} |")
    lines += ["", "## Test sensitivity (recall) per label", "",
              "| Model | Seed | " + " | ".join(CLASS_NAMES) + " |", "|---|---:|" + "---:|" * len(CLASS_NAMES)]
    for r in rows:
        values = {p["label"]: p["recall_sensitivity"] for p in per_label if p["run_id"] == r["run_id"]}
        lines.append(f"| {r['model']} ({r['protocol']}) | {r['seed']} | " + " | ".join(f"{values[n]:.3f}" for n in CLASS_NAMES) + " |")
    for title, summaries, note in [
            ("Seed variability (protocol v2, identical trainer)", seed_summary, None),
            ("Seed variability, all seeds (v1 seed 42 pooled with v2 seeds 43/44)", pooled_summary,
             "Pooling is supported by the trainer equivalence check (v1 and v2 trainers give bit-identical weights; "
             "experiments/reviews/2026-11-01/trainer_equivalence_check.json). Seed 42 was trained on the Mac CPU, "
             "seeds 43/44 on the PC GPU, so the SD also contains any hardware effect.")]:
        if not summaries:
            continue
        lines += ["", f"## {title}", "", *([note, ""] if note else []),
                  "| Model | Seeds | Test macro AUROC mean ± SD | Test macro F1 mean ± SD | Val macro AUROC mean ± SD |",
                  "|---|---|---|---|---|"]
        for s in summaries:
            def ms(metric):
                sd = s[f"{metric}_sd"]
                return f"{s[f'{metric}_mean']:.4f}" + (f" ± {sd:.4f}" if sd != "" else " (1 seed)")
            lines.append(f"| {s['model']} | {s['seeds']} | {ms('test_macro_auroc')} | {ms('test_macro_f1')} | "
                         f"{ms('validation_macro_auroc')} |")
    if differences:
        lines += ["", "## Paired test differences, seed 42 (500 patient-cluster bootstrap resamples)", "",
                  "| Comparison | Metric | Difference | 95% CI | Reading |", "|---|---|---:|---|---|"]
        for d in differences:
            lines.append(f"| {d['comparison']} | {d['metric']} | {d['difference']:+.4f} | "
                         f"[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}] | {d['interpretation']} |")
        lines += ["", "Intervals are conditional on one fitted model per architecture and exclude training-seed "
                  "variability; no multiplicity correction is applied. Treat them as exploratory."]
    (OUT / "master_table.md").write_text("\n".join(lines) + "\n")
    _plot(rows, per_label)
    print("\n".join(lines))


def _plot(rows, per_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seed42 = [r for r in rows if r["seed"] == 42 and not (r["protocol"] == "v2" and any(
        o["model_key"] == r["model_key"] and o["protocol"] == "v1" for o in rows))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    names = [r["model"] for r in seed42]
    values = [r["test_macro_auroc"] for r in seed42]
    errors = []
    for r in seed42:
        lo, hi = [float(x) for x in r["test_macro_auroc_ci"].strip("[]").split(",")]
        errors.append((r["test_macro_auroc"] - lo, hi - r["test_macro_auroc"]))
    axes[0].errorbar(values, range(len(names)), xerr=np.asarray(errors).T, fmt="o", color="#2a6f97", capsize=3)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test macro AUROC (95% patient-bootstrap CI)")
    axes[0].grid(axis="x", alpha=.3)
    width = 0.8 / max(len(seed42), 1)
    for i, r in enumerate(seed42):
        recalls = [next(p["recall_sensitivity"] for p in per_label if p["run_id"] == r["run_id"] and p["label"] == n)
                   for n in CLASS_NAMES]
        axes[1].bar(np.arange(5) + (i - (len(seed42) - 1) / 2) * width, recalls, width, label=r["model"])
    axes[1].set_xticks(range(5), CLASS_NAMES)
    axes[1].set_ylabel("Test sensitivity at threshold 0.5")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(axis="y", alpha=.3)
    fig.suptitle("PTB-XL rendered-image benchmark, fold 10 test (2,158 ECGs), seed 42")
    fig.savefig(OUT / "master_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
