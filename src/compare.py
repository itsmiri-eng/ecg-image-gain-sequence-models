"""Compare already saved evidence; never rerun models or tune against test labels."""
import argparse
import csv
from pathlib import Path

from src.common import ROOT, read_json, write_csv, write_json
from src.evaluation.metrics import evaluate

import matplotlib.pyplot as plt
import numpy as np


def compare(lda_id, cnn_id):
    out = ROOT / "results/comparisons"
    out.mkdir(parents=True, exist_ok=True)
    runs, evidence = {}, {}
    for name, run_id in [("lda", lda_id), ("cnn", cnn_id)]:
        directory = ROOT / "results" / name / run_id
        run = read_json(directory / "run.json")
        if run["status"] != "completed":
            raise ValueError("Can only compare actually completed runs")
        with (directory / "test_predictions.csv").open() as f:
            rows = list(csv.DictReader(f))
        names = run["metrics"]["test"]["class_names"]
        y = np.array([names.index(r["true_label"]) for r in rows])
        p = np.array([[float(r[f"probability_{c}"]) for c in names] for r in rows])
        recalculated = evaluate(y, p, names)
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
            if not np.isclose(recalculated[key], run["metrics"]["test"][key], atol=1e-12):
                raise ValueError(f"Saved metric/prediction mismatch: {name}/{key}")
        runs[name] = run
        evidence[name] = rows
    for key in ["dataset_version", "split_version"]:
        if runs["lda"][key] != runs["cnn"][key]:
            raise ValueError(f"Models differ in {key}")
    if [(r["sample_id"], r["true_label"]) for r in evidence["lda"]] != [(r["sample_id"], r["true_label"]) for r in evidence["cnn"]]:
        raise ValueError("Models did not evaluate identical ordered test records")
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    table = [{"model": name.upper(), **{k: run["metrics"]["test"][k] for k in metric_keys},
              "notes": "Train-only PCA(32), uniform-prior shrinkage LDA" if name == "lda" else
              "40,212 parameters; weighted CE; validation-only early stopping",
              "runtime_seconds": run["total_runtime_seconds"], "experiment_id": run["experiment_id"]}
             for name, run in runs.items()]
    write_csv(out / "comparison.csv", table)
    per_class = []
    for name in names:
        a, b = [runs[m]["metrics"]["test"]["per_class"][name] for m in ["lda", "cnn"]]
        per_class.append({"class": name, "test_support": a["support"],
                          "lda_precision": a["precision"], "cnn_precision": b["precision"],
                          "lda_recall": a["recall"], "cnn_recall": b["recall"],
                          "lda_f1": a["f1-score"], "cnn_f1": b["f1-score"],
                          "cnn_minus_lda_f1": b["f1-score"] - a["f1-score"]})
    write_csv(out / "per_class_comparison.csv", per_class)
    paired = {"both_correct": 0, "lda_only_correct": 0, "cnn_only_correct": 0, "both_wrong": 0}
    for a, b in zip(evidence["lda"], evidence["cnn"]):
        ac, bc = a["true_label"] == a["predicted_label"], b["true_label"] == b["predicted_label"]
        paired["both_correct" if ac and bc else "lda_only_correct" if ac else "cnn_only_correct" if bc else "both_wrong"] += 1
    write_json(out / "comparison_verification.json", {"saved_predictions_match_metrics": True,
               "identical_test_records": True, "dataset_version": runs["lda"]["dataset_version"],
               "split_version": runs["lda"]["split_version"], "paired_correctness": paired})
    text = ["# Executed LDA vs CNN comparison", "",
            "Both models used the same frozen cropped-image dataset and identity-grouped split. Metrics below are recomputed from saved test predictions; no model was refitted to produce this comparison.", "",
            "| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Notes |",
            "|---|---:|---:|---:|---:|---|"]
    for row in table:
        text.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['notes']} |")
    deltas = {k: runs["cnn"]["metrics"]["test"][k] - runs["lda"]["metrics"]["test"][k] for k in metric_keys}
    text += ["", f"CNN minus LDA: accuracy {deltas['accuracy']:+.4f}, balanced accuracy {deltas['balanced_accuracy']:+.4f}, macro F1 {deltas['macro_f1']:+.4f}, weighted F1 {deltas['weighted_f1']:+.4f}.", "",
             "These are observed differences on one small split, not evidence of general clinical superiority. Printed IDs are unverified patient proxies, and near-duplicate screening is not proof of independence.", "",
             "| Class | Test n | LDA recall | CNN recall | LDA F1 | CNN F1 | F1 change |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in per_class:
        text.append(f"| {row['class']} | {row['test_support']:.0f} | {row['lda_recall']:.4f} | {row['cnn_recall']:.4f} | {row['lda_f1']:.4f} | {row['cnn_f1']:.4f} | {row['cnn_minus_lda_f1']:+.4f} |")
    text += ["", "## Generalization checks", "", "| Model | Train accuracy | Validation accuracy | Test accuracy | Train macro F1 | Validation macro F1 | Test macro F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, run in runs.items():
        mm = run["metrics"]
        values = [mm[s][k] for k in ["accuracy", "macro_f1"] for s in ["train", "validation", "test"]]
        text.append("| " + model.upper() + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    train = runs["cnn"]["training"]
    text += ["", f"CNN selected epoch {train['best_epoch']} of {train['epochs_run']} by minimum weighted validation loss. See its saved history/curve for the training and validation trajectories.", "",
             "LDA tests linear discriminative structure after an unsupervised train-only 32-component PCA projection. Good or poor performance here does not establish whether raw ECG physiology is linearly separable: crop resolution, PCA information loss, source artifacts, class overlap, and small support all matter.", "",
             "Both models may use page-layout/grid artifacts remaining within the trace crop. The 128×256 image reduction is a computational baseline, not a claim of clinically lossless digitization. Neither model receives calibrated physical waveforms.", "",
             "Resolve the supervisor's LSM and remembered convolution term before implementing those steps. Before CNN+LSTM, validate lead/time extraction and obtain more independent minority-class data.", ""]
    (out / "comparison.md").write_text("\n".join(text))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    locations = np.arange(4)
    for j, model in enumerate(["lda", "cnn"]):
        axes[0].bar(locations + (j - .5) * .34, [runs[model]["metrics"]["test"][k] for k in metric_keys], .34, label=model.upper())
        axes[1].bar(locations + (j - .5) * .34, [runs[model]["metrics"]["test"]["per_class"][c]["f1-score"] for c in names], .34, label=model.upper())
    axes[0].set_xticks(locations, ["Accuracy", "Balanced\naccuracy", "Macro F1", "Weighted F1"])
    axes[1].set_xticks(locations, names)
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("Same held-out test records")
    axes[1].set_title("Per-class test F1")
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    print("Comparison verified and saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lda-id", default="lda_v1_seed42")
    p.add_argument("--cnn-id", default="cnn_v1_seed42")
    a = p.parse_args()
    compare(a.lda_id, a.cnn_id)
