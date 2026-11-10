"""Compare three saved image-model runs without fitting or new inference."""
import argparse
import csv
from pathlib import Path

import numpy as np

from src.common import ROOT, read_json, sha256_file, write_csv, write_json
from src.evaluation.metrics import evaluate


def compare(cnn_lstm_id):
    out = ROOT / "results/comparisons" / cnn_lstm_id
    if out.exists():
        raise FileExistsError("Comparison already exists; do not overwrite research evidence")
    records = [("LDA", "lda", "lda_v1_seed42"), ("CNN", "cnn", "cnn_v1_seed42"),
               ("CNN–LSTM", "cnn_lstm", cnn_lstm_id)]
    runs, predictions, hashes = {}, {}, {}
    metric_keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro"]
    for display, directory, run_id in records:
        folder = ROOT / "results" / directory / run_id
        run = read_json(folder / "run.json")
        if run["status"] != "completed":
            raise ValueError("Only completed runs can be compared")
        runs[display] = run
        predictions[display] = {}
        hashes[str((folder / "run.json").relative_to(ROOT))] = sha256_file(folder / "run.json")
        for subset in ["train", "validation", "test"]:
            path = folder / f"{subset}_predictions.csv"
            with path.open() as f:
                rows = list(csv.DictReader(f))
            names = run["metrics"][subset]["class_names"]
            truth = np.asarray([names.index(row["true_label"]) for row in rows])
            p = np.asarray([[float(row[f"probability_{c}"]) for c in names] for row in rows])
            checked = evaluate(truth, p, names)
            for key in metric_keys:
                if not np.isclose(checked[key], run["metrics"][subset][key], rtol=0, atol=1e-12):
                    raise ValueError(f"Saved predictions/metrics disagree: {display}, {subset}, {key}")
            if checked["confusion_matrix"] != run["metrics"][subset]["confusion_matrix"]:
                raise ValueError("Confusion matrix differs from saved predictions")
            predictions[display][subset] = rows
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    reference = runs["LDA"]
    for display, run in runs.items():
        for key in ["dataset_version", "split_version"]:
            if run[key] != reference[key]:
                raise ValueError(f"Different {key}")
        for subset in ["train", "validation", "test"]:
            identity = lambda rows: [(r["sample_id"], r["path"], r["true_label"]) for r in rows]
            if identity(predictions[display][subset]) != identity(predictions["LDA"][subset]):
                raise ValueError(f"Different ordered {subset} records")
    out.mkdir(parents=True)
    table, per_class = [], []
    for display, run in runs.items():
        for subset in ["train", "validation", "test"]:
            metrics = run["metrics"][subset]
            table.append({"model": display, "experiment_id": run["experiment_id"], "partition": subset,
                          **{key: metrics[key] for key in metric_keys}, "samples": metrics["n_samples"]})
            for name in metrics["class_names"]:
                cls = metrics["per_class"][name]
                i = metrics["class_names"].index(name)
                per_class.append({"model": display, "partition": subset, "class": name,
                                  "support": cls["support"], "correct": metrics["confusion_matrix"][i][i],
                                  "precision": cls["precision"], "recall": cls["recall"], "f1": cls["f1-score"]})
    write_csv(out / "metrics_comparison.csv", table)
    write_csv(out / "per_class_comparison.csv", per_class)
    report = ["# Executed image-model comparison", "",
              "Three completed models use the same 491-image dataset and 295/98/98 grouped split. Values were reconciled with saved predictions; this report performs no inference.", "",
              "## Previously observed test set — exploratory comparison", "",
              "| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        if row["partition"] == "test":
            report.append(f"| {row['model']} | {row['accuracy']:.2%} | {row['balanced_accuracy']:.2%} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | {row['roc_auc_ovr_macro']:.4f} |")
    new, old = runs["CNN–LSTM"]["metrics"]["test"], runs["CNN"]["metrics"]["test"]
    report += ["", f"CNN–LSTM minus CNN: accuracy {(new['accuracy']-old['accuracy'])*100:+.2f} percentage points; macro F1 {new['macro_f1']-old['macro_f1']:+.4f}; balanced accuracy {(new['balanced_accuracy']-old['balanced_accuracy'])*100:+.2f} percentage points.", "",
               "| Class | Test n | LDA correct | CNN correct | CNN–LSTM correct | CNN–LSTM recall |",
               "|---|---:|---:|---:|---:|---:|"]
    for i, name in enumerate(new["class_names"]):
        correct = [str(run["metrics"]["test"]["confusion_matrix"][i][i]) for run in runs.values()]
        report.append(f"| {name} | {int(new['per_class'][name]['support'])} | {' | '.join(correct)} | {new['per_class'][name]['recall']:.2%} |")
    report += ["", "## Development and fitting", "",
               "| Model | Train accuracy | Validation accuracy | Train macro F1 | Validation macro F1 |",
               "|---|---:|---:|---:|---:|"]
    for name, run in runs.items():
        a, b = run["metrics"]["train"], run["metrics"]["validation"]
        report.append(f"| {name} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} |")
    training = runs["CNN–LSTM"]["training"]
    report += ["", f"CNN–LSTM: {training['num_parameters']:,} parameters, {training['epochs_run']} epochs run, selected epoch {training['best_epoch']}, {training['training_seconds']:.2f} seconds training on {training['device']}.", "",
               "CNN–LSTM uses the CNN's three convolution blocks followed by coarse vertical pooling and 16 left-to-right spatial positions. A one-layer LSTM with 32 hidden units processes these positions; the classifier uses their mean output. This sequence is image geometry, not calibrated physiological time. No ECG waveform is digitized.", "",
               "The new checkpoint was selected by validation macro F1, with balanced accuracy then lower weighted loss as tie-breaks. The original CNN selected minimum weighted validation loss. Architecture, pooling/readout, gradient clipping and selection policy differ: this is a comparison of complete pipelines, not an isolated test of adding an LSTM.", "",
               "One seed and one configuration were run; there was no test-guided retuning. The test set had already been inspected during the baseline project, so this is exploratory evidence. Only six test images are MI. No significance or clinical-superiority claim is supported.", "",
               "Inputs retain the original low resolution, aspect distortion and known crop clipping. Labels are supplied folder labels, and grouping IDs are not verified patients. These limitations can affect every image model.", "",
               "PTB-XL is a separate proposed dataset transition. No PTB-XL results are included in this comparison; its patient folds, input representation and multilabel targets require a separately defined experiment.", ""]
    (out / "comparison.md").write_text("\n".join(report))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    metrics_to_plot = metric_keys[:4]
    positions = np.arange(4)
    for j, (name, run) in enumerate(runs.items()):
        m = run["metrics"]["test"]
        axes[0].bar(positions + (j-1)*0.25, [m[k] for k in metrics_to_plot], 0.25, label=name)
        axes[1].bar(positions + (j-1)*0.25, [m["per_class"][c]["f1-score"] for c in new["class_names"]], 0.25, label=name)
    axes[0].set_xticks(positions, ["Accuracy", "Balanced\naccuracy", "Macro F1", "Weighted F1"])
    axes[1].set_xticks(positions, new["class_names"])
    axes[0].set_title("Overall metrics")
    axes[1].set_title("Per-class F1")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Same previously observed test set (n=98): exploratory comparison")
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    write_json(out / "verification.json", {"saved_metrics_reconciled": True,
               "same_ordered_train_validation_test_records": True, "model_inference_performed": False,
               "dataset_version": reference["dataset_version"], "split_version": reference["split_version"],
               "source_artifact_sha256": hashes, "comparison_script_sha256": sha256_file(__file__)})
    print("Verified three-model comparison:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-lstm-id", default="cnn_lstm_image_v1_seed42")
    args = parser.parse_args()
    compare(args.cnn_lstm_id)
