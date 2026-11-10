"""Report a verified hybrid ViT and compare saved predictions to both baselines.

No training, checkpoint loading, model inference, threshold selection, image
access or rewriting of prior model artifacts. Passed verification files must
bind unchanged numerical artifacts. Both comparisons use the existing paired
patient-bootstrap implementation: 500 resamples, seed 2024, all six metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from src.compare_ptbxl_models import (
    CLASS_NAMES, PARTITIONS, METRICS, METRIC_LABELS, LIMITATION,
    _Inputs, _load_run, _align, _same, _require, _sha, _summary,
    _write_csv, _format, _run_timing, paired_patient_bootstrap,
)
from src.report_ptbxl_spatiotemporal import (
    _figures as _standard_figures, _save_figure, _runtime_summary, _runtime_text, _interval,
)
from src.verify_ptbxl_run import _verify_bootstrap, _verify_history

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "ptbxl_hybrid_vit_image_v1_seed42"
REFERENCES = (
    ("tcn_primary", "ptbxl_spatiotemporal", "ptbxl_spatiotemporal_image_v1_seed42", "TCN (primary)"),
    ("cnn_lstm_secondary", "ptbxl_cnn_lstm", "ptbxl_cnn_lstm_image_v1_seed42", "CNN–LSTM (secondary)"),
)
FIGURES = ("roc_curves", "precision_recall_curves", "per_label_metrics", "partition_summary", "training_history")
FIGURE_DATA = ("partition_metrics.csv", "per_label_metrics.csv", "training_history.csv")
COMPARISON_FILES = ("comparison.json", "comparison.md", "partition_metrics.csv", "metric_differences.csv",
                    "per_label_metrics.csv", "per_label_differences.csv", "comparison.png", "comparison.svg",
                    "per_label_differences.png", "per_label_differences.svg", "manifest.json")
LABEL_MEANINGS = {"NORM": "normal ECG", "MI": "myocardial infarction", "STTC": "ST/T changes",
                  "CD": "conduction disturbance", "HYP": "hypertrophy"}
LABEL_METRICS = ("roc_auc", "average_precision", "precision", "recall", "specificity", "f1", "tp", "fp", "tn", "fn")


def _load_bundle(root, run_id):
    """Reuse strict passed-verification hash, metric and row alignment checks."""
    root = Path(root).resolve()
    inputs = _Inputs()
    hybrid = _load_run(root, "ptbxl_hybrid_vit", run_id, inputs)
    _same("ptbxl_image_hybrid_vit", hybrid["config"]["model"], "hybrid model identity")
    _same("LeadImageHybridViT", hybrid["training"]["model"], "hybrid training identity")
    _same(23881, hybrid["training"]["num_parameters"], "frozen hybrid parameter count")
    _same(23881, hybrid["config"]["architecture"]["trainable_parameters"], "hybrid configured capacity")
    for key, value in (("resamples", 500), ("seed", 2024), ("confidence_level", .95)):
        _same(value, hybrid["config"]["uncertainty"][key], f"single-model uncertainty.{key}")
        _same(value, hybrid["config"]["paired_comparison"][key], f"paired uncertainty.{key}")
    references = {}
    for role, family, reference_id, title in REFERENCES:
        reference = _load_run(root, family, reference_id, inputs)
        _align(reference, hybrid)
        references[role] = reference
    _same(REFERENCES[0][2], hybrid["config"]["comparison_reference"], "primary comparison reference")
    _same(REFERENCES[1][2], hybrid["config"]["secondary_comparison_reference"], "secondary comparison reference")
    model_dir = root / hybrid["run"]["model_directory"]
    _, history = inputs.csv(model_dir / "history.csv")
    _verify_history(history, hybrid["training"], hybrid["partitions"]["validation"]["metrics"], hybrid["config"]["training"])
    ci = inputs.json(hybrid["out"] / "test_patient_bootstrap.json")
    patients = [row["patient_id"] for row in hybrid["partitions"]["test"]["rows"]]
    _verify_bootstrap(ci, hybrid["partitions"]["test"]["metrics"], patients, hybrid["config"]["uncertainty"])
    verification = inputs.json(hybrid["out"] / "verification.json")
    _require(verification["run_artifact_sha256"].get("test_patient_bootstrap.json") == _sha(hybrid["out"] / "test_patient_bootstrap.json"),
             "Saved verification must bind single-model bootstrap evidence")
    # Dataset prose may receive factual updates. It is not a numerical-input
    # identity requirement; bind its current bytes as report context only.
    card = root / "docs/ptbxl_image_dataset.md"
    inputs.text(card)
    for path in (Path(__file__), ROOT / "src/compare_ptbxl_models.py", ROOT / "src/report_ptbxl_spatiotemporal.py"):
        inputs.check_hash(path, _sha(path))
    return {"root": root, "hybrid": hybrid, "references": references, "inputs": inputs,
            "history": history, "ci": ci, "model_dir": model_dir}


def _comparison_results(bundle):
    hybrid = bundle["hybrid"]
    result = {"status": "completed", "run_id": hybrid["run_id"], "class_names": CLASS_NAMES,
              "threshold": .5, "primary_reference": REFERENCES[0][2], "secondary_reference": REFERENCES[1][2],
              "primary_metric": "macro_roc_auc", "difference_direction": "hybrid_minus_reference", "comparisons": {},
              "model_inference_performed": False, "model_training_performed": False,
              "threshold_tuning_performed": False, "paired_bootstrap_performed": True,
              "multiplicity_adjustment": "none; secondary reference and nonprimary metrics are descriptive",
              "limitations": LIMITATION}
    after = hybrid["partitions"]["test"]
    for role, _, _, title in REFERENCES:
        reference = bundle["references"][role]
        before = reference["partitions"]["test"]
        boot = paired_patient_bootstrap(before["truth"], before["probabilities"], after["probabilities"],
                                        [row["patient_id"] for row in before["rows"]], CLASS_NAMES,
                                        n_resamples=500, seed=2024)
        pair = _summary(reference, hybrid, boot)
        pair.update(reference_title=title, role=role)
        pair["models"]["old"]["actual_device"] = reference["training"].get("device", "unrecorded")
        pair["models"]["new"]["actual_device"] = hybrid["training"].get("device", "unrecorded")
        result["comparisons"][role] = pair
    return result


def _comparison_tables(result):
    partitions, differences, labels, label_differences = [], [], [], []
    for role, pair in result["comparisons"].items():
        for split, part in pair["partitions"].items():
            for model_role in ("old", "new"):
                model = pair["models"][model_role]
                partitions.append({"comparison": role, "partition": split, "model_role": "reference" if model_role == "old" else "hybrid",
                                   "run_id": model["run_id"], "n_records": part["n_records"], "n_patients": part["n_unique_patients"],
                                   **{key: part[model_role][key] for key in (*METRICS, "weighted_f1", "binary_cross_entropy")},
                                   "num_parameters": model["num_parameters"], "actual_device": model["actual_device"],
                                   "best_epoch": model["best_epoch"], "epochs_run": model["epochs_run"],
                                   **{key: model[key] for key in ("training_recorded_timer_seconds", "run_recorded_timer_seconds", "run_utc_wall_span_seconds")}})
                for label in CLASS_NAMES:
                    labels.append({"comparison": role, "partition": split, "model_role": "reference" if model_role == "old" else "hybrid",
                                   "run_id": model["run_id"], "label": label, **part[model_role]["per_label"][label]})
            for metric in METRICS:
                ci = pair["paired_test_bootstrap"]["metrics"][metric] if split == "test" else {}
                differences.append({"comparison": role, "partition": split, "metric": metric,
                                    "reference": part["old"][metric], "hybrid": part["new"][metric],
                                    "hybrid_minus_reference": part["differences"][metric],
                                    **{key: ci.get(key) for key in ("ci_lower", "ci_upper", "defined_resamples", "undefined_resamples", "interpretation")}})
            for label in CLASS_NAMES:
                before, after = (part[key]["per_label"][label] for key in ("old", "new"))
                row = {"comparison": role, "partition": split, "label": label,
                       "support": after["support"], "negative_support": after["negative_support"]}
                for metric in LABEL_METRICS:
                    row.update({f"reference_{metric}": before[metric], f"hybrid_{metric}": after[metric],
                                f"delta_{metric}": None if before[metric] is None or after[metric] is None else after[metric] - before[metric]})
                label_differences.append(row)
    return partitions, differences, labels, label_differences


def _finding(pair):
    item = pair["paired_test_bootstrap"]["metrics"]["macro_roc_auc"]
    phrase = {"includes_zero_no_resolved_difference": "The interval includes or touches zero; a test macro ROC-AUC difference is unresolved.",
              "above_zero_new_higher_conditional": "The interval lies above zero; hybrid test macro ROC-AUC is higher conditional on these fitted models and this cohort.",
              "below_zero_old_higher_conditional": "The interval lies below zero; hybrid test macro ROC-AUC is lower conditional on these fitted models and this cohort.",
              "not_estimable": "The paired interval is not estimable."}[item["interpretation"]]
    return (f"Hybrid − {pair['reference_title']}: **{_format(item['difference'], True)}** macro ROC-AUC; "
            f"95% paired interval **[{_format(item['ci_lower'], True)}, {_format(item['ci_upper'], True)}]**. {phrase}")


def _comparison_markdown(result, output):
    lines = ["# PTB-XL hybrid ViT: paired exploratory comparisons", "", _finding(result["comparisons"]["tcn_primary"]), "",
             "TCN is the prespecified primary reference; CNN–LSTM is secondary. Both comparisons preserve every metric, including deterioration. "
             "All differences below are hybrid minus reference. No checkpoint or threshold is chosen from these test results.", "", LIMITATION, "",
             "All train/validation/test ECG IDs, patient IDs, image paths, five truth columns and row order match; source manifests, cache hashes, "
             "preprocessing, training-only normalization, folds, class order, selection rule and threshold match. Passed verification records bind unchanged saved artifacts.", "",
             "Each comparison uses 500 patient-cluster draws, seed 2024, with all records and full multiplicity for every sampled patient. "
             "The same group order and seed also yield the same draw sequence across the two comparisons. Percentile limits use linear 2.5th/97.5th quantiles. "
             "Undefined paired replicates are counted and omitted; no multiplicity adjustment is applied.", ""]
    for role, pair in result["comparisons"].items():
        lines += [f"## {pair['reference_title']}", "", _finding(pair), "",
                  f"Validation ordering: **{pair['validation_ranking']['ordering'].replace('new', 'hybrid').replace('old', 'reference').replace('_', ' ')}**, "
                  "using macro ROC-AUC, then macro AP, then lower probability BCE. This is descriptive ordering of separately selected checkpoints, independent of the test comparison.", "",
                  "| Test metric | Reference | Hybrid | Hybrid − reference | 95% paired interval | Defined / 500 |", "|---|---:|---:|---:|---|---:|"]
        for metric in METRICS:
            item = pair["paired_test_bootstrap"]["metrics"][metric]
            lines.append(f"| {METRIC_LABELS[metric]} | {_format(item['old_point_estimate'])} | {_format(item['new_point_estimate'])} | "
                         f"{_format(item['difference'], True)} | [{_format(item['ci_lower'], True)}, {_format(item['ci_upper'], True)}] | {item['defined_resamples']} |")
        lines += ["", "| Test label | Δ ROC-AUC | Δ AP | Δ Recall | Δ Specificity | Reference TP / FN | Hybrid TP / FN | Δ TP / FN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label in CLASS_NAMES:
            a, b = (pair["partitions"]["test"][key]["per_label"][label] for key in ("old", "new"))
            delta = [None if a[k] is None or b[k] is None else b[k] - a[k] for k in ("roc_auc", "average_precision", "recall", "specificity")]
            lines.append(f"| {label} | " + " | ".join(_format(v, True) for v in delta)
                         + f" | {a['tp']} / {a['fn']} | {b['tp']} / {b['fn']} | {b['tp']-a['tp']:+d} / {b['fn']-a['fn']:+d} |")
        lower_recall = [label for label in CLASS_NAMES if pair["partitions"]["test"]["new"]["per_label"][label]["recall"] < pair["partitions"]["test"]["old"]["per_label"][label]["recall"]]
        lines += ["", "Labels with lower hybrid recall at 0.5: **" + (", ".join(lower_recall) if lower_recall else "none") + "**. "
                  "These are descriptive per-label differences without per-label confidence intervals. Positive ΔFN means more missed positive labels; positive ΔTP means more detected positive labels.", ""]
    lines += ["## Capacity, device and timing", "", "| Model | Parameters | Actual device | Selected / completed epochs | Training timer (s) | Run timer (s) | UTC wall span (s) |", "|---|---:|---|---:|---:|---:|---:|"]
    models = [(pair["reference_title"], pair["models"]["old"]) for pair in result["comparisons"].values()]
    models.append(("Hybrid ViT", result["comparisons"]["tcn_primary"]["models"]["new"]))
    for title, model in models:
        lines.append(f"| {title} | {model['num_parameters']:,} | {model['actual_device']} | {model['best_epoch']} / {model['epochs_run']} | "
                     f"{model['training_recorded_timer_seconds']:.1f} | {model['run_recorded_timer_seconds']:.1f} | {model['run_utc_wall_span_seconds']:.1f} |")
    lines += ["", "The encoder family, inputs, seed and optimizer/epoch cap match, but heads, parameter counts, dropout placement, learning trajectories and stopping epochs differ. "
              "This is a single-seed pipeline comparison, not a capacity-controlled isolation of attention. Recorded elapsed timers include overhead and waits; they are not pure CPU time. "
              "CPU thread counts, backend selection, thermals, concurrent load and sleep/suspend can affect timings. UTC wall spans and performance-counter durations are reported separately; "
              "these runs do not constitute a controlled speed or efficiency benchmark, even when all actual devices are CPU.", "",
              "Label meanings: " + "; ".join(f"{key} = {value}" for key, value in LABEL_MEANINGS.items()) + ". Labels overlap. "
              "Label accuracy counts individual binary decisions; exact-match accuracy requires all five labels correct for an ECG.", "",
              f"![All six scores and paired differences](<{output / 'comparison.png'}>)", "",
              f"![All five per-label differences](<{output / 'per_label_differences.png'}>)", "",
              "CSV evidence includes every partition, model, all six summary metrics, and per-label AUC/AP/precision/recall/specificity/F1/confusion counts and deltas.", ""]
    return "\n".join(lines)


def _comparison_figures(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14.5, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.25))
    top = fig.add_subplot(grid[0, :])
    primary, secondary = (result["comparisons"][key] for key in ("tcn_primary", "cnn_lstm_secondary"))
    bars = ((secondary["partitions"]["test"]["old"], "CNN–LSTM", "#577895"),
            (primary["partitions"]["test"]["old"], "TCN", "#B48036"),
            (primary["partitions"]["test"]["new"], "Hybrid ViT", "#237C72"))
    x = np.arange(len(METRICS))
    for index, (values, title, color) in enumerate(bars):
        heights = [np.nan if values[key] is None else values[key] for key in METRICS]
        top.bar(x + (index - 1) * .24, heights, width=.23, label=title, color=color)
    top.set(xticks=x, xticklabels=["Macro ROC-AUC\n(primary)", "Macro AP", "Macro F1", "Micro F1", "Exact-match\naccuracy", "Label\naccuracy"],
            ylim=(0, 1.04), ylabel="Saved test score", title="All six test metrics | fixed threshold 0.5 for decisions")
    top.legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3, frameon=False)
    top.grid(axis="y", alpha=.15)
    for column, pair in enumerate((primary, secondary)):
        ax = fig.add_subplot(grid[1, column])
        ax.axvline(0, color=".5", linestyle="--", linewidth=1)
        for index, metric in enumerate(METRICS):
            item = pair["paired_test_bootstrap"]["metrics"][metric]
            color = "#237C72" if item["difference"] is not None and item["difference"] >= 0 else "#B34D48"
            if item["ci_lower"] is not None:
                ax.hlines(index, item["ci_lower"], item["ci_upper"], color=color, linewidth=2)
                ax.vlines([item["ci_lower"], item["ci_upper"]], index - .08, index + .08, color=color, linewidth=1)
            if item["difference"] is not None:
                ax.plot(item["difference"], index, "o", color=color)
            else:
                ax.text(0, index, " undefined", va="center")
        ax.set(yticks=np.arange(6), yticklabels=[METRIC_LABELS[key] for key in METRICS],
               xlabel="Hybrid − reference score (95% paired interval)", title=f"Hybrid vs {pair['reference_title']}")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        ax.margins(x=.15)
    fig.suptitle("Exploratory PTB-XL comparison | same patients, labels and cached inputs", fontsize=15)
    fig.supxlabel("500 paired patient-cluster resamples, seed 2024. Conditional on fitted models; no seed-variation or multiplicity correction.", fontsize=10)
    _save_figure(fig, directory, "comparison")
    plt.close(fig)
    measures = ("roc_auc", "average_precision", "recall", "specificity")
    matrices = []
    for pair in (primary, secondary):
        matrices.append(np.asarray([[np.nan if pair["partitions"]["test"]["new"]["per_label"][label][key] is None or pair["partitions"]["test"]["old"]["per_label"][label][key] is None
                                     else pair["partitions"]["test"]["new"]["per_label"][label][key] - pair["partitions"]["test"]["old"]["per_label"][label][key]
                                     for key in measures] for label in CLASS_NAMES]))
    limit = max(.01, max(float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0 for matrix in matrices))
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.2), constrained_layout=True)
    for ax, matrix, pair in zip(axes, matrices, (primary, secondary)):
        image = ax.imshow(matrix, cmap="BrBG", vmin=-limit, vmax=limit, aspect="auto")
        for row in range(5):
            for col in range(4):
                value = matrix[row, col]
                ax.text(col, row, "undefined" if not np.isfinite(value) else f"{value:+.3f}", ha="center", va="center",
                        color="white" if np.isfinite(value) and abs(value) > .65 * limit else "#202A35")
        ax.set(xticks=range(4), xticklabels=["ROC-AUC", "Average\nprecision", "Recall", "Specificity"],
               yticks=range(5), yticklabels=CLASS_NAMES, title=f"Hybrid − {pair['reference_title']}")
    fig.colorbar(image, ax=axes, shrink=.8, label="Score difference; positive = higher hybrid score")
    fig.suptitle("Every diagnostic label | descriptive test differences", fontsize=14)
    fig.supxlabel("Recall/specificity use probability ≥ 0.5. Per-label intervals are not estimated; negative values are retained.", fontsize=10)
    _save_figure(fig, directory, "per_label_differences")
    plt.close(fig)


def _report_data(bundle):
    hybrid = bundle["hybrid"]
    return {"metrics": {split: hybrid["partitions"][split]["metrics"] for split in PARTITIONS},
            "predictions": {split: (hybrid["partitions"][split]["truth"], hybrid["partitions"][split]["probabilities"]) for split in PARTITIONS},
            "ci": bundle["ci"]}


def _training_figure(bundle, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history, best = bundle["history"], bundle["hybrid"]["training"]["best_epoch"]
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1), constrained_layout=True)
    for key, title, color in (("train_online_bce", "Train: online pre-update BCE", "#577895"),
                              ("val_binary_cross_entropy", "Validation: epoch-end probability BCE", "#237C72")):
        axes[0].plot(epochs, [float(row[key]) for row in history], label=title, color=color)
    axes[0].set(ylabel="Unweighted BCE", title="Training and validation loss")
    axes[0].legend(fontsize=8)
    for key, title, color in (("val_macro_roc_auc", "Macro ROC-AUC", "#237C72"), ("val_macro_average_precision", "Macro AP", "#B48036")):
        axes[1].plot(epochs, [float(row[key]) for row in history], label=title, color=color)
    axes[1].set(ylabel="Validation score", title="Validation checkpoint criteria", ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    axes[2].plot(epochs, [float(row["learning_rate"]) for row in history], color="#59617D")
    axes[2].set(ylabel="Learning rate", title="Prespecified cosine schedule")
    for ax in axes:
        ax.axvline(best, linestyle="--", color=".45", label=f"Selected epoch {best}")
        ax.set(xlabel="Epoch", xlim=(.5, max(epochs) + .5))
        ax.grid(alpha=.15)
    fig.suptitle(f"Hybrid ViT training history | selected epoch {best} | validation-only selection", fontsize=14)
    fig.supxlabel("Online training loss includes dropout and changing weights; it is not a full epoch-end training evaluation.", fontsize=9)
    _save_figure(fig, directory, "training_history")
    plt.close(fig)


def _report_markdown(bundle, comparison, figures, comparison_dir):
    hybrid = bundle["hybrid"]
    cfg, training, run, ci = hybrid["config"], hybrid["training"], hybrid["run"], bundle["ci"]
    metrics = {split: hybrid["partitions"][split]["metrics"] for split in PARTITIONS}
    test = metrics["test"]
    lines = ["# PTB-XL image hybrid ViT report", "",
             f"**Completed and verified: `{hybrid['run_id']}`.** Test macro ROC-AUC is **{_format(test['macro_roc_auc'])}** "
             f"(saved 95% patient-bootstrap interval **{_interval(ci, 'macro_roc_auc')}**). Selected checkpoint: epoch **{training['best_epoch']} of {training['epochs_run']}**.", "",
             _finding(comparison["comparisons"]["tcn_primary"]), "",
             "This is an exploratory evaluation on a test cohort already examined for the CNN–LSTM and TCN. "
             "Reporting reads verified saved predictions and performs two new paired bootstrap comparisons; it performs no inference, refitting, threshold tuning or calibration.", "",
             "## Cohort, representation and architecture", "",
             "The source is [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) "
             "([version DOI](https://doi.org/10.13026/kfzx-aw45); [original publication](https://doi.org/10.1038/s41597-020-0495-6)), "
             "rendered into a single clean synthetic page style from native 500 Hz, ten-second, 12-lead signals. "
             "Of 21,799 rendered records, 21,388 map to at least one diagnostic superclass; 411 unmapped records are excluded rather than assigned all-negative or normal labels. "
             "Official patient folds 1–8 / 9 / 10 define train / validation / test. "
             "All present diagnostic SCP keys, including zero recorded likelihood, contribute to the superclass union.", "",
             "Targets overlap: " + "; ".join(f"**{key}** = {value}" for key, value in LABEL_MEANINGS.items()) + ".", "",
             "| Partition | ECGs | Patients |", "|---|---:|---:|"]
    for split in PARTITIONS:
        part = hybrid["partitions"][split]
        lines.append(f"| {split.capitalize()} | {len(part['rows']):,} | {len({row['patient_id'] for row in part['rows']}):,} |")
    lines += ["", "The unchanged **12 × 120 × 1120** input retains complete aligned lead rows and calibration pulses. "
              "Grayscale inversion and isotropic 2×2 maximum-ink pooling precede global training-only standardization. IDs, labels and numeric metadata are not predictor features.", "",
              f"The **{training['num_parameters']:,}-parameter** model uses the same depthwise/pointwise CNN encoder family as the two references, independently trained from scratch. "
              "Its **70 image-feature tokens × 96 features** are projected to width 32, receive learned positional embeddings, and pass through two independently initialized "
              "pre-norm Transformer encoder layers with four attention heads, feedforward width 64, GELU and dropout. Final LayerNorm, mean token pooling and a five-logit head "
              "produce independent sigmoid probabilities. It is a CNN-feature hybrid ViT, not a vanilla raw-patch ViT or a recovered-waveform model. "
              "The global attention, encoder context and pooling make this an offline classifier; image tokens are not native waveform samples or a causal streaming signal.", "",
              f"Seed {cfg['seed']}; unweighted BCE with logits; AdamW; batch {cfg['training']['batch_size']}; learning rate {cfg['training']['learning_rate']:g}; "
              f"weight decay {cfg['training']['weight_decay']:g}; dropout {cfg['training']['dropout']:g}; gradient clipping {cfg['training']['gradient_clip_norm']:g}; "
              f"cosine schedule to {cfg['training']['eta_min']:g}; cap {cfg['training']['max_epochs']} epochs, patience {cfg['training']['patience']}. "
              f"Actual device: **{training['device']}**; requested device: {cfg['training']['device']}; configured CPU threads: {cfg['training']['num_threads']}. "
              "No augmentation, pretraining, class weighting or configuration search was used. The fixed decision threshold is probability ≥ 0.5.", "",
              "Checkpoint selection lexicographically maximizes validation macro ROC-AUC, then macro average precision, then minimizes probability BCE; earliest exact ties are retained. "
              "The verified bookkeeping records one final test inference after checkpoint/configuration freezing and an exact training-only checkpoint replay. "
              "Artifact verification cannot rule out unrecorded activity.", "",
              "## Selected-checkpoint metrics", "", "| Metric | Train | Validation | Test | Test 95% patient interval |", "|---|---:|---:|---:|---|"]
    for key in METRICS:
        lines.append(f"| {METRIC_LABELS[key]} | " + " | ".join(_format(metrics[split][key]) for split in PARTITIONS) + f" | {_interval(ci, key)} |")
    for key, title in (("weighted_f1", "Positive-support weighted F1"), ("binary_cross_entropy", "Probability BCE")):
        lines.append(f"| {title} | " + " | ".join(_format(metrics[split][key]) for split in PARTITIONS) + " | Not bootstrapped |")
    lines += ["", "Macro metrics weight labels equally; micro F1 pools binary decisions. Label accuracy measures correct record-label decisions, while exact-match accuracy requires all five labels correct for each ECG. "
              "Neither is a four-class single-label accuracy, and neither supports a direct ranking against the separate Mendeley experiments.", "",
              "| Test label | Positives | ROC-AUC | AP | Precision | Recall | Specificity | F1 | TP | FN | FP | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in CLASS_NAMES:
        per = test["per_label"][label]
        lines.append(f"| {label} | {per['support']} | " + " | ".join(_format(per[key]) for key in ("roc_auc", "average_precision", "precision", "recall", "specificity", "f1"))
                     + " | " + " | ".join(str(per[key]) for key in ("tp", "fn", "fp", "tn")) + " |")
    lowest = min(CLASS_NAMES, key=lambda label: test["per_label"][label]["recall"])
    low = test["per_label"][lowest]
    lines += ["", f"Lowest test recall at 0.5: **{lowest}**, {low['tp']}/{low['support']} positives detected ({_format(low['recall'])}), "
              f"with {low['fn']} false negatives and {low['fp']} false positives. This describes the saved result; it does not trigger threshold tuning.", "",
              f"Train minus validation gaps: macro ROC-AUC **{metrics['train']['macro_roc_auc']-metrics['validation']['macro_roc_auc']:+.4f}**; "
              f"macro F1 **{metrics['train']['macro_f1']-metrics['validation']['macro_f1']:+.4f}**. These gaps do not isolate an overfitting mechanism.", "",
              "## Comparisons and uncertainty", "", _finding(comparison["comparisons"]["cnn_lstm_secondary"]), "",
              f"[Full paired comparison with TCN and CNN–LSTM](<{comparison_dir / 'comparison.md'}>) preserves all six metrics, "
              "all five label deltas and both improvements and deteriorations. Capacity and dropout topology differ, so this is not an attention-only ablation.", "",
              f"Single-model intervals are the saved {ci['n_resamples']} patient-cluster percentile resamples, seed {ci['seed']}; this report checks their structure and point estimates rather than resampling them. "
              "The two paired comparisons are newly recomputed from saved predictions using 500 shared-patient draws each, seed 2024. "
              "All intervals are conditional on these fitted models and exclude training-seed variability, adaptive model selection, clinical annotation uncertainty and external-domain shift. "
              "There is no multiplicity adjustment. Undefined ranking metrics/resamples remain explicitly undefined; no defined-label-only score replaces the primary metric.", "",
              "Clean synthetic pages do not establish performance on clinical scans, photographs, other layouts or prospective populations. Gain reduction, rasterization and pooling lose fine detail. "
              "The retained calibration pulses do not constitute validated physical-signal reconstruction. Further development cannot call this repeatedly observed test cohort untouched.", "",
              "## Training history and recorded runtime", "",
              "The plotted training loss is online, pre-update batch BCE with dropout and changing weights, not a full epoch-end training evaluation. "
              "The table's training metrics use the selected checkpoint with dropout disabled; validation history evaluates the complete validation split after each epoch.", "",
              _runtime_text(_runtime_summary(run, training)), "",
              "Device/backend differences, thread use, thermal conditions, concurrent work, sleep/suspend and differing stopping epochs prevent a controlled CPU-speed or efficiency conclusion.", "",
              "## Figures and provenance", ""]
    for name in FIGURES:
        lines += [f"[{name.replace('_', ' ').capitalize()} PNG](<{figures / (name+'.png')}>) · [SVG](<{figures / (name+'.svg')}>)", ""]
    lines += [f"[Verification](<{hybrid['out'] / 'verification.json'}>) · [Configuration](<{hybrid['out'] / 'config.json'}>) · "
              f"[Saved predictions and metrics](<{hybrid['out'] / 'run.json'}>) · [Dataset card](<{bundle['root'] / 'docs/ptbxl_image_dataset.md'}>)", "",
              f"Checkpoint SHA-256: `{run['checkpoint_sha256']}`. Configuration SHA-256: `{hybrid['provenance']['configuration_sha256']}`.", "",
              "Report and comparison manifests bind every read numerical artifact and every generated output. No old result or model artifact is rewritten.", ""]
    return "\n".join(lines)


def generate_report(run_id=DEFAULT_RUN, *, root=ROOT, overwrite=False):
    root = Path(root).resolve()
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id), "Invalid run identifier")
    figures = root / "results/ptbxl_hybrid_vit" / run_id / "figures"
    report = root / "docs/ptbxl_hybrid_vit_report.md"
    comparison_dir = root / "results/comparisons" / run_id
    figure_names = {f"{name}.{ext}" for name in FIGURES for ext in ("png", "svg")} | set(FIGURE_DATA) | {"report_manifest.json"}
    for directory, allowed in ((figures, figure_names), (comparison_dir, set(COMPARISON_FILES))):
        _require(not directory.is_symlink(), "Report directory must not be a symlink")
        if directory.exists():
            _require(directory.is_dir(), "Report destination is not a directory")
            _require({path.name for path in directory.iterdir()} <= allowed, "Output directory contains unrelated artifacts")
    destinations = [report] + [figures / name for name in figure_names] + [comparison_dir / name for name in COMPARISON_FILES]
    if any(path.is_symlink() for path in destinations):
        raise ValueError("Refusing a symlink reporting destination")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("Report outputs already exist; use --overwrite only for this command's own reports")
    bundle = _load_bundle(root, run_id)
    result = _comparison_results(bundle)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["source_artifact_sha256"] = dict(bundle["inputs"].hashes)
    with tempfile.TemporaryDirectory(prefix="ptbxl-hybrid-report-") as temporary:
        stage = Path(temporary)
        staged_figures, staged_comparison = stage / "figures", stage / "comparison"
        staged_figures.mkdir(); staged_comparison.mkdir()
        _standard_figures(_report_data(bundle), staged_figures)
        _training_figure(bundle, staged_figures)
        hybrid = bundle["hybrid"]
        _write_csv(staged_figures / "training_history.csv", bundle["history"])
        _write_csv(staged_figures / "partition_metrics.csv", [{"partition": split,
                   **{key: hybrid["partitions"][split]["metrics"][key] for key in (*METRICS, "weighted_f1", "binary_cross_entropy")}}
                  for split in PARTITIONS])
        _write_csv(staged_figures / "per_label_metrics.csv", [{"partition": split, "label": label,
                   **hybrid["partitions"][split]["metrics"]["per_label"][label]} for split in PARTITIONS for label in CLASS_NAMES])
        for name, rows in zip(("partition_metrics.csv", "metric_differences.csv", "per_label_metrics.csv", "per_label_differences.csv"), _comparison_tables(result)):
            _write_csv(staged_comparison / name, rows)
        (staged_comparison / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (staged_comparison / "comparison.md").write_text(_comparison_markdown(result, comparison_dir))
        _comparison_figures(result, staged_comparison)
        staged_report = stage / report.name
        staged_report.write_text(_report_markdown(bundle, result, figures, comparison_dir))
        bundle["inputs"].unchanged()
        mappings = {report: staged_report}
        mappings.update({figures / path.name: path for path in staged_figures.iterdir()})
        mappings.update({comparison_dir / path.name: path for path in staged_comparison.iterdir()})
        manifest = {"created_utc": result["created_utc"], "run_id": run_id, "status": "completed",
                    "model_inference_performed": False, "model_training_performed": False, "threshold_tuning_performed": False,
                    "paired_bootstrap_performed": True, "paired_resamples_per_comparison": 500, "paired_seed": 2024,
                    "single_model_bootstrap_resampled": False, "numerical_inputs_bound_to_passed_verifications": True,
                    "input_sha256": bundle["inputs"].hashes,
                    "output_sha256": {str(destination): _sha(source) for destination, source in mappings.items()}}
        for destination in (figures / "report_manifest.json", comparison_dir / "manifest.json"):
            source = stage / ("report_manifest.json" if destination.parent == figures else "comparison_manifest.json")
            source.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
            mappings[destination] = source
        if not overwrite and any(destination.exists() or destination.is_symlink() for destination in mappings):
            raise FileExistsError("Reporting destination appeared while output was staged")
        for destination, source in mappings.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if overwrite:
                _require(not destination.is_symlink(), "Reporting destination became a symlink")
                destination.write_bytes(source.read_bytes())
            else:
                with destination.open("xb") as handle:
                    handle.write(source.read_bytes())
    return {"run_id": run_id, "report": str(report), "figure_directory": str(figures), "comparison_directory": str(comparison_dir),
            "new_model_inference_performed": False, "paired_comparisons": 2, "paired_resamples_each": 500,
            "figure_files": 14, "csv_files": 7}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this command's own report artifacts")
    args = parser.parse_args()
    print(json.dumps(generate_report(args.run_id, overwrite=args.overwrite), indent=2))
