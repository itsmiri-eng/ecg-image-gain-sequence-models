"""Report one completed, verified PTB-XL image run from saved predictions only.

No model/checkpoint loading, image access, inference, fitting, threshold search,
or bootstrap resampling occurs here. Existing reporting outputs are protected
unless --overwrite explicitly authorizes replacing this command's own files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.multilabel import evaluate_multilabel


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
PARTITIONS = ("train", "validation", "test")
FIGURE_NAMES = ("roc_curves", "precision_recall_curves", "per_label_metrics", "partition_summary")
SUMMARY_METRICS = ("macro_roc_auc", "macro_average_precision", "macro_f1", "micro_f1", "label_accuracy", "subset_accuracy")
PARTITION_COLORS = {"train": "#477B9C", "validation": "#E0A23D", "test": "#3B9278"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{name} must be finite")
    return float(value)


def _runtime_summary(run, training):
    """Keep performance-counter durations separate from the UTC wall span."""
    runner = _finite(run["total_runtime_seconds"], "runner elapsed timer")
    trainer = _finite(training["runtime_seconds"], "training elapsed timer")
    _require(runner >= 0 and trainer >= 0, "Recorded elapsed timers must be nonnegative")
    wall = None
    if run.get("started_utc") is not None and run.get("completed_utc") is not None:
        timestamps = []
        for key in ("started_utc", "completed_utc"):
            try:
                parsed = datetime.fromisoformat(run[key].replace("Z", "+00:00"))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid run timestamp: {key}") from error
            _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"Run timestamp lacks timezone: {key}")
            timestamps.append(parsed.astimezone(timezone.utc))
        wall = (timestamps[1] - timestamps[0]).total_seconds()
        _require(wall >= 0, "Run completion precedes its start")
    return {"runner_elapsed_timer_seconds": runner, "training_elapsed_timer_seconds": trainer,
            "utc_wall_elapsed_seconds": wall, "wall_time_basis": "completed_utc minus started_utc when both are available",
            "timer_basis": "Recorded perf_counter elapsed durations; not pure CPU execution time",
            "suspend_caveat": "Elapsed timers may exclude system-suspend time on this host; UTC wall span includes it and is subject to clock corrections. The difference alone cannot quantify time asleep."}


def _runtime_text(timing):
    text = (f"Recorded runner elapsed-timer duration: **{timing['runner_elapsed_timer_seconds'] / 60:.2f} minutes**; "
            f"recorded training-module elapsed-timer duration: **{timing['training_elapsed_timer_seconds'] / 60:.2f} minutes**. ")
    if timing["utc_wall_elapsed_seconds"] is None:
        text += "The run's UTC wall span is unavailable because both start and completion timestamps were not recorded. "
    else:
        text += (f"The run's UTC wall span was **{timing['utc_wall_elapsed_seconds'] / 60:.2f} minutes**, "
                 "computed from completion minus start timestamps. ")
    return (text + "The recorded performance-counter timers include execution and waits; they are not pure CPU execution time "
            "and may exclude system-suspend time on this host. The UTC span includes sleep/suspend gaps, subject to system-clock corrections. "
            "A timer-versus-wall difference alone cannot quantify time asleep. These durations describe this execution and are not a controlled speed benchmark.")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Inputs:
    def __init__(self):
        self.hashes = {}

    def text(self, path):
        data = path.read_bytes()
        self.hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8")

    def json(self, path):
        def invalid_constant(value):
            raise ValueError(f"Nonfinite JSON constant {value} in {path}")
        return json.loads(self.text(path), parse_constant=invalid_constant)

    def csv(self, path):
        return list(csv.DictReader(io.StringIO(self.text(path))))

    def unchanged(self):
        _require(all(_sha(Path(path)) == digest for path, digest in self.hashes.items()), "Reporting inputs changed during generation")


def _close(actual, expected, name):
    if actual is None or expected is None:
        _require(actual is None and expected is None, f"Undefined metric mismatch: {name}")
    else:
        _require(abs(_finite(actual, name) - _finite(expected, name)) <= 1e-10, f"Saved/recomputed metric mismatch: {name}")


def _prediction_arrays(rows, split):
    _require(bool(rows), f"No {split} predictions")
    ids = [row["ecg_id"] for row in rows]
    _require(len(ids) == len(set(ids)), f"Repeated ECG identifier in {split} predictions")
    _require(all(row["patient_id"].strip() for row in rows), f"Missing patient ID in {split} predictions")
    truth = np.asarray([[float(row[f"truth_{name}"]) for name in CLASS_NAMES] for row in rows], dtype=np.float64)
    p = np.asarray([[float(row[f"probability_{name}"]) for name in CLASS_NAMES] for row in rows], dtype=np.float64)
    metrics = evaluate_multilabel(truth, p, CLASS_NAMES, threshold=.5)
    for index, name in enumerate(CLASS_NAMES):
        saved = np.asarray([float(row[f"prediction_{name}"]) for row in rows])
        _require(np.array_equal(saved, (p[:, index] >= .5).astype(int)), f"Saved threshold decisions differ: {split}/{name}")
    return truth.astype(np.int8), p, metrics


def _load_inputs(root, run_id):
    _require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id)), "Invalid run ID")
    inputs = _Inputs()
    out = root / "results/ptbxl_spatiotemporal" / run_id
    run = inputs.json(out / "run.json")
    _require(run.get("status") == "completed" and run.get("experiment_id") == run_id, "A matching completed run is required")
    verification = inputs.json(out / "verification.json")
    _require(verification.get("status") == "passed", "Run verification must pass before reporting")
    _require(verification.get("run_id") == run_id, "Verification belongs to a different run")
    cfg = inputs.json(out / "config.json")
    training = inputs.json(out / "training_metadata.json")
    provenance = inputs.json(out / "provenance.json")
    ci = inputs.json(out / "test_patient_bootstrap.json")
    frozen = inputs.json(out / "selection_frozen.json")
    replay = inputs.json(out / "checkpoint_replay.json")
    test_state = inputs.json(out / "test_inference_state.json")
    preservation = inputs.json(out / "preservation.json")
    _require(cfg["model"] == "ptbxl_image_spatiotemporal" and training["model"] == "LeadImageSpatiotemporal", "Unsupported model identity")
    _require(cfg["class_names"] == list(CLASS_NAMES) and cfg["input_shape"] == [12, 120, 1120], "Unsupported target order or input shape")
    _require(cfg["threshold"] == .5 and run["threshold"] == .5, "This report requires the frozen 0.5 threshold")
    _require(training["num_parameters"] == 17161 and training["config"]["hidden_channels"] == 32 and training["config"]["dilations"] == [1, 2] and training["config"]["kernel_size"] == 3, "Unsupported model architecture")
    _require(training["config"] == cfg["training"], "Training configuration differs from run configuration")
    _require(training["normalization"]["fit_partition"] == "train", "Normalization must be training-only")
    _require(run["test_inference_calls"] == 1 and test_state["status"] == "completed" and test_state["inference_calls"] == 1,
             "Expected one completed final test inference")
    _require(replay["passed"] is True and replay["split"] == "train" and replay["maximum_absolute_probability_difference"] == 0,
             "Training-only checkpoint replay must pass exactly")
    _require(not preservation["changed_artifacts"], "Prior artifact preservation failed")
    _require(run["checkpoint_sha256"] == frozen["checkpoint_sha256"] == replay["checkpoint_sha256"] == test_state["checkpoint_sha256"],
             "Checkpoint identity differs across saved artifacts")
    _require(provenance["configuration_sha256"] == frozen["configuration_sha256"], "Frozen configuration identity differs")
    _require(verification.get("checkpoint_sha256") == run["checkpoint_sha256"]
             and verification.get("configuration_sha256") == provenance["configuration_sha256"], "Verification identity differs from the frozen run")
    model_dir = (root / run["model_directory"]).resolve()
    _require(model_dir == (root / "models/ptbxl_spatiotemporal" / run_id).resolve(), "Unexpected model-directory binding")
    history = inputs.csv(model_dir / "history.csv")
    inputs.text(model_dir / "architecture.txt")
    _require(inputs.json(model_dir / "config.json") == cfg["training"], "Saved trainer configuration differs")
    _require(len(history) == training["epochs_run"] and bool(history), "Training history is incomplete")
    _require([int(row["epoch"]) for row in history] == list(range(1, len(history) + 1)), "Noncontiguous training history")
    keys = [(float(row["val_macro_roc_auc"]), float(row["val_macro_average_precision"]), -float(row["val_binary_cross_entropy"])) for row in history]
    _require(np.isfinite(keys).all(), "Nonfinite selection history")
    selected_index = max(range(len(history)), key=lambda index: keys[index])
    _require(selected_index + 1 == training["best_epoch"], "Selected checkpoint does not match the frozen lexicographic rule")
    metrics, predictions, prediction_rows = {}, {}, {}
    for split in PARTITIONS:
        rows = inputs.csv(out / f"{split}_predictions.csv")
        truth, p, recomputed = _prediction_arrays(rows, split)
        saved = inputs.json(out / f"{split}_metrics.json")
        _require(saved["class_names"] == list(CLASS_NAMES) and saved["n_samples"] == len(rows), f"Invalid {split} metric dimensions")
        for name in (*SUMMARY_METRICS, "weighted_f1", "binary_cross_entropy"):
            _close(saved[name], recomputed[name], f"{split}/{name}")
            _close(run["metrics"][split][name], saved[name], f"run.json/{split}/{name}")
        for name in CLASS_NAMES:
            for key, value in recomputed["per_label"][name].items():
                _close(saved["per_label"][name][key], value, f"{split}/{name}/{key}")
        metrics[split], predictions[split], prediction_rows[split] = saved, (truth, p), rows
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for identifier in ("ecg_id", "patient_id"):
            _require(not ({row[identifier] for row in prediction_rows[first]} & {row[identifier] for row in prediction_rows[second]}),
                     f"{identifier} overlap between {first} and {second}")
    for key, selected in zip(("macro_roc_auc", "macro_average_precision", "binary_cross_entropy"),
                             (keys[selected_index][0], keys[selected_index][1], -keys[selected_index][2])):
        _close(metrics["validation"][key], selected, f"selected validation/{key}")
    _require(ci["confidence_level"] == .95 and ci["n_resamples"] == cfg["uncertainty"]["resamples"], "Unexpected bootstrap protocol")
    _require(ci["n_records"] == metrics["test"]["n_samples"], "Bootstrap record count differs")
    _require(ci["n_unique_patients"] == len({row["patient_id"] for row in prediction_rows["test"]}), "Bootstrap patient count differs")
    for name in SUMMARY_METRICS:
        interval = ci["metrics"][name]
        _close(interval["point_estimate"], metrics["test"][name], f"bootstrap point/{name}")
        if interval["interval_status"] == "defined":
            lower, upper = _finite(interval["ci_lower"], name), _finite(interval["ci_upper"], name)
            _require(0 <= lower <= upper <= 1, f"Invalid interval bounds for {name}")
        else:
            _require(interval["ci_lower"] is None and interval["ci_upper"] is None, f"Inconsistent undefined interval for {name}")
    card = inputs.text(root / "docs/ptbxl_image_dataset.md")
    citations = ("https://physionet.org/content/ptb-xl/1.0.3/", "https://doi.org/10.13026/kfzx-aw45", "https://doi.org/10.1038/s41597-020-0495-6")
    _require(all(url in card for url in citations), "Expected source citations missing from the dataset card")
    verified_hashes = verification["run_artifact_sha256"]
    for path, digest in inputs.hashes.items():
        path = Path(path)
        if path.is_relative_to(out.resolve()) and path != (out / "verification.json").resolve():
            relative = str(path.relative_to(out.resolve()))
            _require(verified_hashes.get(relative) == digest, f"Verification is stale or does not cover reporting input: {relative}")
    return {"root": root, "out": out, "run": run, "config": cfg, "training": training, "provenance": provenance,
            "ci": ci, "metrics": metrics, "predictions": predictions, "rows": prediction_rows,
            "verification": verification, "inputs": inputs, "history": history, "model_dir": model_dir}


def _save_figure(figure, directory, name):
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")


def _figures(data, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    truth, probabilities = data["predictions"]["test"]
    test = data["metrics"]["test"]
    for name, is_pr in (("roc_curves", False), ("precision_recall_curves", True)):
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.7), constrained_layout=True)
        for index, (label, axis) in enumerate(zip(CLASS_NAMES, axes)):
            per_label = test["per_label"][label]
            if is_pr:
                if per_label["support"]:
                    precision, recall, _ = precision_recall_curve(truth[:, index], probabilities[:, index])
                    axis.step(recall, precision, where="post", color="#2F7B73", linewidth=1.8,
                              label=f"AP = {per_label['average_precision']:.3f}")
                else:
                    axis.text(.5, .5, "AP undefined: no positives", ha="center", transform=axis.transAxes)
                axis.axhline(per_label["support"] / len(truth), color="0.6", linestyle="--", linewidth=1, label="Prevalence")
                axis.set(xlabel="Recall", ylabel="Precision")
            else:
                if per_label["roc_auc"] is not None:
                    false_positive, true_positive, _ = roc_curve(truth[:, index], probabilities[:, index])
                    axis.plot(false_positive, true_positive, color="#2F7B73", linewidth=1.8, label=f"AUC = {per_label['roc_auc']:.3f}")
                else:
                    axis.text(.5, .5, "AUC undefined", ha="center", transform=axis.transAxes)
                axis.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1, label="Chance")
                axis.set(xlabel="False-positive rate", ylabel="True-positive rate")
            axis.set(title=f"{label} | positive n={per_label['support']:,}", xlim=(0, 1), ylim=(0, 1.02))
            axis.grid(alpha=.15)
            axis.legend(loc="lower right" if not is_pr else "lower left", fontsize=8)
        fig.suptitle(f"Held-out test {'precision–recall' if is_pr else 'ROC'} curves | {len(truth):,} ECGs | saved independent probabilities", fontsize=13)
        _save_figure(fig, directory, name)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    x, width = np.arange(5), .24
    for axis, metric in zip(axes, ("precision", "recall", "f1")):
        for index, split in enumerate(PARTITIONS):
            values = [data["metrics"][split]["per_label"][label][metric] for label in CLASS_NAMES]
            axis.bar(x + (index - 1) * width, values, width, label=split.capitalize(), color=PARTITION_COLORS[split])
        axis.set(xticks=x, xticklabels=CLASS_NAMES, ylim=(0, 1.02), ylabel=metric.capitalize() if metric != "f1" else "F1")
        axis.grid(axis="y", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Selected checkpoint: per-label decisions at the fixed 0.5 threshold", fontsize=13)
    _save_figure(fig, directory, "per_label_metrics")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), constrained_layout=True)
    for axis, metric, title in zip(axes, ("macro_roc_auc", "macro_average_precision", "macro_f1"), ("Macro ROC-AUC (primary)", "Macro average precision", "Macro F1 at 0.5")):
        values = [data["metrics"][split][metric] for split in PARTITIONS]
        axis.bar(np.arange(3), values, width=.6, color=[PARTITION_COLORS[split] for split in PARTITIONS])
        interval = data["ci"]["metrics"][metric]
        if interval["interval_status"] == "defined":
            low, high = interval["ci_lower"], interval["ci_upper"]
            axis.vlines(2, low, high, color="black", linewidth=1.4)
            axis.hlines([low, high], 1.91, 2.09, color="black", linewidth=1.4)
        for index, value in enumerate(values):
            top = max(value, interval["ci_upper"]) if index == 2 and interval["interval_status"] == "defined" else value
            axis.text(index, top + .025, f"{value:.3f}", ha="center", fontsize=10)
        axis.set(title=title, xticks=np.arange(3), xticklabels=[split.capitalize() for split in PARTITIONS], ylim=(0, 1.10))
        axis.grid(axis="y", alpha=.15)
    fig.legend(handles=[Line2D([0], [0], color="black", label="95% patient-bootstrap interval: test only")], loc="outside lower center", frameon=False)
    fig.suptitle("Selected checkpoint | independent patient partitions | no direct four-class comparison", fontsize=13)
    _save_figure(fig, directory, "partition_summary")
    plt.close(fig)


def _link(label, path):
    return f"[{label}](<{path.resolve()}>)"


def _format(value):
    return "undefined" if value is None else f"{_finite(value, 'reported score'):.4f}"


def _interval(ci, metric):
    value = ci["metrics"][metric]
    if value["interval_status"] != "defined":
        return "not estimable"
    return f"{value['ci_lower']:.4f}–{value['ci_upper']:.4f}"


def _report(data, figures):
    run, cfg, training, metrics, ci = (data[key] for key in ("run", "config", "training", "metrics", "ci"))
    train, validation, test = (metrics[split] for split in PARTITIONS)
    root, out = data["root"], data["out"]
    timing = _runtime_summary(run, training)
    scores = {"macro_roc_auc": "Macro ROC-AUC (primary)", "macro_average_precision": "Macro average precision",
              "macro_f1": "Macro F1", "micro_f1": "Micro F1", "label_accuracy": "Label accuracy", "subset_accuracy": "Exact-match accuracy"}
    lines = [
        "# PTB-XL small image spatiotemporal network report", "",
        f"**Completed and verified run: `{run['experiment_id']}`.** Held-out test macro ROC-AUC is **{_format(test['macro_roc_auc'])}** "
        f"(95% patient-bootstrap interval **{_interval(ci, 'macro_roc_auc')}**). "
        f"The checkpoint was selected at epoch **{training['best_epoch']} of {training['epochs_run']} completed epochs** using validation data only.", "",
        "This report is derived from saved predictions, metrics, training history, and provenance. It performs no model inference, refitting, threshold selection, or new bootstrap sampling. "
        "The test cohort was previously examined for the CNN–LSTM, so this next-model evaluation is exploratory rather than a fresh confirmatory test.", "",
        "## Cohort and model", "",
        "The source is the custom clean rendering of real native 500 Hz, 10-second, 12-lead "
        "[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) signals. "
        "Use the [versioned dataset DOI](https://doi.org/10.13026/kfzx-aw45) and "
        "[original publication](https://doi.org/10.1038/s41597-020-0495-6), with attribution to this derived image-rendering procedure. "
        f"The {_link('dataset card', root / 'docs/ptbxl_image_dataset.md')} documents source licensing (CC BY 4.0), labels, rendering and exclusions.", "",
        "All 21,799 records have rendered pages; 21,388 have at least one of the five mapped diagnostic superclasses and enter classification. "
        "The 411 unmapped records are excluded from classification, not treated as all-negative or normal. "
        "Targets are overlapping NORM, MI, STTC, CD and HYP labels derived from all present diagnostic SCP keys, including keys with zero recorded likelihood. "
        "Official patient folds 1–8, 9 and 10 define training, validation and test respectively.", "",
        "| Partition | Eligible ECGs | Unique patients | Official folds |", "|---|---:|---:|---|",
    ]
    for split, folds in zip(PARTITIONS, ("1–8", "9", "10")):
        lines.append(f"| {split.capitalize()} | {metrics[split]['n_samples']:,} | {len({row['patient_id'] for row in data['rows'][split]}):,} | {folds} |")
    lines += ["", "No ECG or patient identifier overlaps were found between the saved prediction partitions.", "",
        "Input preprocessing retains every complete lead row and its calibration pulse: original page rows y=80:2960, full width x=0:2240, "
        "then grayscale-to-ink conversion and nonoverlapping 2×2 maximum-ink pooling. The tensor is **12 × 120 × 1120**, "
        "with one channel per aligned lead. A single global mean and standard deviation fitted on training pixels standardize all partitions. "
        "Patient IDs, ECG IDs, SCP dictionaries, gain values and other numeric metadata are not model features.", "",
        "The **17,161-parameter** model starts with depthwise lead filtering; pointwise convolutions then combine aligned lead channels. "
        "Three convolution stages produce a 24 × 15 × 70 map, pooled to four vertical bands and arranged into **70 spatial steps × 96 features**. "
        "A **96-to-32 projection and two residual temporal-convolution blocks** replace the LSTM. Each block contains two kernel-3 left-padded convolutions, at dilation 1 in the first block and 2 in the second. "
        "Each convolution is followed by LayerNorm across 32 channels independently at each position, ReLU and dropout; each residual sum is followed by ReLU. "
        "Mean pooling, dropout and a linear head produce five logits. The head's receptive field is 13 supplied feature positions before whole-sequence averaging. "
        "The sequence includes calibration/margins and denotes image-feature columns, not recovered 500 Hz waveform samples. "
        "Five independent sigmoid probabilities need not sum to one. Left padding makes the temporal head causal given its inputs, but the upstream image CNN/GroupNorm and global pooling use whole-image context: this is an offline classifier, not streaming ECG processing. The design is inspired by [Bai et al.](https://arxiv.org/abs/1803.01271), not an exact reproduction or a claim of superior ECG performance.", "",
        f"Training used unweighted BCE with logits and AdamW, seed {cfg['training']['seed']}, "
        f"batch size {cfg['training']['batch_size']}, learning rate {cfg['training']['learning_rate']:g}, "
        f"weight decay {cfg['training']['weight_decay']:g}, dropout {cfg['training']['dropout']:g}, "
        f"gradient clipping at {cfg['training']['gradient_clip_norm']:g}, and {cfg['training']['num_threads']} CPU threads. "
        f"The {cfg['training']['scheduler']} schedule has minimum learning rate {cfg['training']['eta_min']:g}. "
        f"The prespecified budget was {cfg['training']['max_epochs']} epochs with patience {cfg['training']['patience']}. "
        "No augmentation, pretrained weights, class weighting or hyperparameter search was used.", "",
        "Checkpoint selection strictly maximized validation macro ROC-AUC, then macro average precision, then minimized validation probability BCE; "
        "the earliest exact tie was retained. All label decisions use the prespecified **probability ≥ 0.5** rule. "
        "There was one final test inference after the checkpoint, threshold, inputs and code were frozen; no test-guided tuning or calibration was performed.", "",
        "## Selected-checkpoint results", "",
        "| Metric | Train | Validation | Test | Test 95% patient interval |", "|---|---:|---:|---:|---|",
    ]
    for metric, label in scores.items():
        lines.append(f"| {label} | {_format(train[metric])} | {_format(validation[metric])} | {_format(test[metric])} | {_interval(ci, metric)} |")
    lines.append(f"| Weighted F1 (positive-support weights) | {_format(train['weighted_f1'])} | {_format(validation['weighted_f1'])} | {_format(test['weighted_f1'])} | Not bootstrapped |")
    lines += ["", "Macro scores average the five labels equally. Micro F1 pools all record-label decisions. "
        "Label accuracy counts correct individual binary decisions; exact-match accuracy requires all five decisions on an ECG to be correct. "
        "These accuracies are not interchangeable. Supports overlap and therefore do not sum to the ECG count.", "",
        "| Label | Train positives | Validation positives | Test positives | Test ROC-AUC | Test AP | Test precision | Test recall | Test specificity | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASS_NAMES:
        per = test["per_label"][name]
        lines.append(f"| {name} | {train['per_label'][name]['support']:,} | {validation['per_label'][name]['support']:,} | {per['support']:,} | "
                     + " | ".join(_format(per[key]) for key in ("roc_auc", "average_precision", "precision", "recall", "specificity", "f1")) + " |")
    lowest_recall = min(CLASS_NAMES, key=lambda name: test["per_label"][name]["recall"])
    per = test["per_label"][lowest_recall]
    auc_gap = train["macro_roc_auc"] - validation["macro_roc_auc"]
    f1_gap = train["macro_f1"] - validation["macro_f1"]
    lines += ["", f"At the fixed threshold, the lowest test recall is for **{lowest_recall}**: {per['tp']:,}/{per['support']:,} positives detected "
        f"({_format(per['recall'])}); {per['fn']:,} positives missed and {per['fp']:,} false positives. "
        "This describes the saved held-out error profile and is not a basis for retuning this test set.", "",
        f"Selected-checkpoint train minus validation differences are **{auc_gap:+.4f}** for macro ROC-AUC and **{f1_gap:+.4f}** for macro F1. "
        "These gaps alone do not establish an overfitting mechanism or isolate the contribution of the temporal-convolution head. "
        "Only one configuration and training seed were evaluated.", "",
        "The epoch training curve contains online, pre-update batch BCE averaged by record count. "
        "It is not a full training-partition metric computed after each epoch, and it includes training-time dropout. "
        "The training metrics in the tables above were evaluated separately at the selected checkpoint with dropout disabled. "
        "Validation metrics in the history use the complete validation partition after each epoch.", "",
        _runtime_text(timing), "",
        f"Early stopping: {'triggered' if training['stopped_early'] else 'not triggered before the epoch cap'}.", "",
        "## Figures", "",
    ]
    for name, title in zip(FIGURE_NAMES, ("Five test ROC curves", "Five test precision–recall curves", "Per-label precision, recall and F1", "Partition summary with test uncertainty")):
        lines += [f"{_link(title + ' (PNG)', figures / (name + '.png'))} · {_link('SVG', figures / (name + '.svg'))}", ""]
    lines += [f"{_link('Training curve (online training loss and epoch-end validation metrics)', data['model_dir'] / 'training_curve.png')}", "",
              f"{_link('Five test binary confusion matrices at the fixed 0.5 threshold', out / 'test_confusion_matrices.png')}", ""]
    lines += ["## Uncertainty and limitations", "",
        f"Saved 95% intervals use **{ci['n_resamples']:,} patient-cluster percentile bootstrap resamples**, seed {ci['seed']}, "
        f"over {ci['n_unique_patients']:,} test patients. Each sampled patient contributes all of their ECGs, with multiplicity when resampled. "
        "Intervals quantify test-sampling uncertainty conditional on this fitted model. They do not cover training-seed variation, "
        "model-selection uncertainty, annotation error, or external-domain shift. No multiplicity adjustment or bias correction is applied.", "",
        "| Metric | Defined bootstrap resamples | Undefined resamples |", "|---|---:|---:|",
    ]
    for metric, label in scores.items():
        value = ci["metrics"][metric]
        lines.append(f"| {label} | {value['defined_resamples']:,} | {value['undefined_resamples']:,} |")
    lines += ["", "ROC-AUC requires positive and negative truth examples, and AP requires positives. "
        "Full-five-label ranking scores remain undefined if a requisite label outcome is missing; intervals use only defined resamples. "
        "No defined-label-only average substitutes for the primary metric. Precision, recall and F1 use zero for zero denominators. "
        "Probability BCE uses float64 clipping and can differ from logits BCE at saturated probabilities.", "",
        "These clean, consistently rendered images do not establish performance on photographs, scanned paper, other ECG-machine layouts, "
        "or clinical deployment. Acquisition artifacts and annotation uncertainty persist. Page-wide gain reduction can compress small features; "
        "rasterization and maximum-ink pooling merge fine details and thicken strokes. Retained calibration pulses document scale, "
        "but the model does not explicitly reconstruct physical amplitudes or source samples. No external or prospective clinical validation was performed.", "",
        "This is a five-label PTB-XL patient cohort, while the prior Mendeley runs used four mutually exclusive folder labels from a different image source. "
        "A direct accuracy ranking against those LDA, CNN or spatial CNN+LSTM results would confound cohort, targets, preprocessing and evaluation. "
        "No such ranking is made. The PTB-XL test result had already been observed in the CNN–LSTM experiment before this run was specified. The same-cohort architecture comparison is exploratory and uses paired patient-level uncertainty; further development cannot present this cohort as newly untouched evidence. Head capacity and dropout placement also differ. Several-seed and independent-data evaluations remain future work.", "",
        f"See the {_link('paired comparison with the preserved PTB-XL CNN–LSTM', root / 'results/comparisons' / run['experiment_id'] / 'comparison.md')} for matched model differences; that file is produced separately from saved verified predictions.", "",
        "## Reproducibility evidence", "",
        f"- Completed UTC: `{run['completed_utc']}`.",
        f"- Frozen configuration SHA-256: `{data['provenance']['configuration_sha256']}`.",
        f"- Selected checkpoint SHA-256: `{run['checkpoint_sha256']}`.",
        f"- Training-only checkpoint replay matched saved probabilities exactly; prior protected artifacts were unchanged.",
        f"- {_link('Passed run verification', out / 'verification.json')}; {_link('run record', out / 'run.json')}; {_link('frozen configuration', out / 'config.json')}.",
        "- Verification checks saved-artifact integrity and consistency. It does not independently prove the absence of unrecorded inference; bootstrap interval bounds are structurally checked, not numerically resampled again.",
        f"- {_link('Provenance/source hashes', out / 'provenance.json')}; {_link('training metadata', out / 'training_metadata.json')}; {_link('history', data['model_dir'] / 'history.csv')}.",
        f"- {_link('Patient-bootstrap results', out / 'test_patient_bootstrap.json')}; {_link('report input/output hashes', figures / 'report_manifest.json')}.", "",
        "Regenerate reporting outputs from this completed run (default refuses existing reporting files):", "", "```bash",
        f'cd "{root}"', f"/opt/anaconda3/bin/python -m src.report_ptbxl_spatiotemporal --run-id {run['experiment_id']}", "```", "",
        "Use `--overwrite` only to replace this command's Markdown/figure/manifest outputs; saved predictions, metrics, models and provenance remain read-only.", "",
    ]
    return "\n".join(lines)


def generate_report(run_id, *, root=ROOT, overwrite=False):
    """Generate the report; root override supports isolated synthetic fixtures."""
    root = Path(root).resolve()
    data = _load_inputs(root, run_id)
    figures = data["out"] / "figures"
    report_path = root / "docs/ptbxl_spatiotemporal_report.md"
    destinations = [report_path, figures / "report_manifest.json"] + [figures / f"{name}.{extension}" for name in FIGURE_NAMES for extension in ("png", "svg")]
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("Reporting output already exists; use --overwrite to replace only reporting artifacts")
    with tempfile.TemporaryDirectory(prefix="ptbxl_report_") as directory:
        staging = Path(directory)
        _figures(data, staging)
        staged_report = staging / report_path.name
        staged_report.write_text(_report(data, figures))
        data["inputs"].unchanged()
        mapping = {report_path: staged_report}
        mapping.update({figures / path.name: path for path in staging.iterdir() if path.suffix in (".png", ".svg")})
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                    "source": "completed verified run; saved predictions only", "new_model_inference_performed": False,
                    "model_training_performed": False, "threshold_tuning_performed": False,
                    "bootstrap_resampling_performed": False, "input_sha256": data["inputs"].hashes,
                    "runtime_summary": _runtime_summary(data["run"], data["training"]),
                    "output_sha256": {str(destination): _sha(source) for destination, source in mapping.items()},
                    "test_case_gallery_generated": False}
        manifest_file = staging / "report_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        mapping[figures / manifest_file.name] = manifest_file
        for destination, source in mapping.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.report.tmp")
            temporary.write_bytes(source.read_bytes())
            temporary.replace(destination)
    return {"run_id": run_id, "report": str(report_path), "figure_directory": str(figures), "figures": 8,
            "new_model_inference_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace only this command's report, figures and reporting manifest")
    arguments = parser.parse_args()
    print(json.dumps(generate_report(arguments.run_id, overwrite=arguments.overwrite), indent=2))
