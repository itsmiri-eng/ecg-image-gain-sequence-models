"""Five-label ECG evaluation with one prespecified 0.5 decision threshold.

Sigmoid probabilities are independent across labels and need not sum to one.
Undefined ranking metrics remain null. No threshold fitting is performed.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.common import write_csv, write_json


def _validated(y_true, probabilities, class_names, threshold):
    names = list(class_names)
    if len(names) != 5 or any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != 5:
        raise ValueError("Expected five unique, nonempty class names")
    if not isinstance(threshold, Real) or not math.isfinite(threshold) or threshold != 0.5:
        raise ValueError("This protocol uses one fixed threshold of 0.5; no threshold tuning")
    truth = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if truth.ndim != 2 or truth.shape[1] != 5 or truth.shape[0] == 0:
        raise ValueError("Expected a nonempty N x 5 binary truth matrix")
    if truth.dtype.kind not in "biuf" or not np.isfinite(truth).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("Truth labels must be finite binary values")
    if probabilities.shape != truth.shape or probabilities.dtype.kind not in "biuf":
        raise ValueError("Probability matrix must have the same N x 5 shape as truth")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return truth.astype(np.int8), probabilities, names


def _zero_division_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _mean_defined(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def evaluate_multilabel(y_true, probabilities, class_names, threshold=0.5):
    """Return JSON-safe metrics for five overlapping diagnostic targets.

    Primary comparison key is macro_roc_auc (requires both outcomes for every
    label); binary_cross_entropy is available for a prespecified loss tie-break.
    Full-five-label macro ranking metrics are null when any constituent is
    undefined. Explicitly named defined-label-only means are supplemental.
    Precision/recall/F1 use standard zero_division=0. ROC-AUC needs positives and
    negatives; average precision needs positives (an all-positive label has AP=1).
    """
    truth, p, names = _validated(y_true, probabilities, class_names, threshold)
    predicted = (p >= threshold).astype(np.int8)
    per_label, matrices = {}, {}
    for index, name in enumerate(names):
        y, pred = truth[:, index], predicted[:, index]
        tp = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        tn = int(np.sum((y == 0) & (pred == 0)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        positives, negatives = tp + fn, tn + fp
        precision = _zero_division_ratio(tp, tp + fp)
        recall = _zero_division_ratio(tp, positives)
        specificity = float(tn / negatives) if negatives else None
        auc = float(roc_auc_score(y, p[:, index])) if positives and negatives else None
        ap = float(average_precision_score(y, p[:, index])) if positives else None
        per_label[name] = {
            "precision": precision, "recall": recall, "f1": _zero_division_ratio(2 * tp, 2 * tp + fp + fn),
            "specificity": specificity,
            "balanced_accuracy": float((recall + specificity) / 2) if positives and negatives else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "support": positives, "negative_support": negatives,
            "predicted_positive": tp + fp, "roc_auc": auc, "average_precision": ap,
        }
        matrices[name] = [[tn, fp], [fn, tp]]

    aucs = [per_label[name]["roc_auc"] for name in names]
    aps = [per_label[name]["average_precision"] for name in names]
    supports = np.array([per_label[name]["support"] for name in names], dtype=np.int64)
    f1s = np.array([per_label[name]["f1"] for name in names])
    total_tp = sum(per_label[name]["tp"] for name in names)
    total_fp = sum(per_label[name]["fp"] for name in names)
    total_fn = sum(per_label[name]["fn"] for name in names)
    flat_y, flat_p = truth.ravel(), p.ravel()
    has_positive, has_negative = bool(np.any(flat_y == 1)), bool(np.any(flat_y == 0))
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(p, epsilon, 1 - epsilon)
    bce = float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)))
    return {
        "n_samples": len(truth), "n_labels": 5, "class_names": names, "threshold": float(threshold),
        "decision_rule": "Independent probability >= 0.5 for each label; no row-sum normalization",
        "subset_accuracy": float(np.mean(np.all(truth == predicted, axis=1))),
        "label_accuracy": float(np.mean(truth == predicted)),
        "hamming_loss": float(np.mean(truth != predicted)),
        "macro_f1": float(np.mean(f1s)),
        "micro_f1": _zero_division_ratio(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        "weighted_f1": float(np.average(f1s, weights=supports)) if supports.sum() else 0.0,
        "macro_precision": float(np.mean([per_label[name]["precision"] for name in names])),
        "macro_recall": float(np.mean([per_label[name]["recall"] for name in names])),
        "macro_roc_auc": float(np.mean(aucs)) if all(value is not None for value in aucs) else None,
        "micro_roc_auc": float(roc_auc_score(flat_y, flat_p)) if has_positive and has_negative else None,
        "macro_average_precision": float(np.mean(aps)) if all(value is not None for value in aps) else None,
        "micro_average_precision": float(average_precision_score(flat_y, flat_p)) if has_positive else None,
        "macro_roc_auc_defined_labels": _mean_defined(aucs),
        "macro_average_precision_defined_labels": _mean_defined(aps),
        "roc_auc_defined_labels": [name for name in names if per_label[name]["roc_auc"] is not None],
        "average_precision_defined_labels": [name for name in names if per_label[name]["average_precision"] is not None],
        "binary_cross_entropy": bce, "bce_probability_clip_epsilon": float(epsilon),
        "per_label": per_label, "confusion_matrices": matrices,
        "confusion_matrix_axes": "rows=true [negative, positive]; columns=predicted [negative, positive]",
        "accuracy_definitions": {
            "subset_accuracy": "Fraction of records with all five label decisions exactly correct",
            "label_accuracy": "Fraction of individual record-label decisions correct",
            "hamming_loss": "Fraction of individual record-label decisions incorrect",
        },
        "zero_division_policy": "Precision, recall and F1 use zero_division=0; weighted F1 is zero when total positive support is zero",
        "undefined_metric_policy": "ROC-AUC needs positive and negative truth; AP needs positive truth. Full-five-label macro ranking scores are null if any label is undefined. Defined-label-only means are named separately. Specificity is null without negatives; balanced accuracy is null without both outcomes.",
    }


def _manifest_binary(value):
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return 1
        if value.lower() in {"false", "0"}:
            return 0
        raise ValueError("Prediction manifest contains a nonbinary label")
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)) and value in (0, 1):
        return int(value)
    raise ValueError("Prediction manifest contains a nonbinary label")


def save_multilabel_evaluation(out, split, rows, y_true, probabilities, class_names, threshold=0.5):
    """Save metrics, aligned source-level predictions and five binary matrices.

    Rows require ecg_id, patient_id and label_<class> fields; image_path is copied
    when present. No identifier is put in the plot. Each CSV prediction carries
    truth_<class>, probability_<class> and prediction_<class> for all five labels.
    """
    truth, p, names = _validated(y_true, probabilities, class_names, threshold)
    if not isinstance(split, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", split):
        raise ValueError("Invalid evaluation split name")
    rows = list(rows)
    if len(rows) != len(truth):
        raise ValueError("Prediction manifest and truth have different record counts")
    ecg_ids = set()
    for row, labels in zip(rows, truth):
        if any(key not in row or not str(row[key]).strip() for key in ("ecg_id", "patient_id")):
            raise ValueError("Prediction manifest requires ECG and patient identifiers")
        ecg_id = str(row["ecg_id"])
        if ecg_id in ecg_ids:
            raise ValueError("Prediction manifest repeats an ECG identifier")
        ecg_ids.add(ecg_id)
        if "split" in row and row["split"] != split:
            raise ValueError("Prediction manifest contains a different partition")
        for index, name in enumerate(names):
            key = f"label_{name}"
            if key not in row or _manifest_binary(row[key]) != int(labels[index]):
                raise ValueError("Prediction manifest and evaluation labels are not aligned")
    result = evaluate_multilabel(truth, p, names, threshold)
    predicted = (p >= threshold).astype(np.int8)
    predictions = []
    for row, y, probability, pred in zip(rows, truth, p, predicted):
        record = {"ecg_id": row["ecg_id"], "patient_id": row["patient_id"], "image_path": row.get("image_path", "")}
        for index, name in enumerate(names):
            record.update({f"truth_{name}": int(y[index]), f"probability_{name}": float(probability[index]),
                           f"prediction_{name}": int(pred[index])})
        predictions.append(record)
    out = Path(out)
    write_json(out / f"{split}_metrics.json", result)
    write_csv(out / f"{split}_predictions.csv", predictions)
    write_csv(out / f"{split}_per_label.csv", [{"label": name, **result["per_label"][name]} for name in names])
    _plot_confusion_matrices(out / f"{split}_confusion_matrices.png", split, result)
    return result


def _plot_confusion_matrices(destination, split, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = [np.asarray(metrics["confusion_matrices"][name]) for name in metrics["class_names"]]
    maximum = max(1, max(int(matrix.max()) for matrix in matrices))
    figure, axes = plt.subplots(1, 5, figsize=(15.5, 3.9), constrained_layout=True)
    for axis, name, matrix in zip(axes, metrics["class_names"], matrices):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=11,
                          color="white" if matrix[row, column] > maximum * .55 else "black")
        axis.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"],
                 yticklabels=["Negative", "Positive"], xlabel="Predicted label")
    axes[0].set_ylabel("True label")
    figure.suptitle(f"{split.capitalize()}: five binary confusion matrices | n = {metrics['n_samples']:,} | threshold = 0.5",
                   fontsize=14)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
