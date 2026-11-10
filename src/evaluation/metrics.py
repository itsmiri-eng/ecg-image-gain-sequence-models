from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)

from src.common import write_csv, write_json


def evaluate(y_true, probabilities, class_names):
    y_true = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y_true), len(class_names)) or not np.isfinite(p).all():
        raise ValueError("Invalid probability matrix")
    if np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    pred = p.argmax(axis=1)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    result = {
        "n_samples": len(y_true), "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=labels, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y_true, p, labels=labels)),
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted; class_names order",
        "class_names": class_names,
        "zero_division_policy": "undefined precision/recall/F1 reported as zero",
    }
    if set(y_true) == set(labels):
        result["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro"))
        result["roc_auc_ovr_weighted"] = float(roc_auc_score(y_true, p, multi_class="ovr", average="weighted"))
        result["roc_auc_per_class"] = {name: float(roc_auc_score(y_true == i, p[:, i]))
                                       for i, name in enumerate(class_names)}
        result["roc_auc_note"] = "One-vs-rest probability ranking, not diagnostic validation"
    else:
        result["roc_auc_ovr_macro"] = None
        result["roc_auc_note"] = "Undefined because an evaluation class is absent"
    return result


def save_evaluation(out, subset, rows, y_true, probabilities, class_names):
    out = Path(out)
    if len(rows) != len(y_true) or any(int(r["label"]) != int(y) for r, y in zip(rows, y_true)):
        raise ValueError("Prediction manifest and evaluation labels are not aligned")
    metrics = evaluate(y_true, probabilities, class_names)
    write_json(out / f"{subset}_metrics.json", metrics)
    predictions = []
    for row, y, p in zip(rows, y_true, probabilities):
        item = {"sample_id": row["sample_id"], "path": row["path"],
                "true_label": class_names[int(y)], "predicted_label": class_names[int(np.argmax(p))]}
        item.update({f"probability_{name}": float(p[i]) for i, name in enumerate(class_names)})
        predictions.append(item)
    write_csv(out / f"{subset}_predictions.csv", predictions)
    write_csv(out / f"{subset}_per_class.csv", [
        {"class": name, **metrics["per_class"][name], "roc_auc_ovr": metrics.get("roc_auc_per_class", {}).get(name)}
        for name in class_names])
    if subset == "test":
        cm = np.array(metrics["confusion_matrix"])
        write_csv(out / "test_confusion_matrix.csv", [
            {"true_class": name, **{c: int(cm[i, j]) for j, c in enumerate(class_names)}}
            for i, name in enumerate(class_names)])
        fig, ax = plt.subplots(figsize=(6.8, 5.5))
        ax.imshow(cm, cmap="Blues")
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=range(len(cm)), yticks=range(len(cm)), xticklabels=class_names,
               yticklabels=class_names, xlabel="Predicted class", ylabel="True class",
               title=f"Held-out test set (n={len(y_true)})")
        fig.tight_layout()
        fig.savefig(out / "test_confusion_matrix.png", dpi=180)
        plt.close(fig)
    return metrics
