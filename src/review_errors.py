"""Review frozen baseline errors without training or model inference.

Case-level analysis is restricted to development partitions. Previously recorded
test confusion matrices are reproduced only as explicitly post-hoc aggregates.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.common import ROOT, read_json, sha256_file, utc_now, write_csv, write_json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ("lda", "cnn")
DEVELOPMENT = ("train", "validation")
MINORITY = ("MI", "History_MI")
OUTPUT = ROOT / "results/readiness/errors"
REPORT = ROOT / "docs/minority_error_review.md"
LOCK = ROOT / "experiments/reviews/2026-09-24/baseline_lock.json"


def csv_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def verify_lock():
    lock = read_json(LOCK)
    changed = [path for path, expected in lock["files"].items() if sha256_file(ROOT / path) != expected]
    if changed:
        raise ValueError(f"Locked baseline artifacts changed: {changed}")
    return len(lock["files"])


def metrics_from_confusion(cm):
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    correct = cm.diagonal()
    recall = np.divide(correct, support, out=np.zeros(4), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(4), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4), where=precision + recall > 0)
    return {
        "accuracy": float(correct.sum() / cm.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / support.sum()),
    }, support, predicted, precision, recall, f1


def make_figures(matrices, names, cases, gallery_rows, cache):
    short = [name.replace("_", " ") for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        cm = matrices[model, "validation"]
        fraction = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{fraction[i, j]:.0%}", ha="center", va="center",
                        color="white" if fraction[i, j] > .55 else "black", fontsize=10)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short,
               xlabel="Saved prediction", ylabel="Supplied folder label", title=model.upper())
    fig.suptitle("Validation confusion: counts and percentage of each supplied class (n=98)")
    fig.savefig(OUTPUT / "validation_confusion.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    for ax, partition in zip(axes, DEVELOPMENT):
        supplied = matrices["lda", partition].sum(axis=1)
        vectors = [supplied] + [matrices[model, partition].sum(axis=0) for model in MODELS]
        colors = ["#9ca3af", "#2563eb", "#d97706"]
        for offset, (counts, label, color) in enumerate(zip(vectors, ["Supplied labels", "LDA predictions", "CNN predictions"], colors)):
            bars = ax.bar(np.arange(4) + (offset - 1) * .25, counts, width=.25, label=label, color=color)
            ax.bar_label(bars, fontsize=8, padding=2)
        ax.set(xticks=range(4), xticklabels=short, ylabel="Images", title=f"{partition.title()} (n={supplied.sum()})")
        ax.set_ylim(0, max(max(vector) for vector in vectors) * 1.16)
        ax.legend(fontsize=8)
    fig.suptitle("Development prediction counts: CNN concentrates on Normal and Abnormal")
    fig.savefig(OUTPUT / "development_prediction_counts.png", dpi=160)
    plt.close(fig)

    # Only these explicitly selected validation indices are rendered. No test gallery.
    with np.load(cache, allow_pickle=False) as loaded:
        tensors = loaded["images"]
        fig, axes = plt.subplots(4, 3, figsize=(12.4, 10.0), constrained_layout=True)
        for ax, row in zip(axes.ravel(), gallery_rows):
            image = tensors[int(row["array_index"]), 0]
            if hashlib.sha256(image[None].tobytes()).hexdigest() != row["input_sha256"]:
                raise ValueError("Selected validation tensor does not match frozen input hash")
            ax.imshow(1 - image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Supplied: {row['supplied_label']} · {row['sample_id'][:8]}\n"
                         f"LDA: {row['lda_prediction']} | CNN: {row['cnn_prediction']}", fontsize=8)
            ax.axis("off")
        for ax in axes.ravel()[len(gallery_rows):]:
            ax.axis("off")
        fig.suptitle("Validation inputs for visual quality review only — no diagnosis or relabeling\n"
                     "Frozen 128×256 crops; up to six lexicographically first sample IDs per minority class", fontsize=11)
        fig.savefig(OUTPUT / "validation_minority_inputs.png", dpi=160)
        plt.close(fig)


def main():
    if REPORT.exists() or (OUTPUT.exists() and any(OUTPUT.iterdir())):
        raise FileExistsError("Preserve the existing error review; archive its new outputs explicitly before rerunning")
    locked_count = verify_lock()
    protocol = read_json(ROOT / "data/splits/protocol.json")
    names = protocol["config"]["class_names"]
    label_index = {name: index for index, name in enumerate(names)}
    split_path = ROOT / "data/splits/fixed_split.csv"
    split_rows = csv_rows(split_path)
    if sha256_file(split_path) != protocol["audit"]["split_sha256"]:
        raise ValueError("Frozen split hash changed")
    sources = {str(split_path.relative_to(ROOT)): sha256_file(split_path)}
    matrices, predictions, saved_metrics = {}, {}, {}
    group_rows = defaultdict(list)
    for row in split_rows:
        if row["split"] in DEVELOPMENT:
            group_rows[row["group_id"]].append(row)
    group_labels = {group: sorted({row["class_name"] for row in rows}) for group, rows in group_rows.items()}
    for model in MODELS:
        run_dir = ROOT / "results" / model / f"{model}_v1_seed42"
        if sha256_file(run_dir / "split_used.csv") != protocol["audit"]["split_sha256"]:
            raise ValueError(f"{model} used a different split")
        for partition in (*DEVELOPMENT, "test"):
            metric_path = run_dir / f"{partition}_metrics.json"
            saved = read_json(metric_path)
            sources[str(metric_path.relative_to(ROOT))] = sha256_file(metric_path)
            cm = np.asarray(saved["confusion_matrix"], dtype=np.int64)
            if partition in DEVELOPMENT:
                prediction_path = run_dir / f"{partition}_predictions.csv"
                source_rows = csv_rows(prediction_path)
                sources[str(prediction_path.relative_to(ROOT))] = sha256_file(prediction_path)
                expected = [row for row in split_rows if row["split"] == partition]
                if [row["sample_id"] for row in source_rows] != [row["sample_id"] for row in expected]:
                    raise ValueError("Development predictions are not aligned with the frozen split")
                recomputed = np.zeros((4, 4), dtype=np.int64)
                for row, source in zip(source_rows, expected):
                    if row["true_label"] != source["class_name"] or row["path"] != source["path"]:
                        raise ValueError("Saved development labels/paths differ from the split")
                    p = np.array([float(row[f"probability_{name}"]) for name in names])
                    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-6):
                        raise ValueError("Invalid saved probabilities")
                    ranking = np.argsort(-p, kind="stable")
                    if names[ranking[0]] != row["predicted_label"]:
                        raise ValueError("Saved prediction differs from probability argmax")
                    true = label_index[row["true_label"]]
                    recomputed[true, ranking[0]] += 1
                    predictions[model, row["sample_id"]] = {
                        "prediction": row["predicted_label"], "probabilities": p,
                        "rank": int(np.flatnonzero(ranking == true)[0]) + 1,
                        "error": row["predicted_label"] != row["true_label"],
                    }
                if not np.array_equal(cm, recomputed):
                    raise ValueError("Development confusion matrix differs from saved probabilities")
            metrics, *_ = metrics_from_confusion(cm)
            if any(abs(metrics[key] - saved[key]) > 1e-12 for key in metrics):
                raise ValueError("Saved aggregate metrics disagree with confusion matrix")
            matrices[model, partition], saved_metrics[model, partition] = cm, saved

    cases = []
    for source in split_rows:
        if source["split"] not in DEVELOPMENT:
            continue
        row = {key: source[key] for key in ["sample_id", "array_index", "path", "file_sha256", "pixel_sha256", "input_sha256", "group_id", "split"]}
        row.update(supplied_label=source["class_name"], group_size=len(group_rows[source["group_id"]]),
                   group_supplied_labels="|".join(group_labels[source["group_id"]]),
                   group_has_multiple_supplied_labels=len(group_labels[source["group_id"]]) > 1)
        flags = []
        if row["supplied_label"] in MINORITY:
            flags.append("minority_supplied_class")
        if row["group_has_multiple_supplied_labels"]:
            flags.append("group_contains_multiple_supplied_labels")
        for model in MODELS:
            record = predictions[model, row["sample_id"]]
            row.update({f"{model}_prediction": record["prediction"], f"{model}_error": record["error"],
                        f"{model}_supplied_label_rank": record["rank"], f"{model}_supplied_label_in_top2": record["rank"] <= 2})
            row.update({f"{model}_probability_{name}": float(record["probabilities"][i]) for i, name in enumerate(names)})
        if row["lda_prediction"] != row["cnn_prediction"]:
            flags.append("model_disagreement")
        if row["lda_error"] and row["cnn_error"]:
            flags.append("both_models_disagree_with_supplied_label")
        if row["supplied_label"] in MINORITY and row["cnn_error"]:
            flags.append("cnn_minority_error")
        row["review_flags"] = "|".join(flags)
        cases.append(row)
    if len(cases) != 393 or any(row["split"] == "test" for row in cases):
        raise ValueError("Unexpected development-only review population")

    class_rows, confusion_rows, destination_rows, rank_rows = [], [], [], []
    partition_summary = {}
    for partition in (*DEVELOPMENT, "test"):
        partition_summary[partition] = {}
        for model in MODELS:
            cm = matrices[model, partition]
            metrics, support, predicted, precision, recall, f1 = metrics_from_confusion(cm)
            role = "development" if partition in DEVELOPMENT else "previously_recorded_test_aggregate_posthoc"
            model_summary = {**metrics, "n_samples": int(cm.sum()), "predicted_class_counts": dict(zip(names, map(int, predicted))),
                             "normal_or_abnormal_predictions": int(predicted[0] + predicted[2]), "minority": {}}
            for i, name in enumerate(names):
                selected = [r for r in cases if r["split"] == partition and r["supplied_label"] == name]
                top2 = sum(r[f"{model}_supplied_label_in_top2"] for r in selected) if selected else None
                class_rows.append({"split": partition, "evidence_role": role, "model": model, "supplied_label": name,
                                   "support": int(support[i]), "predicted_count": int(predicted[i]), "correct": int(cm[i, i]),
                                   "errors": int(support[i] - cm[i, i]), "precision": float(precision[i]), "recall": float(recall[i]),
                                   "f1": float(f1[i]), "development_supplied_label_top2_count": top2,
                                   "development_supplied_label_top2_fraction": top2 / len(selected) if selected else None})
                if selected:
                    for rank in range(1, 5):
                        rank_rows.append({"split": partition, "model": model, "supplied_label": name, "supplied_label_rank": rank,
                                          "count": sum(r[f"{model}_supplied_label_rank"] == rank for r in selected)})
                if name in MINORITY:
                    model_summary["minority"][name] = {"support": int(support[i]), "correct": int(cm[i, i]),
                                                       "errors": int(support[i] - cm[i, i]), "recall": float(recall[i]),
                                                       "prediction_destinations": dict(zip(names, map(int, cm[i])))}
                    if selected:
                        model_summary["minority"][name]["supplied_label_top2_count"] = top2
                for j, predicted_name in enumerate(names):
                    confusion_rows.append({"split": partition, "evidence_role": role, "model": model,
                                           "supplied_label": name, "predicted_label": predicted_name, "count": int(cm[i, j])})
                    if name in MINORITY and i != j:
                        destination_rows.append({"split": partition, "evidence_role": role, "model": model, "supplied_label": name,
                                                 "error_destination": predicted_name, "count": int(cm[i, j]), "fraction_of_supplied_class": float(cm[i, j] / support[i])})
            partition_summary[partition][model] = model_summary

    overlap_rows, association_rows = [], []
    for partition in DEVELOPMENT:
        for scope in ["all_classes", "MI", "History_MI", "minority_combined"]:
            selected = [r for r in cases if r["split"] == partition and
                        (scope == "all_classes" or r["supplied_label"] == scope or (scope == "minority_combined" and r["supplied_label"] in MINORITY))]
            counts = {
                "both_correct": sum(not r["lda_error"] and not r["cnn_error"] for r in selected),
                "lda_only_wrong": sum(r["lda_error"] and not r["cnn_error"] for r in selected),
                "cnn_only_wrong": sum(not r["lda_error"] and r["cnn_error"] for r in selected),
                "both_wrong_same_prediction": sum(r["lda_error"] and r["cnn_error"] and r["lda_prediction"] == r["cnn_prediction"] for r in selected),
                "both_wrong_different_prediction": sum(r["lda_error"] and r["cnn_error"] and r["lda_prediction"] != r["cnn_prediction"] for r in selected),
            }
            if sum(counts.values()) != len(selected):
                raise ValueError("Error overlap categories do not partition the selected development cases")
            overlap_rows.append({"split": partition, "scope": scope, "samples": len(selected), **counts,
                                 "model_agreement": sum(r["lda_prediction"] == r["cnn_prediction"] for r in selected)})
            for diverse in (False, True):
                subset = [r for r in selected if r["group_has_multiple_supplied_labels"] == diverse]
                for model in MODELS:
                    errors = sum(r[f"{model}_error"] for r in subset)
                    association_rows.append({"split": partition, "scope": scope, "model": model,
                                             "group_has_multiple_supplied_labels": diverse, "samples": len(subset),
                                             "groups": len({r["group_id"] for r in subset}), "errors": errors,
                                             "error_fraction": errors / len(subset) if subset else None})
    gallery_rows = []
    for name in MINORITY:
        gallery_rows.extend(sorted([row for row in cases if row["split"] == "validation" and row["supplied_label"] == name], key=lambda r: r["sample_id"])[:6])
    if len(gallery_rows) > 12 or any(row["split"] != "validation" for row in gallery_rows):
        raise ValueError("Gallery violates validation-only scope")
    cache_path = ROOT / "data/processed/inputs_v1.npz"
    if sha256_file(cache_path) != protocol["audit"]["cache_sha256"]:
        raise ValueError("Frozen input cache changed")
    sources[str(cache_path.relative_to(ROOT))] = sha256_file(cache_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("development_cases.csv", cases), ("partition_class_metrics.csv", class_rows),
        ("confusion_counts.csv", confusion_rows), ("minority_error_destinations.csv", destination_rows),
        ("development_supplied_label_ranks.csv", rank_rows), ("development_error_overlap.csv", overlap_rows),
        ("development_group_associations.csv", association_rows), ("validation_gallery_index.csv", gallery_rows),
    ]:
        write_csv(OUTPUT / filename, rows)
    make_figures(matrices, names, cases, gallery_rows, cache_path)
    summary = {
        "created_utc": utc_now(), "status": "completed", "dataset_version": protocol["audit"]["dataset_version"],
        "split_version": protocol["audit"]["split_version"], "no_new_model_inference": True,
        "no_retraining": True, "no_test_case_gallery": True, "new_test_case_analysis": False,
        "test_prediction_csvs_read": False, "thresholds_calibration_or_settings_changed": False,
        "development_case_rows": len(cases), "validation_gallery_images": len(gallery_rows),
        "partition_summary": partition_summary, "development_error_overlap": overlap_rows,
        "development_group_associations": association_rows,
        "rank_policy": "Development only; descending saved probability with stable class-order tie breaking. Top2 is descriptive, not a new decision rule.",
        "group_policy": "Whether the frozen report-ID group has more than one supplied folder label within its development partition. A review flag, not an adjudicated contradiction or verified patient identity.",
        "test_policy": "Only existing aggregate test confusion/metrics JSON was read. Test results were already exposed, so reuse is exploratory; no test-guided settings are proposed.",
        "gallery_policy": "First six sample IDs lexicographically per validation minority supplied class, rendered from identical frozen model tensors. Unblinded input-quality review, separate from the blind annotation queue; no diagnoses assigned.",
        "provenance": {"source_sha256": sources, "review_script_sha256": sha256_file(Path(__file__)),
                       "baseline_lock_sha256": sha256_file(LOCK), "locked_artifacts_unchanged": verify_lock(),
                       "locked_artifacts_checked_before_review": locked_count},
        "output_files": sorted(path.name for path in OUTPUT.iterdir()),
    }
    write_json(OUTPUT / "summary.json", summary)
    write_report(summary, association_rows)
    print({"status": "completed", "development_rows": len(cases), "gallery_images": len(gallery_rows),
           "locked_artifacts_unchanged": locked_count, "summary": str(OUTPUT / "summary.json")})


def write_report(summary, associations):
    p = summary["partition_summary"]
    lines = ["# Minority-class development error review", "",
             "This review uses the already saved LDA/CNN predictions. It performs **no inference, retraining, calibration, threshold adjustment, or model selection**. Case-level analysis and images are limited to training and validation. Folder labels are described as supplied labels, not independently adjudicated diagnoses.", "",
             "## Validation findings", "",
             "| Supplied class | Images | LDA correct | CNN correct | CNN error destinations |", "|---|---:|---:|---:|---|"]
    for name in MINORITY:
        lda, cnn = p["validation"]["lda"]["minority"][name], p["validation"]["cnn"]["minority"][name]
        destinations = ", ".join(f"{label}: {count}" for label, count in cnn["prediction_destinations"].items() if label != name and count)
        lines.append(f"| {name} | {cnn['support']} | {lda['correct']} | {cnn['correct']} | {destinations} |")
    lines.extend(["", "CNN predicts Normal or Abnormal on **92/98 validation images (93.9%)**, compared with **68/98 (69.4%)** for LDA. The supplied validation labels contain 75/98 Normal or Abnormal images. CNN's validation accuracy is higher (71.4% vs 62.2%), while balanced accuracy is lower (50.9% vs 54.5%) and macro F1 is lower (0.461 vs 0.530). These conclusions use validation evidence alone.", "",
                  "The same minority failure occurs in training: CNN correctly predicts **1/18 MI** and **0/52 History_MI** images, compared with LDA **13/18** and **45/52**. Both models nevertheless have **213/295 training predictions correct**. CNN predicts Normal or Abnormal on **276/295 training images (93.6%)**. This is evidence of uneven class learning; it does not isolate whether representation, optimization, sample size, label meaning, or image artifacts cause it. Minority failure is not explained by a train-to-validation accuracy gap alone.", "",
                  "## Supplied-label rank and shared errors", "",
                  "Ranks below describe the existing probability order; they do not change the classifier or recommend a new decision rule.", "",
                  "| Validation class | LDA supplied label in top 2 | CNN supplied label in top 2 |", "|---|---:|---:|"])
    for name in MINORITY:
        lda, cnn = p["validation"]["lda"]["minority"][name], p["validation"]["cnn"]["minority"][name]
        lines.append(f"| {name} | {lda['supplied_label_top2_count']}/{lda['support']} | {cnn['supplied_label_top2_count']}/{cnn['support']} |")
    lines.extend(["", "Across all validation images, both models are wrong on **16/98**, LDA alone on **21/98**, CNN alone on **12/98**, and both correct on **49/98**. They select the same wrong label on **10 of their 16 shared errors**. For History_MI, 8 are shared errors and 9 are CNN-only errors; neither model pair is jointly correct on a History_MI case. For MI, 3 are shared errors, 1 is LDA-only, and 2 are CNN-only, with no jointly correct case. Agreement is not label validation.", "",
                  "## Group label diversity: descriptive association", "",
                  "A flagged group contains multiple supplied folder labels. Equal report IDs were already grouped before splitting; their relationship to patients remains unverified. Different labels within a group can reflect different recordings or inconsistent labels, so the flag does not establish an error.", "",
                  "| Validation scope | Multi-label group? | Images / groups | LDA errors | CNN errors |", "|---|---|---:|---:|---:|"])
    for scope in ["all_classes", "minority_combined", "MI", "History_MI"]:
        for diverse in (False, True):
            selected = [r for r in associations if r["split"] == "validation" and r["scope"] == scope and r["group_has_multiple_supplied_labels"] == diverse]
            lda = next(r for r in selected if r["model"] == "lda")
            cnn = next(r for r in selected if r["model"] == "cnn")
            lines.append(f"| {scope.replace('_', ' ')} | {'Yes' if diverse else 'No'} | {lda['samples']} / {lda['groups']} | {lda['errors']} | {cnn['errors']} |")
    lines.extend(["", "These are descriptive counts, not a causal or significance analysis. Images within groups are dependent, group sizes and class composition differ, and some cells are very small. MI has only six validation images. No diagnosis is inferred from the group association.", "",
                  "## Already exposed test results: post-hoc aggregate context only", "",
                  "The test prediction CSVs were not read for this review. Existing test confusion matrices reproduce the recorded aggregates: LDA correctly predicts MI **3/6** and History_MI **10/17**; CNN **1/6** and **4/17**. CNN predicts Normal or Abnormal on **88/98** test images. This context does not drive changes or examples here. The test set has already been exposed; further models on this cohort remain exploratory, not fresh unbiased validation.", "",
                  "## Review artifacts and limits", "",
                  "- `results/readiness/errors/development_cases.csv`: 393 training/validation cases, supplied labels, paths, hashes, groups, saved predictions/probabilities, supplied-label ranks, and review flags. It contains no test cases and is unblinded; keep it separate from the independent blind label queue.",
                  "- `partition_class_metrics.csv`, `confusion_counts.csv`, and `minority_error_destinations.csv`: reproducible partition aggregates with an explicit post-hoc role on test rows.",
                  "- `development_error_overlap.csv`, `development_supplied_label_ranks.csv`, and `development_group_associations.csv`: development-only tables.",
                  "- `validation_confusion.png` and `development_prediction_counts.png`: development figures.",
                  "- `validation_minority_inputs.png` and `validation_gallery_index.csv`: deterministic first six sample IDs per minority class, at most 12 validation crops identical to model inputs. Headers are excluded by the frozen crop. Inspect only clipping, layout, grid/text remnants, or other input-quality features; no diagnoses or new labels were assigned.",
                  "- `summary.json`: machine-readable counts, source hashes, scope flags, and verification that all 67 locked baseline artifacts are unchanged.", "",
                  "Before future development, the findings justify documenting class-wise failure and independently reviewing label meaning and input quality. They do not justify a particular architecture, calibration procedure, threshold, or claimed remedy. Predeclare any next experiment using development evidence and retain the existing baselines unchanged.", "",
                  "Reproduce after explicitly archiving this review's new outputs:", "", "```bash", "/opt/anaconda3/bin/python -m src.review_errors", "```", ""])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
