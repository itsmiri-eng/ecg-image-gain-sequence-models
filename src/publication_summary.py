"""Build mutable publication views from immutable, verified experiment artifacts.

No fitting, inference, checkpoint deserialization, bootstrap or legacy-log writes.
Run: python -m src.publication_summary [--output-dir DIR] [--master-log FILE].
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIELDS = ["experiment_id", "date", "model", "status", "dataset_version",
                 "split_version", "preprocessing", "random_seed", "hyperparameters",
                 "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "notes",
                 "output_directory"]
PARTITIONS = ("train", "validation", "test")
AUDIT = "experiments/reviews/2026-11-01/results_audit.json"
OLD_RUNS = {"lda_v1_seed42": ("experiments/logs/delivery_manifest.json", "files"),
            "cnn_v1_seed42": ("experiments/logs/delivery_manifest.json", "files"),
            "cnn_lstm_image_v1_seed42": ("experiments/logs/cnn_lstm_delivery_manifest.json", "artifacts_sha256")}
METRICS = ["n_samples", "accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "weighted_f1",
           "macro_precision", "macro_recall", "macro_specificity", "macro_roc_auc",
           "macro_average_precision", "micro_roc_auc", "micro_average_precision",
           "subset_accuracy", "label_accuracy", "hamming_loss", "log_loss", "binary_cross_entropy"]
BASE_FIELDS = ["experiment_id", "cohort", "model", "architecture", "status", "central_log_status",
               "status_source", "status_last_updated_utc", "liveness_verified", "verification_status",
               "exclusion_reason", "started_utc", "completed_utc", "git_commit", "dataset_version",
               "split_version", "class_names", "preprocessing", "normalization", "random_seed",
               "hyperparameters", "parameters", "parameter_definition", "epochs_run", "best_epoch",
               "checkpoint_selection", "training_seconds", "training_module_seconds",
               "runner_elapsed_seconds", "utc_wall_span_seconds", "device", "threshold",
               "accuracy_definition", "balanced_accuracy_definition", "test_inference_calls_recorded",
               "limitations", "output_directory", "model_directory", "checkpoint_path",
               "verification_paths", "run_sha256", "checkpoint_sha256", "recorded_source_sha256"]
MASTER_FIELDS = BASE_FIELDS + [f"{p}_{m}" for p in PARTITIONS for m in METRICS]
COMPARISON_FIELDS = ["experiment_id", "cohort", "model", "partition", "accuracy_definition",
                     "balanced_accuracy_definition", "threshold"] + METRICS
LABEL_FIELDS = ["experiment_id", "cohort", "model", "partition", "label", "threshold", "n_samples",
                "tp", "fp", "tn", "fn", "support", "negative_support", "predicted_positive",
                "precision", "sensitivity", "specificity", "f1", "binary_balanced_accuracy",
                "roc_auc", "average_precision"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class Inputs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def path(self, value: str | Path, base: Path | None = None) -> Path:
        path = ((base or self.root) / value).resolve()
        _require(path.is_relative_to(self.root), f"Input escapes repository: {value}")
        return path

    def read(self, path: Path) -> bytes:
        path = self.path(path)
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        name = path.relative_to(self.root).as_posix()
        _require(name not in self.hashes or self.hashes[name] == digest, f"Input changed: {name}")
        self.hashes[name] = digest
        return value

    def json(self, path: Path) -> dict:
        return json.loads(self.read(path), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))

    def check(self, path: Path, expected: str) -> None:
        actual = hashlib.sha256(self.read(path)).hexdigest()
        _require(actual == expected, f"SHA256 mismatch: {path}")

    def unchanged(self) -> None:
        for name, digest in self.hashes.items():
            _require(_sha(self.root / name) == digest, f"Input changed while summarizing: {name}")


def _cohort(dataset: str) -> str:
    if dataset == "ptbxl_images_v1":
        return "PTB-XL five-label"
    if dataset.startswith("image_dedup_"):
        return "Mendeley four-class"
    return "unavailable"


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _close(actual: Any, expected: float | None, context: str) -> None:
    _require(actual is None and expected is None or
             isinstance(actual, (int, float)) and expected is not None and
             math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), f"Inconsistent {context}")


def _metric_rows(run_id: str, model: str, cohort: str, partition: str, metric: dict) -> tuple[dict, list[dict]]:
    multilabel = cohort == "PTB-XL five-label"
    names = metric["class_names"]
    n = metric["n_samples"]
    _require(n > 0 and len(names) == (5 if multilabel else 4) and len(set(names)) == len(names),
             f"Invalid metric dimensions: {run_id}/{partition}")
    common = dict(experiment_id=run_id, cohort=cohort, model=model, partition=partition,
                  threshold=metric.get("threshold", "NA"))
    rows = []
    matrix = metric.get("confusion_matrix")
    if not multilabel:
        _require(len(matrix) == len(names) and all(len(r) == len(names) for r in matrix), "Invalid confusion shape")
        _require(all(isinstance(v, int) and v >= 0 for r in matrix for v in r) and
                 sum(map(sum, matrix)) == n, "Invalid multiclass confusion counts")
    for i, name in enumerate(names):
        saved = metric["per_label" if multilabel else "per_class"][name]
        if multilabel:
            tp, fp, tn, fn = (saved[k] for k in ("tp", "fp", "tn", "fn"))
            _require(all(isinstance(v, int) and v >= 0 for v in (tp, fp, tn, fn)) and tp+fp+tn+fn == n,
                     "Invalid binary confusion counts")
        else:
            tp = matrix[i][i]
            fn = sum(matrix[i]) - tp
            fp = sum(r[i] for r in matrix) - tp
            tn = n - tp - fn - fp
        sensitivity, specificity = _ratio(tp, tp+fn), _ratio(tn, tn+fp)
        precision = _ratio(tp, tp+fp)
        f1 = _ratio(2*tp, 2*tp+fp+fn)
        # Saved evaluators explicitly use zero for undefined precision/recall/F1.
        _close(saved["recall"], sensitivity or 0.0, "recall")
        _close(saved["precision"], precision or 0.0, "precision")
        _close(saved["f1" if multilabel else "f1-score"], f1 or 0.0, "F1")
        _close(saved["support"], tp+fn, "support")
        ba = (sensitivity + specificity)/2 if sensitivity is not None and specificity is not None else None
        if multilabel:
            _close(saved["specificity"], specificity, "specificity")
            _close(saved["balanced_accuracy"], ba, "binary balanced accuracy")
        rows.append(dict(common, label=name, n_samples=n, tp=tp, fp=fp, tn=tn, fn=fn,
                         support=tp+fn, negative_support=tn+fp, predicted_positive=tp+fp,
                         precision=saved["precision"], sensitivity=saved["recall"], specificity=specificity,
                         f1=saved["f1" if multilabel else "f1-score"], binary_balanced_accuracy=ba,
                         roc_auc=saved.get("roc_auc") if multilabel else metric.get("roc_auc_per_class", {}).get(name),
                         average_precision=saved.get("average_precision")))
    def mean(field: str) -> float | None:
        values = [r[field] for r in rows]
        return sum(values)/len(values) if all(v is not None for v in values) else None
    accuracy_definition = "exact label-set match" if multilabel else "correct single class / records"
    ba_definition = "mean over labels of (sensitivity + specificity) / 2" if multilabel else "mean class recall"
    ba = mean("binary_balanced_accuracy" if multilabel else "sensitivity")
    summary = dict(common, accuracy_definition=accuracy_definition, balanced_accuracy_definition=ba_definition)
    summary.update({key: metric.get(key) for key in METRICS})
    summary.update(accuracy=metric["subset_accuracy"] if multilabel else metric["accuracy"],
                   balanced_accuracy=ba if multilabel else metric["balanced_accuracy"],
                   macro_specificity=mean("specificity"),
                   macro_precision=metric.get("macro_precision", mean("precision")),
                   macro_recall=metric.get("macro_recall", mean("sensitivity")),
                   macro_roc_auc=metric["macro_roc_auc"] if multilabel else metric.get("roc_auc_ovr_macro"))
    _close(metric["macro_f1"], mean("f1"), "macro F1")
    if not multilabel:
        _close(metric["accuracy"], sum(matrix[i][i] for i in range(len(names)))/n, "accuracy")
        _close(metric["balanced_accuracy"], ba, "multiclass balanced accuracy")
    else:
        _close(metric["label_accuracy"], sum(r["tp"]+r["tn"] for r in rows)/(n*len(names)), "label accuracy")
    return summary, rows


def _verify(inputs: Inputs, directory: Path, run: dict, model_dir: Path) -> tuple[bool, str, list[str], str]:
    run_id = run["experiment_id"]
    paths = []
    checkpoint = model_dir / ("pipeline.joblib" if run.get("model") == "lda" else "best.pt")
    if run_id in OLD_RUNS:
        audit_path = inputs.root / AUDIT
        if not audit_path.exists():
            return False, "fresh historical audit unavailable", [], ""
        audit = inputs.json(audit_path)
        _require(not audit.get("failures"), "Historical audit records failures")
        entry = audit.get("runs", {}).get(run_id)
        if not entry:
            return False, "run absent from historical audit", [AUDIT], ""
        _require(entry.get("status") == "completed", "Audit run not completed")
        inputs.check(directory / "run.json", entry["run_sha256"])
        for partition in PARTITIONS:
            _require(entry["fresh_verification"]["partitions"][partition]["all_saved_metrics_recomputed"],
                     "Historical saved metrics were not verified")
            _require(run["metrics"][partition] == entry[f"{partition}_metrics"], "Historical audit metric mismatch")
        checkpoint_hash = entry["model_artifact_inventory"][checkpoint.name]["sha256"]
        inputs.check(checkpoint, checkpoint_hash)
        manifest_name, key = OLD_RUNS[run_id]
        manifest = inputs.json(inputs.root / manifest_name)[key]
        prefix = directory.relative_to(inputs.root).as_posix() + "/"
        protected = {}
        for name, value in manifest.items():
            if name.startswith(prefix):
                protected[name[len(prefix):]] = value["sha256"] if isinstance(value, dict) else value
        paths = [AUDIT, manifest_name]
    else:
        verification_path = directory / "verification.json"
        if not verification_path.exists():
            return False, "passing per-run verification unavailable", [], ""
        verification = inputs.json(verification_path)
        if verification.get("status") != "passed" or verification.get("failures"):
            return False, "per-run verification has not passed", [str(verification_path.relative_to(inputs.root))], ""
        _require(verification.get("run_id") == run_id, "Verification run ID mismatch")
        protected = verification.get("run_artifact_sha256", {})
        checkpoint_hash = verification["checkpoint_sha256"]
        inputs.check(checkpoint, checkpoint_hash)
        _require(run.get("checkpoint_sha256") == checkpoint_hash, "Run checkpoint hash mismatch")
        paths = [verification_path.relative_to(inputs.root).as_posix()]
    required = {"run.json", "config.json", "provenance.json"} | {f"{p}_{suffix}" for p in PARTITIONS for suffix in ("metrics.json", "predictions.csv")}
    _require(required.issubset(protected), f"Verification does not bind required artifacts: {run_id}: {required-set(protected)}")
    for name, digest in protected.items():
        path = inputs.path(name, directory)
        _require(path.is_relative_to(directory), "Artifact manifest escapes run directory")
        inputs.check(path, digest)
    return True, "", paths, checkpoint_hash


def _wall(start: str, end: str) -> float | None:
    if not start or not end or "unavailable" in (start, end):
        return None
    a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
    _require(a.tzinfo is not None and b.tzinfo is not None and b >= a, "Invalid run timestamps")
    return (b-a).total_seconds()


def collect(root: Path = ROOT) -> tuple[list[dict], list[dict], list[dict], Inputs, list[Path]]:
    inputs = Inputs(root)
    legacy = inputs.root / "experiments/experiment_log.csv"
    reader = csv.DictReader(io.StringIO(inputs.read(legacy).decode()))
    _require(reader.fieldnames == LEGACY_FIELDS, "Legacy log schema changed")
    entries = list(reader)
    _require(all(None not in e and all(v is not None for v in e.values()) for e in entries), "Malformed central log")
    _require(len({e['experiment_id'] for e in entries}) == len(entries), "Duplicate experiment IDs")
    for entry in entries:
        entry["central_log_status"] = entry["status"]
    # A terminal run.json can exist before its central-log append is observed.
    logged = {e["experiment_id"]: e.get("output_directory") for e in entries}
    for path in sorted((inputs.root / "results").glob("*/*/run.json")):
        run = inputs.json(path)
        if run.get("experiment_id") not in logged:
            entries.append(dict(experiment_id=run["experiment_id"], status=run["status"],
                                model=run.get("model", "unavailable"),
                                output_directory=path.parent.relative_to(inputs.root).as_posix()))
            logged[run["experiment_id"]] = path.parent.relative_to(inputs.root).as_posix()
        else:
            _require(logged[run["experiment_id"]] and
                     inputs.path(logged[run["experiment_id"]]) == path.parent.resolve(),
                     f"Duplicate run ID at different artifact paths: {run['experiment_id']}")
    # The runner saves provenance before training but only writes run.json on
    # completion/failure. Include these attempts without inferring liveness.
    for path in sorted((inputs.root / "results").glob("*/*/provenance.json")):
        if (path.parent / "run.json").exists():
            continue
        run_id = path.parent.name
        if run_id in logged:
            _require(logged[run_id] and inputs.path(logged[run_id]) == path.parent.resolve(),
                     f"Duplicate run ID at different artifact paths: {run_id}")
            continue
        provenance = inputs.json(path)
        config_path = path.parent / "config.json"
        config = inputs.json(config_path) if config_path.exists() else {}
        directory = path.parent.relative_to(inputs.root).as_posix()
        entries.append(dict(experiment_id=run_id, status="incomplete",
                            model=config.get("model", "unavailable"),
                            date=provenance.get("started_utc", "unavailable"),
                            dataset_version=config.get("dataset_id", config.get("dataset_version", "unavailable")),
                            split_version=config.get("split_version", "unavailable"),
                            hyperparameters=_json(config["training"]) if "training" in config else "unavailable",
                            output_directory=directory))
        logged[run_id] = directory
    master, comparison, labels, protected_dirs = [], [], [], []
    for entry in entries:
        run_id = entry["experiment_id"]
        directory = inputs.path(entry["output_directory"]) if entry.get("output_directory") else None
        run_path = directory / "run.json" if directory else None
        run = inputs.json(run_path) if run_path and run_path.exists() else {}
        if run:
            _require(run["experiment_id"] == run_id, "Log/run ID mismatch")
            _require(run["status"] == entry["status"], f"Log/run status mismatch: {run_id}")
        status = entry["status"]
        status_source = "run.json" if run else "experiments/experiment_log.csv"
        status_updated = run.get("completed_utc", run.get("failed_utc", "unavailable"))
        progress = {}
        if not run and directory and (directory / "provenance.json").exists():
            progress_path = directory / "evaluation_progress.json"
            progress = inputs.json(progress_path) if progress_path.exists() else {}
            if progress:
                _require(isinstance(progress.get("status"), str) and bool(progress["status"].strip()),
                         f"Invalid recorded progress status: {run_id}")
                status = progress["status"]
                status_source = progress_path.relative_to(inputs.root).as_posix()
                status_updated = progress.get("updated_utc") or "unavailable"
            elif not entry.get("central_log_status"):
                status = "incomplete"
                status_source = "provenance present; run.json and evaluation_progress.json absent"
        dataset = run.get("dataset_version", run.get("dataset_id", entry.get("dataset_version", "unavailable")))
        cohort = _cohort(dataset)
        training = run.get("training", {})
        config = inputs.json(directory / "config.json") if directory and (directory / "config.json").exists() else {}
        provenance = inputs.json(directory / "provenance.json") if directory and (directory / "provenance.json").exists() else {}
        model_dir = inputs.path(run["model_directory"]) if run.get("model_directory") else None
        if directory:
            protected_dirs.append(directory)
        if model_dir:
            protected_dirs.append(model_dir)
        passed, reason, verification_paths, checkpoint_hash = False, f"run status is {status}", [], ""
        if status == "completed" and run and model_dir:
            passed, reason, verification_paths, checkpoint_hash = _verify(inputs, directory, run, model_dir)
        elif status == "completed":
            reason = "completed central entry lacks run or model artifacts"
        if cohort == "unavailable":
            passed, reason = False, "unsupported or unavailable cohort"
        if passed:
            _require(not any((directory / n).exists() for n in ("failure.json", "logging_failure.json")),
                     f"Completed run has failure artifacts: {run_id}")
        is_lda = run.get("model", entry.get("model", "")).lower() == "lda"
        start = run.get("started_utc", entry.get("date") or "unavailable")
        end = run.get("completed_utc", "unavailable")
        legacy_notes = entry.get("notes", "").replace(
            "Single-label accuracy/balanced_accuracy columns intentionally blank",
            "In the legacy experiments/experiment_log.csv only, single-label accuracy/balanced_accuracy columns are intentionally blank",
        )
        limitations = [legacy_notes, run.get("comparison_limit", ""),
                       config.get("comparison_limit", ""), training.get("reproducibility_limit", "")]
        limitations.append("One configuration/seed; no external clinical validation.")
        if cohort == "Mendeley four-class":
            limitations.append("Printed-ID groups are not verified patient IDs; small test cohort with few MI cases.")
        elif cohort == "PTB-XL five-label":
            limitations.append("Clean pages rendered from real ECG recordings; fixed 0.5 threshold; repeated test-set observation makes comparisons exploratory.")
        else:
            limitations.append("Task cohort unavailable; no comparative claim.")
        if is_lda:
            limitations.append("Neural parameter and epoch counts do not apply to PCA/LDA.")
        if not passed:
            limitations.append(reason)
        if not run and directory and (directory / "provenance.json").exists():
            limitations.append("No terminal run.json is available. Status is the saved progress record, or incomplete when absent; process liveness has not been verified.")
        params = training.get("num_parameters", "unavailable") if passed else "unavailable"
        row = dict(experiment_id=run_id, cohort=cohort, model=entry.get("model", run.get("model", "unavailable")),
                   architecture=training.get("model", "PCA + shrinkage LDA" if is_lda else "unavailable"), status=status,
                   central_log_status=entry.get("central_log_status", ""), status_source=status_source,
                   status_last_updated_utc=status_updated, liveness_verified=False,
                   verification_status="verified" if passed else "not_verified", exclusion_reason=reason,
                   started_utc=start, completed_utc=end,
                   git_commit=provenance.get("git_commit") or run.get("git_commit") or "unavailable",
                   dataset_version=dataset, split_version=run.get("split_version", entry.get("split_version", "unavailable")),
                   class_names=_json(config.get("class_names", training.get("class_names", []))),
                   preprocessing=_json(config["preprocessing"]) if "preprocessing" in config else entry.get("preprocessing", "unavailable"),
                   normalization=_json(training.get("normalization", {})),
                   random_seed=training.get("config", {}).get("seed", training.get("seed", config.get("seed", entry.get("random_seed", "unavailable")))),
                   hyperparameters=_json(training["config"]) if "config" in training else entry.get("hyperparameters", "unavailable"),
                   parameters="NA" if is_lda else params,
                   parameter_definition="not applicable: non-neural PCA/LDA" if is_lda else "trainable neural parameters",
                   epochs_run="NA" if is_lda else training.get("epochs_run", "unavailable"),
                   best_epoch="NA" if is_lda else training.get("best_epoch", "unavailable"),
                   checkpoint_selection=training.get("checkpoint_selection", "NA: one prespecified fit" if is_lda else config.get("selection_rule", "unavailable")),
                   training_seconds=training.get("training_seconds", training.get("fit_seconds")),
                   training_module_seconds=training.get("runtime_seconds"), runner_elapsed_seconds=run.get("total_runtime_seconds"),
                   utc_wall_span_seconds=_wall(start, end), device=training.get("device", "unavailable"),
                   threshold=run.get("threshold", config.get("threshold", "NA")),
                   accuracy_definition="exact label-set match" if cohort == "PTB-XL five-label" else "correct single class / records",
                   balanced_accuracy_definition="mean over labels of (sensitivity + specificity) / 2" if cohort == "PTB-XL five-label" else "mean class recall",
                   test_inference_calls_recorded=run.get("test_inference_calls", "unavailable"),
                   limitations=" ".join(dict.fromkeys(s for s in limitations if s)),
                   output_directory=entry.get("output_directory", "unavailable"),
                   model_directory=run.get("model_directory", "unavailable"),
                   checkpoint_path=(model_dir / ("pipeline.joblib" if is_lda else "best.pt")).relative_to(inputs.root).as_posix() if model_dir else "unavailable",
                   verification_paths=_json(verification_paths),
                   run_sha256=inputs.hashes.get(run_path.relative_to(inputs.root).as_posix(), "unavailable") if run_path else "unavailable",
                   checkpoint_sha256=checkpoint_hash or "unavailable", recorded_source_sha256=_json(provenance.get("source_sha256", {})))
        if passed:
            for partition in PARTITIONS:
                metric = inputs.json(directory / f"{partition}_metrics.json")
                _require(metric == run["metrics"][partition], f"Run and standalone metrics differ: {run_id}/{partition}")
                summary, per_label = _metric_rows(run_id, row["model"], cohort, partition, metric)
                comparison.append(summary)
                labels.extend(per_label)
                row.update({f"{partition}_{key}": summary[key] for key in METRICS})
        master.append(row)
    return master, comparison, labels, inputs, protected_dirs


def _csv(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _display(value: Any) -> str:
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(master: list[dict], comparison: list[dict], labels: list[dict]) -> str:
    text = ["# Publication model comparison", "", "Metrics come only from saved, completed, verified runs. Incomplete and unverified attempts remain in the run inventory without metrics. Status and update times come from saved artifacts; process liveness is not checked and elapsed time never determines completion or failure. CSV files retain full floating-point precision; displayed metrics are rounded to six decimals. No fitting, model inference or resampling is performed. Cohorts are separate and are never ranked against one another.", "",
            "Accuracy means the correct single-class fraction for Mendeley and exact label-set match for PTB-XL. PTB-XL label accuracy is separate. Balanced accuracy means mean class recall for Mendeley and mean over labels of (sensitivity + specificity)/2 for PTB-XL. Undefined specificity or binary balanced accuracy remains blank in CSV; saved zero-division precision/recall/F1 policy is retained.", "",
            "Neural parameter counts exclude PCA/LDA (NA). Dates and git commits come from saved artifacts; unavailable commit IDs are not replaced with the current checkout. Training-loop, module, runner elapsed and UTC wall-span times are distinct; system suspension can enlarge UTC spans. Missing numeric observations are blank, while NA denotes inapplicable metadata.", ""]
    def table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
        text.extend(["| " + " | ".join(title for _, title in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"])
        text.extend("| " + " | ".join(_display(row.get(key)) for key, _ in columns) + " |" for row in rows)
        text.append("")
    for cohort in ("Mendeley four-class", "PTB-XL five-label"):
        text.extend([f"## {cohort}", ""])
        table([r for r in comparison if r["cohort"] == cohort and r["partition"] == "test"],
              [("model", "Model"), ("n_samples", "Test N"), ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
               ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1"), ("macro_roc_auc", "Macro AUROC"),
               ("macro_average_precision", "Macro AP"), ("label_accuracy", "Label accuracy")])
        table([r for r in master if r["cohort"] == cohort], [("model", "Model"), ("status", "Status"),
              ("verification_status", "Verification"), ("status_last_updated_utc", "Recorded status updated"),
              ("parameters", "Parameters"), ("best_epoch", "Selected epoch"),
              ("epochs_run", "Epochs"), ("training_seconds", "Training seconds"), ("runner_elapsed_seconds", "Runner seconds")])
        text.extend(["### Test sensitivity and specificity", ""])
        table([r for r in labels if r["cohort"] == cohort and r["partition"] == "test"],
              [("model", "Model"), ("label", "Label"), ("support", "Positive support"), ("sensitivity", "Sensitivity"),
               ("specificity", "Specificity"), ("precision", "Precision"), ("f1", "F1"), ("roc_auc", "AUROC")])
    text.extend(["## Run-specific limits and provenance", ""])
    for row in master:
        text.append(f"- **{row['experiment_id']}**: {row['limitations']} Artifacts: `{row['output_directory']}`; verification: `{row['verification_paths']}`.")
    text.extend(["", "All partition metrics are in `publication_model_comparison.csv`; all partition label confusions and sensitivity/specificity are in `publication_model_per_label_metrics.csv`. The expanded master log includes preprocessing, hyperparameters, timestamps, runtimes, sources and exclusions. `publication_model_comparison.provenance.json` binds input/output hashes. Original per-run artifacts and `experiment_log.csv` remain authoritative and unchanged.", ""])
    return "\n".join(text)


def build(root: Path = ROOT, output_dir: Path | None = None, master_log: Path | None = None) -> dict:
    root = root.resolve()
    output_dir = (output_dir or root / "results/comparisons").resolve()
    master_log = (master_log or root / "experiments/master_experiment_log.csv").resolve()
    master, comparison, labels, inputs, protected_dirs = collect(root)
    outputs = {
        master_log: _csv(master, MASTER_FIELDS),
        (output_dir / "publication_model_comparison.csv").resolve(): _csv(comparison, COMPARISON_FIELDS),
        (output_dir / "publication_model_per_label_metrics.csv").resolve(): _csv(labels, LABEL_FIELDS),
        (output_dir / "publication_model_comparison.md").resolve(): _markdown(master, comparison, labels),
    }
    manifest_path = (output_dir / "publication_model_comparison.provenance.json").resolve()
    for path in [*outputs, manifest_path]:
        _require(path != root / "experiments/experiment_log.csv", "Cannot overwrite legacy log")
        _require(not any(path.is_relative_to(d) for d in protected_dirs), "Cannot write inside immutable run/model directory")
        _require(not path.is_relative_to(root / "models") and not path.is_relative_to(root / "data"), "Cannot write inside models/data")
        _require(not path.exists() or path.is_file(), f"Output is not a file: {path}")
        _require(not path.is_relative_to(root) or path.relative_to(root).as_posix() not in inputs.hashes, "Cannot overwrite input")
    _require(len(set([*outputs, manifest_path])) == 5, "Output paths collide")
    # Reruns may replace this builder's own views, never unrelated existing files.
    owned = set()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(previous.get("schema_version") == 1 and previous.get("source_root") == str(root)
                 and "generator_sha256" in previous, "Existing provenance is not owned by this builder")
        owned = {Path(p).resolve() for p in previous.get("output_artifact_sha256", {})}
    for path in outputs:
        _require(not path.exists() or path in owned, f"Refusing to overwrite unrelated existing file: {path}")
    inputs.unchanged()
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    scope="Publication aggregation; saved verification certificates and artifact hashes checked; no inference or metric rerun from probabilities.",
                    generator_sha256=_sha(Path(__file__)), source_root=str(root),
                    source_artifact_sha256=dict(sorted(inputs.hashes.items())),
                    output_artifact_sha256={str(p): hashlib.sha256(t.encode()).hexdigest() for p, t in outputs.items()},
                    run_count=len(master), verified_completed_runs=[r["experiment_id"] for r in master if r["verification_status"] == "verified"],
                    excluded_runs=[{"experiment_id": r["experiment_id"], "status": r["status"], "reason": r["exclusion_reason"]} for r in master if r["verification_status"] != "verified"],
                    legacy_log_preserved=True, ranking_between_cohorts=False,
                    limitations=["Hash checks certify agreement with saved evidence, not independent clinical validation.",
                                 "No current git HEAD is substituted for an unrecorded run commit.",
                                 "Historical audit certifies saved prediction metrics; PTB-XL uses its per-run verification manifest."])
    outputs[manifest_path] = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    staged = []
    try:
        for path, text in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                             prefix=".publication-summary-", delete=False) as handle:
                handle.write(text)
                staged.append((Path(handle.name), path))
        inputs.unchanged()
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
    inputs.unchanged()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--master-log", type=Path)
    args = parser.parse_args()
    report = build(args.root, args.output_dir, args.master_log)
    print(json.dumps({k: report[k] for k in ("run_count", "verified_completed_runs", "excluded_runs")}, indent=2))


if __name__ == "__main__":
    main()
