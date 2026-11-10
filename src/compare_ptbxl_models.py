"""Compare two completed, verified PTB-XL models using saved probabilities.

No image decoding, checkpoint loading, inference, fitting or threshold search
occurs. Paired patient-cluster resampling estimates test differences conditional
on the two fitted models. The previously inspected test cohort makes this an
exploratory comparison, not a fresh confirmatory model-selection experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
import re
import tempfile

import numpy as np

from src.evaluation.multilabel import evaluate_multilabel
from src.evaluation.ptbxl_uncertainty import _patient_groups, _sample_record_indices

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
PARTITIONS = ("train", "validation", "test")
METRICS = ("macro_roc_auc", "macro_average_precision", "macro_f1", "micro_f1",
           "subset_accuracy", "label_accuracy")
METRIC_LABELS = {"macro_roc_auc": "Macro ROC-AUC (primary)",
                 "macro_average_precision": "Macro average precision",
                 "macro_f1": "Macro F1", "micro_f1": "Micro F1",
                 "subset_accuracy": "Exact-match accuracy", "label_accuracy": "Label accuracy"}
OLD_RUN = "ptbxl_cnn_lstm_image_v1_seed42"
NEW_RUN = "ptbxl_spatiotemporal_image_v1_seed42"
ARTIFACTS = ("comparison.json", "partition_metrics.csv", "per_label_metrics.csv", "metric_differences.csv",
             "comparison.md", "comparison.png", "comparison.svg")
LIMITATION = (
    "Exploratory comparison: the test cohort was inspected for the previous model before this next model was developed. "
    "Patient-bootstrap intervals quantify sampling uncertainty conditional on these two fitted models, their fixed 0.5 "
    "threshold and this observed cohort. They exclude training-seed variability, architecture-search/adaptive-selection "
    "uncertainty and external-domain generalization. No multiplicity adjustment or bias correction is applied. "
    "Test outcomes do not revise either checkpoint or the separately reported validation ranking."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root, relative):
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Expected a relative artifact path")
    path = (Path(root) / relative).resolve()
    _require(path.is_relative_to(Path(root).resolve()), f"Artifact path escapes its root: {relative}")
    return path


def _same(expected, actual, name):
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(actual) == set(expected), f"Key mismatch: {name}")
        for key, value in expected.items():
            _same(value, actual[key], f"{name}.{key}")
    elif isinstance(expected, (list, tuple)):
        _require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"Sequence mismatch: {name}")
        for index, (a, b) in enumerate(zip(expected, actual)):
            _same(a, b, f"{name}[{index}]")
    elif isinstance(expected, Real) and not isinstance(expected, bool):
        _require(isinstance(actual, Real) and not isinstance(actual, bool) and math.isfinite(actual)
                 and math.isfinite(expected) and math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12),
                 f"Numeric mismatch: {name}")
    else:
        _require(type(expected) is type(actual) and expected == actual, f"Value mismatch: {name}")


class _Inputs:
    def __init__(self):
        self.hashes = {}

    def check_hash(self, path, expected):
        _require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), f"Invalid SHA-256: {path}")
        actual = _sha(path)
        _require(actual == expected, f"Verified artifact hash changed: {path}")
        self.hashes[str(Path(path).resolve())] = actual

    def text(self, path):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        previous = self.hashes.get(str(path.resolve()))
        _require(previous is None or previous == digest, f"Artifact changed while reading: {path}")
        self.hashes[str(path.resolve())] = digest
        return raw.decode("utf-8")

    def json(self, path):
        def invalid(value):
            raise ValueError(f"Nonfinite JSON constant: {value}")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, f"Duplicate JSON key in {path}: {key}")
                result[key] = value
            return result
        return json.loads(self.text(path), parse_constant=invalid, object_pairs_hook=unique)

    def csv(self, path):
        reader = csv.DictReader(io.StringIO(self.text(path), newline=""))
        fields, rows = reader.fieldnames, list(reader)
        _require(fields and len(fields) == len(set(fields)), f"Missing/duplicate CSV columns: {path}")
        _require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
                 f"Malformed CSV rows: {path}")
        return fields, rows

    def unchanged(self):
        for path, digest in self.hashes.items():
            _require(_sha(path) == digest, f"Comparison input changed during execution: {path}")


def _prediction_arrays(fields, rows, split):
    expected_fields = ["ecg_id", "patient_id", "image_path"] + [
        f"{field}_{label}" for label in CLASS_NAMES for field in ("truth", "probability", "prediction")]
    _same(expected_fields, fields, f"{split} prediction class/column order")
    _require(bool(rows), f"Empty {split} predictions")
    ids = [row["ecg_id"] for row in rows]
    _require(len(set(ids)) == len(ids), f"Repeated ECG ID in {split}")
    _require(all(re.fullmatch(r"[1-9][0-9]*", row[key]) for row in rows for key in ("ecg_id", "patient_id")),
             f"Invalid/noncanonical patient or ECG ID in {split}")
    y = np.asarray([[float(row[f"truth_{label}"]) for label in CLASS_NAMES] for row in rows])
    p = np.asarray([[float(row[f"probability_{label}"]) for label in CLASS_NAMES] for row in rows])
    metrics = evaluate_multilabel(y, p, CLASS_NAMES, threshold=.5)
    for index, row in enumerate(rows):
        for j, label in enumerate(CLASS_NAMES):
            _same(str(int(y[index, j])), row[f"truth_{label}"], f"{split} truth format")
            _same(str(int(p[index, j] >= .5)), row[f"prediction_{label}"], f"{split} fixed-threshold decision")
    return {"rows": rows, "truth": y.astype(np.int8), "probabilities": p, "metrics": metrics}


def _run_timing(run, training):
    """Keep recorded perf-counter durations separate from timestamp wall span."""
    values = {"training_recorded_timer_seconds": training["runtime_seconds"],
              "run_recorded_timer_seconds": run["total_runtime_seconds"]}
    for name, value in values.items():
        _require(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0,
                 f"Invalid recorded elapsed timer: {name}")
    timestamps = []
    for name in ("started_utc", "completed_utc"):
        value = run[name]
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run timestamp: {name}") from exc
        _require(parsed.tzinfo is not None and parsed.utcoffset() is not None,
                 f"Run timestamp requires an explicit timezone: {name}")
        timestamps.append(parsed.astimezone(timezone.utc))
    span = (timestamps[1] - timestamps[0]).total_seconds()
    _require(span >= 0, "Run completion timestamp precedes its start")
    return {**values, "run_utc_wall_span_seconds": span,
            "run_started_utc": timestamps[0].isoformat(), "run_completed_utc": timestamps[1].isoformat()}


def _load_run(root, family, run_id, inputs):
    for value in (family, run_id):
        _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value), "Invalid family or run identifier")
    out = root / "results" / family / run_id
    verification = inputs.json(out / "verification.json")
    _require(verification.get("status") == "passed" and verification.get("run_id") == run_id,
             "Comparison requires a matching passed verification for each run")
    _require(verification.get("failures") == [], "Verification reports failures")
    declared = verification["run_artifact_sha256"]
    _require(isinstance(declared, dict) and declared, "Verification lacks artifact hashes")
    required = {"run.json", "config.json", "provenance.json", "training_metadata.json", "selection_frozen.json",
                "test_inference_state.json", *[f"{split}_{suffix}" for split in PARTITIONS
                                                for suffix in ("predictions.csv", "metrics.json")]}
    _require(required <= set(declared), "Verification does not bind all comparison inputs")
    for name, digest in declared.items():
        _require(name != "verification.json", "Verification cannot contain its own hash")
        inputs.check_hash(_safe_path(out, name), digest)
    run = inputs.json(out / "run.json")
    _require(run.get("status") == "completed" and run.get("experiment_id") == run_id,
             "Comparison requires completed matching models")
    _require(not any((out / name).exists() for name in ("failure.json", "logging_failure.json")), "Run contains failure artifacts")
    cfg = inputs.json(out / "config.json")
    provenance = inputs.json(out / "provenance.json")
    training = inputs.json(out / "training_metadata.json")
    frozen = inputs.json(out / "selection_frozen.json")
    test_state = inputs.json(out / "test_inference_state.json")
    _require(all(item.get("threshold") == .5 and not isinstance(item.get("threshold"), bool)
                 for item in (cfg, run, frozen, test_state)), "Every protocol and artifact must use exactly threshold 0.5")
    _same(CLASS_NAMES, cfg["class_names"], "configured class order")
    _same([12, 120, 1120], cfg["input_shape"], "configured input shape")
    _same(.5, cfg["threshold"], "configured threshold")
    for key in ("model", "dataset_id", "split_version", "threshold"):
        _same(cfg[key], run[key], f"run.{key}")
    _same(training, run["training"], "saved training metadata")
    _same(cfg["training"], training["config"], "training configuration")
    _same(training, frozen["training"], "frozen training metadata")
    _same(provenance["normalization"], training["normalization"], "normalization")
    _same("train", training["normalization"]["fit_partition"], "normalization fit partition")
    _same(CLASS_NAMES, training["class_names"], "training target order")
    _same(.5, frozen["threshold"], "frozen threshold")
    _same(False, frozen["test_inference_started"], "checkpoint freeze before test")
    _same(cfg["selection_rule"], provenance["selection_rule"], "selection provenance")
    _same(cfg["selection_rule"], frozen["selection_rule"], "frozen selection rule")
    _same(verification["checkpoint_sha256"], run["checkpoint_sha256"], "verified checkpoint")
    _same(run["checkpoint_sha256"], frozen["checkpoint_sha256"], "frozen checkpoint")
    _same(verification["configuration_sha256"], provenance["configuration_sha256"], "verified configuration")
    _same(provenance["configuration_sha256"], frozen["configuration_sha256"], "frozen configuration")
    _same(1, run["test_inference_calls"], "run test inference count")
    for key, value in {"status": "completed", "inference_calls": 1, "threshold": .5,
                       "checkpoint_sha256": run["checkpoint_sha256"]}.items():
        _same(value, test_state[key], f"test inference bookkeeping.{key}")
    model_dir = _safe_path(root, run["model_directory"])
    _same(root / "models" / family / run_id, model_dir, "model directory")
    inputs.check_hash(model_dir / "best.pt", run["checkpoint_sha256"])
    _require(type(training["num_parameters"]) is int and training["num_parameters"] > 0, "Invalid parameter count")
    _run_timing(run, training)
    _same({"protocol.json", "metadata.json", "cache_manifest.csv"}, set(provenance["cache_sha256"]), "cache binding files")
    for name, digest in provenance["cache_sha256"].items():
        inputs.check_hash(_safe_path(root, cfg["cache_root"]) / name, digest)
    # Metric and cluster definitions must be the same frozen implementations used
    # by both runs. Training/model source is already covered by each verifier.
    for name in ("src/evaluation/multilabel.py", "src/evaluation/ptbxl_uncertainty.py"):
        inputs.check_hash(root / name, provenance["source_sha256"][name])
    partitions = {}
    _same(set(PARTITIONS), set(run["metrics"]), "run metric partitions")
    for split in PARTITIONS:
        fields, rows = inputs.csv(out / f"{split}_predictions.csv")
        part = _prediction_arrays(fields, rows, split)
        _same(part["metrics"], inputs.json(out / f"{split}_metrics.json"), f"{split} saved metrics")
        _same(part["metrics"], run["metrics"][split], f"{split} run metrics")
        partitions[split] = part
    _same(len(partitions["test"]["rows"]), test_state["records"], "test record count")
    _same(partitions["validation"]["metrics"], training["best_validation_metrics"], "selected validation metrics")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("ecg_id", "patient_id"):
            left = {row[field] for row in partitions[first]["rows"]}
            right = {row[field] for row in partitions[second]["rows"]}
            _require(not left & right, f"{field} crosses {first}/{second}")
    return {"family": family, "run_id": run_id, "out": out, "run": run, "config": cfg,
            "provenance": provenance, "training": training, "partitions": partitions,
            "verification_sha256": _sha(out / "verification.json")}


def _align(old, new):
    for key in ("dataset_id", "split_version", "source_dataset_root", "cache_root", "source_manifest_sha256",
                "source_verification_sha256", "source_protocol_sha256", "class_names", "input_shape",
                "preprocessing", "threshold", "selection_rule"):
        _same(old["config"][key], new["config"][key], f"shared protocol.{key}")
    for key in ("cache_sha256", "normalization"):
        _same(old["provenance"][key], new["provenance"][key], f"shared input.{key}")
    for split in PARTITIONS:
        a, b = old["partitions"][split], new["partitions"][split]
        _require(len(a["rows"]) == len(b["rows"]), f"{split} sample counts differ")
        for index, (left, right) in enumerate(zip(a["rows"], b["rows"])):
            for key in ("ecg_id", "patient_id", "image_path", *[f"truth_{label}" for label in CLASS_NAMES]):
                _same(left[key], right[key], f"{split} paired alignment row {index}.{key}")
        _require(np.array_equal(a["truth"], b["truth"]), f"{split} targets differ")


def _difference(old, new):
    return None if old is None or new is None else float(new - old)


def _interval_interpretation(lower, upper):
    if lower is None or upper is None:
        return "not_estimable"
    if lower <= 0 <= upper:
        return "includes_zero_no_resolved_difference"
    return "above_zero_new_higher_conditional" if lower > 0 else "below_zero_old_higher_conditional"


def paired_patient_bootstrap(y, old_probabilities, new_probabilities, patient_ids,
                             class_names, n_resamples=500, seed=2024):
    """Return paired new-minus-old differences and 95% patient-bootstrap CIs.

    Each replicate draws patients once and applies the identical full-cluster
    record indices, including repeated records, to both models. Undefined metric
    pairs are omitted and counted; fewer than two defined pairs produce null CIs.
    """
    for name, value, minimum in (("n_resamples", n_resamples, 1), ("seed", seed, 0)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and value >= minimum, f"{name} must be an integer >= {minimum}")
    n_resamples, seed = int(n_resamples), int(seed)
    truth, old_p, new_p = np.asarray(y), np.asarray(old_probabilities), np.asarray(new_probabilities)
    names = list(class_names)
    old_point = evaluate_multilabel(truth, old_p, names, threshold=.5)
    new_point = evaluate_multilabel(truth, new_p, names, threshold=.5)
    groups = _patient_groups(patient_ids, len(truth))
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in METRICS}
    record_counts = []
    for _ in range(n_resamples):
        indices, _ = _sample_record_indices(groups, rng)
        record_counts.append(len(indices))
        before = evaluate_multilabel(truth[indices], old_p[indices], names, threshold=.5)
        after = evaluate_multilabel(truth[indices], new_p[indices], names, threshold=.5)
        for name in METRICS:
            value = _difference(before[name], after[name])
            if value is not None:
                _require(math.isfinite(value), f"Nonfinite paired metric difference: {name}")
                draws[name].append(value)
    intervals = {}
    for name, values in draws.items():
        if len(values) >= 2:
            lower, upper = [float(x) for x in np.quantile(values, [.025, .975], method="linear")]
        else:
            lower, upper = None, None
        intervals[name] = {"old_point_estimate": old_point[name], "new_point_estimate": new_point[name],
                           "difference": _difference(old_point[name], new_point[name]),
                           "ci_lower": lower, "ci_upper": upper, "defined_resamples": len(values),
                           "undefined_resamples": n_resamples - len(values),
                           "interval_status": "defined" if len(values) >= 2 else "insufficient_defined_resamples",
                           "interpretation": _interval_interpretation(lower, upper)}
    return {"method": "Paired patient-cluster percentile bootstrap", "difference_direction": "new_minus_old",
            "primary_metric": "macro_roc_auc", "class_names": names, "decision_threshold": .5,
            "confidence_level": .95, "percentiles": [2.5, 97.5], "quantile_method": "linear",
            "n_resamples": n_resamples, "seed": seed, "seed_role": "Uncertainty only; local NumPy generator",
            "n_records": len(truth), "n_unique_patients": len(groups), "patient_draws_per_resample": len(groups),
            "patient_record_count_distribution": {str(k): v for k, v in sorted(Counter(len(x) for x in groups).items())},
            "resampled_record_counts": {"minimum": min(record_counts), "maximum": max(record_counts),
                                        "mean": float(np.mean(record_counts))},
            "metrics": intervals,
            "pairing_policy": "One patient draw per replicate; include every record with patient-draw multiplicity, using identical indices for both models.",
            "undefined_policy": "Omit a metric pair if either score is undefined; count omissions. Full-five-label ROC-AUC/AP policies are unchanged. At least two defined pairs are needed for an interval.",
            "interval_interpretation": "An interval including or touching zero does not resolve a test difference. Intervals excluding zero are conditional exploratory cohort evidence, not an overall model-winner claim.",
            "limitations": LIMITATION, "model_inference_performed": False, "model_training_performed": False,
            "threshold_fitting_performed": False, "calibration_performed": False}


def _selection_key(metrics):
    values = [metrics[key] for key in ("macro_roc_auc", "macro_average_precision", "binary_cross_entropy")]
    _require(all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x) for x in values),
             "Validation ranking requires defined finite AUC, AP and BCE")
    return (float(values[0]), float(values[1]), -float(values[2]))


def _summary(old, new, bootstrap):
    old_key = _selection_key(old["partitions"]["validation"]["metrics"])
    new_key = _selection_key(new["partitions"]["validation"]["metrics"])
    ranking = "new_higher" if new_key > old_key else "old_higher" if old_key > new_key else "exact_tie"
    models = {}
    for role, data in (("old", old), ("new", new)):
        models[role] = {"run_id": data["run_id"], "family": data["family"], "model": data["run"]["model"],
                        "num_parameters": data["training"]["num_parameters"], "seed": data["config"]["seed"],
                        "best_epoch": data["training"]["best_epoch"], "epochs_run": data["training"]["epochs_run"],
                        **_run_timing(data["run"], data["training"]),
                        "checkpoint_sha256": data["run"]["checkpoint_sha256"],
                        "verification_sha256": data["verification_sha256"],
                        "configuration_sha256": data["provenance"]["configuration_sha256"],
                        "training_configuration": data["config"]["training"]}
    partitions = {}
    for split in PARTITIONS:
        a, b = old["partitions"][split], new["partitions"][split]
        partitions[split] = {"n_records": len(a["rows"]), "n_unique_patients": len({r["patient_id"] for r in a["rows"]}),
                             "old": a["metrics"], "new": b["metrics"],
                             "differences": {metric: _difference(a["metrics"][metric], b["metrics"][metric]) for metric in METRICS}}
    return {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "models": models, "class_names": CLASS_NAMES, "threshold": .5, "primary_metric": "macro_roc_auc",
            "difference_direction": "new_minus_old", "partitions": partitions, "paired_test_bootstrap": bootstrap,
            "validation_ranking": {"ordering": ranking, "old_key": list(old_key), "new_key": list(new_key),
                                   "criterion": "Lexicographic (validation macro ROC-AUC, macro AP, -probability BCE)",
                                   "interpretation": "Descriptive validation ordering of separately checkpoint-selected runs; independent of the test comparison. No checkpoint is changed."},
            "test_comparison_exploratory": True, "previous_model_test_previously_inspected": True,
            "same_record_order_labels_patient_ids_verified": True, "same_cache_preprocessing_folds_threshold_verified": True,
            "model_inference_performed": False, "limitations": LIMITATION,
            "accuracy_definitions": {"subset_accuracy": "Fraction of ECGs with all five label decisions correct",
                                     "label_accuracy": "Fraction of all record-label decisions correct"},
            "runtime_definitions": {
                "training_recorded_timer_seconds": "Saved training_metadata.json runtime_seconds: elapsed time.perf_counter timer for the training routine, including its overhead.",
                "run_recorded_timer_seconds": "Saved run.json total_runtime_seconds: recorded elapsed time.perf_counter timer for the run.",
                "run_utc_wall_span_seconds": "Difference between the saved run completed_utc and started_utc timestamps, normalized to UTC."},
            "runtime_caveat": "Recorded elapsed timers are not pure CPU time and can account for host sleep differently from UTC timestamps. "
                              "The UTC wall span includes intervening sleep and pauses and can be affected by clock adjustments. "
                              "Host sleep, thermals and competing load can differ between runs; these durations support no speed or efficiency conclusion.",
            "implementation_sha256": {path.relative_to(ROOT).as_posix(): _sha(path) for path in
                                      (Path(__file__), ROOT / "src/evaluation/multilabel.py", ROOT / "src/evaluation/ptbxl_uncertainty.py")}}


def _tables(result):
    summary, per_label, differences = [], [], []
    for split, part in result["partitions"].items():
        for role in ("old", "new"):
            model = result["models"][role]
            summary.append({"split": split, "model_role": role, "run_id": model["run_id"],
                            "n_records": part["n_records"], "n_unique_patients": part["n_unique_patients"],
                            **{name: part[role][name] for name in METRICS},
                            **{name: model[name] for name in ("num_parameters", "seed", "best_epoch", "epochs_run",
                                                             "training_recorded_timer_seconds", "run_recorded_timer_seconds",
                                                             "run_utc_wall_span_seconds")}})
            for label in CLASS_NAMES:
                per_label.append({"split": split, "model_role": role, "run_id": model["run_id"], "label": label,
                                  **{name: part[role]["per_label"][label][name] for name in ("precision", "recall", "f1", "support")}})
        for name in METRICS:
            ci = result["paired_test_bootstrap"]["metrics"][name] if split == "test" else {}
            differences.append({"split": split, "metric": name, "old": part["old"][name], "new": part["new"][name],
                                "new_minus_old": part["differences"][name],
                                **{key: ci.get(key) for key in ("ci_lower", "ci_upper", "defined_resamples", "undefined_resamples", "interpretation")}})
    return summary, per_label, differences


def _format(value, signed=False):
    return "undefined" if value is None else format(value, "+.4f" if signed else ".4f")


def _markdown(result):
    old, new = result["models"]["old"], result["models"]["new"]
    resamples, seed = (result["paired_test_bootstrap"][key] for key in ("n_resamples", "seed"))
    primary = result["paired_test_bootstrap"]["metrics"]["macro_roc_auc"]
    interval = f"[{_format(primary['ci_lower'], True)}, {_format(primary['ci_upper'], True)}]"
    finding = {"includes_zero_no_resolved_difference": "The primary interval includes zero, so this comparison does not resolve a test macro ROC-AUC difference.",
               "above_zero_new_higher_conditional": "The primary interval lies above zero: the new model has higher test macro ROC-AUC under this conditional, exploratory patient-bootstrap comparison.",
               "below_zero_old_higher_conditional": "The primary interval lies below zero: the previous model has higher test macro ROC-AUC under this conditional, exploratory patient-bootstrap comparison.",
               "not_estimable": "The primary paired interval is not estimable."}[primary["interpretation"]]
    lines = ["# Exploratory PTB-XL model comparison", "",
             f"Previous: `{old['run_id']}`. New: `{new['run_id']}`.", "",
             f"**Primary test difference (new − previous): macro ROC-AUC {_format(primary['difference'], True)}; 95% paired patient-bootstrap interval {interval}.** {finding}", "",
             result["limitations"], "",
             "Both completed runs passed artifact verification. ECG IDs, patient IDs, truth labels and row order match exactly in all three partitions; cache, preprocessing, official split and threshold bindings match. Only saved predictions were evaluated.", "",
             "## Validation ordering (separate from test)", "",
             f"Ordering: **{result['validation_ranking']['ordering'].replace('_', ' ')}**, using {result['validation_ranking']['criterion']}. "
             + result["validation_ranking"]["interpretation"], "",
             "## Partition metrics", "",
             "Exact-match accuracy requires all five labels correct for an ECG. Label accuracy counts correct individual record-label decisions. Labels overlap; predictions are independent probabilities at the fixed 0.5 threshold.", "",
             "| Partition | Model | Macro ROC-AUC | Macro AP | Macro F1 | Micro F1 | Exact-match accuracy | Label accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split, part in result["partitions"].items():
        for role in ("old", "new"):
            lines.append(f"| {split} | {'Previous' if role == 'old' else 'New'} | " + " | ".join(_format(part[role][k]) for k in METRICS) + " |")
    lines += ["", "## Paired test differences", "",
              f"{resamples} patient-cluster resamples, seed {seed}, linear 2.5th/97.5th percentiles. Each patient draw contributes every ECG, including full multiplicity for repeat draws, to both models. Undefined paired replicates are omitted and counted separately for each metric.", "",
              f"| Metric | New − previous | 95% paired interval | Defined / {resamples} |", "|---|---:|---|---:|"]
    for name, item in result["paired_test_bootstrap"]["metrics"].items():
        lines.append(f"| {METRIC_LABELS[name]} | {_format(item['difference'], True)} | [{_format(item['ci_lower'], True)}, {_format(item['ci_upper'], True)}] | {item['defined_resamples']} |")
    lines += ["", "## Per-label decisions at 0.5", "",
              "| Partition | Label | Previous precision | New precision | Previous recall | New recall | Previous F1 | New F1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split, part in result["partitions"].items():
        for label in CLASS_NAMES:
            values = [part[role]["per_label"][label][metric] for metric in ("precision", "recall", "f1") for role in ("old", "new")]
            lines.append(f"| {split} | {label} | " + " | ".join(_format(x) for x in values) + " |")
    lines += ["", "## Model size and recorded durations", "",
              "Training and run timers retain the recorded elapsed `time.perf_counter` values. The UTC wall span is computed separately from the saved run start and completion timestamps.", "",
              "| Model | Parameters | Selected epoch / epochs run | Training recorded timer (s) | Run recorded timer (s) | Run UTC wall span (s) | Seed |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for role, model in result["models"].items():
        lines.append(f"| {'Previous' if role == 'old' else 'New'} | {model['num_parameters']:,} | {model['best_epoch']} / {model['epochs_run']} | {model['training_recorded_timer_seconds']:.1f} | {model['run_recorded_timer_seconds']:.1f} | {model['run_utc_wall_span_seconds']:.1f} | {model['seed']} |")
    lines += ["", result["runtime_caveat"], "",
              f"![Test metrics and paired differences](<{Path(result['output_directory']) / 'comparison.png'}>)", "",
              "Machine-readable files: comparison.json, partition_metrics.csv, per_label_metrics.csv and metric_differences.csv.", ""]
    return "\n".join(lines)


def _plot(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]}, constrained_layout=True)
    x = np.arange(3)
    for role, offset, color, label in (("old", -.18, "#537B99", "Previous CNN–LSTM"),
                                       ("new", .18, "#B36A36", "New spatiotemporal")):
        values = [result["partitions"]["test"][role][name] for name in METRICS[:3]]
        left.bar(x + offset, [np.nan if value is None else value for value in values], width=.36, color=color, label=label)
    left.set(xticks=x, xticklabels=["Macro ROC-AUC\n(primary)", "Macro AP", "Macro F1"],
             ylim=(0, 1.05), ylabel="Test score", title="Saved test scores")
    left.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(.5, 1.01))
    left.set_title("Saved test scores", pad=42)
    left.grid(axis="y", alpha=.15)
    right.axvline(0, color="0.5", linestyle="--", linewidth=1)
    for index, name in enumerate(METRICS):
        item = result["paired_test_bootstrap"]["metrics"][name]
        point, lower, upper = item["difference"], item["ci_lower"], item["ci_upper"]
        color = "#313B52" if name == "macro_roc_auc" else "#788392"
        if lower is not None:
            right.hlines(index, lower, upper, color=color, linewidth=2)
            right.vlines([lower, upper], index - .09, index + .09, color=color, linewidth=1)
        if point is not None:
            right.plot(point, index, "o", color=color, markersize=6)
        else:
            right.text(0, index, " undefined", fontsize=9, va="center")
    right.set(yticks=np.arange(6), yticklabels=[METRIC_LABELS[name] for name in METRICS],
              xlabel="New − previous score (95% paired patient-bootstrap interval)", title="Paired test differences")
    right.invert_yaxis()
    right.grid(axis="x", alpha=.15)
    fig.suptitle("Exploratory PTB-XL comparison | same patients and labels | fixed threshold 0.5", fontsize=13)
    bootstrap = result["paired_test_bootstrap"]
    fig.supxlabel(f"{bootstrap['n_resamples']} patient-cluster resamples, seed {bootstrap['seed']}. Conditional on two fitted models; excludes training-seed variability.", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"comparison.{extension}", dpi=180)
    plt.close(fig)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(old_run_id=OLD_RUN, new_run_id=NEW_RUN, *, root=ROOT,
                 old_family="ptbxl_cnn_lstm", new_family="ptbxl_spatiotemporal",
                 output_dir=None, overwrite=False):
    """Write a verified offline comparison; refuse existing outputs by default.

    Uses the fixed comparison protocol of 500 resamples and seed 2024. The lower
    level paired_patient_bootstrap function accepts smaller synthetic test runs.
    Returns the JSON-safe comparison report. Never invokes a run verifier/model.
    """
    root = Path(root).resolve()
    for value in (old_run_id, new_run_id, old_family, new_family):
        _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value), "Invalid model/run identifier")
    output = (Path(output_dir) if output_dir is not None else root / "results/comparisons" / new_run_id).resolve()
    for family, run_id in ((old_family, old_run_id), (new_family, new_run_id)):
        for prefix in ("results", "models"):
            _require(not output.is_relative_to((root / prefix / family / run_id).resolve()), "Comparison output cannot modify either run directory")
    if output.exists():
        _require(output.is_dir(), "Comparison output path is not a directory")
        present = {path.name for path in output.iterdir()}
        if present and not overwrite:
            raise FileExistsError(f"Comparison output exists: {output}; use --overwrite only to replace this comparison's artifacts")
        _require(present <= set(ARTIFACTS), "Output directory contains unrelated files; choose a dedicated comparison directory")
    inputs = _Inputs()
    old = _load_run(root, old_family, old_run_id, inputs)
    new = _load_run(root, new_family, new_run_id, inputs)
    _align(old, new)
    before, after = old["partitions"]["test"], new["partitions"]["test"]
    bootstrap = paired_patient_bootstrap(before["truth"], before["probabilities"], after["probabilities"],
                                          [row["patient_id"] for row in before["rows"]], CLASS_NAMES,
                                          n_resamples=500, seed=2024)
    result = _summary(old, new, bootstrap)
    result["source_artifact_sha256"] = dict(inputs.hashes)
    result["output_directory"] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ptbxl-comparison-", dir=output.parent) as temporary:
        stage = Path(temporary)
        summary, per_label, differences = _tables(result)
        for name, rows in (("partition_metrics.csv", summary), ("per_label_metrics.csv", per_label), ("metric_differences.csv", differences)):
            _write_csv(stage / name, rows)
        (stage / "comparison.md").write_text(_markdown(result), encoding="utf-8")
        _plot(result, stage)
        inputs.unchanged()
        result["output_artifact_sha256"] = {name: _sha(stage / name) for name in ARTIFACTS if name != "comparison.json"}
        (stage / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACTS:
            if not overwrite and (output / name).exists():
                raise FileExistsError(f"Comparison output appeared during generation: {output / name}")
        for name in ARTIFACTS:
            (stage / name).replace(output / name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-id", default=OLD_RUN)
    parser.add_argument("--new-run-id", default=NEW_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = compare_runs(args.old_run_id, args.new_run_id, output_dir=args.output_dir, overwrite=args.overwrite)
    print(json.dumps({"status": result["status"], "output_directory": result["output_directory"],
                      "primary_test_difference": result["paired_test_bootstrap"]["metrics"]["macro_roc_auc"]}, indent=2))


if __name__ == "__main__":
    main()
