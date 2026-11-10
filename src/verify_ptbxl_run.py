"""Independently verify a completed PTB-XL run using saved artifacts only.

``verify(run_id)`` is read-only and raises on missing or inconsistent artifacts.
The explicit command-line entry point writes verification.json after success.
Checkpoint tensors are inspected, never instantiated as a model or evaluated.
Bootstrap point estimates and structural validity are checked; its 500
resamples are not repeated. Saved inference bookkeeping cannot prove the
absence of unrecorded work outside this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np

from src.approved_source_revisions import current_source_sha256
from src.portable_cache import located, relocate
from src.evaluation.multilabel import evaluate_multilabel

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "experiments/configs/ptbxl_cnn_lstm_image_v1.json"
PRETRAINING_PATH = "experiments/protocols/ptbxl_cnn_lstm_image_v1/pretraining_checks.json"
PRIOR_PATH = "experiments/protocols/ptbxl_images_v1/prior_model_artifacts.json"
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
COUNTS = {"train": 17084, "validation": 2146, "test": 2158}
INPUT_SHAPE = [12, 120, 1120]
CODE_FILES = [
    "src/run_ptbxl_cnn_lstm.py", "src/models/ptbxl_cnn_lstm.py",
    "src/ptbxl_learning_data.py", "src/evaluation/multilabel.py",
    "src/evaluation/ptbxl_uncertainty.py", "src/common.py", "src/baseline_preservation.py",
]
SOURCE_FILES = [
    "protocol.json", "generation_state.json", "generation_summary.json", "verification.json",
    "all_images.csv", "classification_manifest.csv", "classification_exclusions.csv",
    "splits/train.csv", "splits/validation.csv", "splits/test.csv",
]
CI_METRICS = ["macro_roc_auc", "macro_average_precision", "macro_f1", "micro_f1",
              "subset_accuracy", "label_accuracy"]
TOLERANCE = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(path, expected):
    _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
             f"Invalid declared SHA-256: {path}")
    _require(Path(path).is_file() and _sha(path) == expected, f"SHA-256 mismatch: {path}")


def _safe_path(root, relative):
    _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
             "Expected a nonempty relative artifact path")
    candidate = (Path(root) / relative).resolve()
    _require(candidate.is_relative_to(Path(root).resolve()), f"Artifact escapes root: {relative}")
    return candidate


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant {value}: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_object,
                      parse_constant=invalid)


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    _require(fields and len(fields) == len(set(fields)), f"Missing/duplicate CSV columns: {path}")
    _require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
             f"Malformed CSV row: {path}")
    return fields, rows


def _compare(expected, actual, context="value"):
    """Strict structure and order, finite numeric equality within absolute 1e-12."""
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and set(expected) == set(actual), f"Key mismatch: {context}")
        for key in expected:
            _compare(expected[key], actual[key], f"{context}.{key}")
    elif isinstance(expected, (list, tuple)):
        _require(isinstance(actual, (list, tuple)) and len(expected) == len(actual),
                 f"Length/type mismatch: {context}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{context}[{index}]")
    elif isinstance(expected, Real) and not isinstance(expected, bool):
        _require(isinstance(actual, Real) and not isinstance(actual, bool)
                 and math.isfinite(expected) and math.isfinite(actual)
                 and math.isclose(expected, actual, rel_tol=0, abs_tol=TOLERANCE),
                 f"Numeric mismatch: {context}: expected {expected!r}, observed {actual!r}")
    else:
        _require(type(expected) is type(actual) and expected == actual, f"Value mismatch: {context}")


def _integer(value, context, minimum=0):
    if isinstance(value, str):
        _require(bool(re.fullmatch(r"0|[1-9][0-9]*", value)), f"Invalid integer: {context}")
        value = int(value)
    _require(type(value) is int and value >= minimum, f"Invalid integer: {context}")
    return value


def _number(value, context):
    _require(not isinstance(value, bool), f"Boolean numeric value: {context}")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid number: {context}") from exc
    _require(math.isfinite(result), f"Nonfinite number: {context}")
    return result


def _verify_predictions(path, manifest_rows):
    fields, rows = _csv(path)
    expected_fields = ["ecg_id", "patient_id", "image_path"] + [
        f"{prefix}_{name}" for name in CLASS_NAMES for prefix in ("truth", "probability", "prediction")]
    _compare(expected_fields, fields, f"{path.name} class order")
    _require(len(rows) == len(manifest_rows) and len(rows) > 0, f"Prediction row count mismatch: {path}")
    y, p = np.empty((len(rows), 5), np.int8), np.empty((len(rows), 5), np.float64)
    for index, (row, source) in enumerate(zip(rows, manifest_rows)):
        for field in ("ecg_id", "patient_id", "image_path"):
            _compare(str(source[field]), row[field], f"{path.name} row {index} {field}")
        for j, name in enumerate(CLASS_NAMES):
            target = source[f"label_{name}"]
            _require(target in ("True", "False"), f"Noncanonical target in manifest: {name}")
            truth = "1" if target == "True" else "0"
            _compare(truth, row[f"truth_{name}"], f"{path.name} row {index} truth {name}")
            y[index, j] = int(truth)
            probability = _number(row[f"probability_{name}"], f"{path.name} row {index} probability {name}")
            _require(0 <= probability <= 1, f"Probability outside [0,1]: {path.name}, row {index}")
            p[index, j] = probability
            _compare(str(int(probability >= .5)), row[f"prediction_{name}"],
                     f"{path.name} row {index} fixed-threshold prediction {name}")
    return y, p


def _verify_per_label(path, metrics):
    fields, rows = _csv(path)
    expected_fields = ["label", *metrics["per_label"][CLASS_NAMES[0]]]
    _compare(expected_fields, fields, f"{path.name} columns")
    _require(len(rows) == 5, f"Expected five per-label rows: {path}")
    for name, row in zip(CLASS_NAMES, rows):
        _compare(name, row["label"], f"{path.name} label order")
        for key, expected in metrics["per_label"][name].items():
            actual = None if row[key] == "" else _number(row[key], f"{path.name}.{name}.{key}")
            _compare(expected, actual, f"{path.name}.{name}.{key}")


def _verify_history(rows, training, final_validation, config):
    _require(1 <= len(rows) <= config["max_epochs"], "Invalid training history length")
    best_key, best_epoch, stale = None, 0, 0
    for index, row in enumerate(rows, start=1):
        _compare(index, _integer(row["epoch"], "history epoch", 1), "history epoch order")
        key = (_number(row["val_macro_roc_auc"], "history macro ROC-AUC"),
               _number(row["val_macro_average_precision"], "history macro AP"),
               -_number(row["val_binary_cross_entropy"], "history BCE"))
        _require(0 <= key[0] <= 1 and 0 <= key[1] <= 1 and key[2] <= 0, "Invalid selection metric range")
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, stale = key, index, 0
        else:
            stale += 1
        _compare(str(improved), row["checkpoint_improved"], f"history epoch {index} improvement")
        _compare(stale, _integer(row["epochs_without_selection_improvement"], "history stale"),
                 f"history epoch {index} patience")
        _require(stale < config["patience"] or index == len(rows), "History continues after early stopping")
        for key_name in ("train_online_bce", "epoch_seconds", "max_gradient_norm_before_clip", "learning_rate"):
            _require(_number(row[key_name], key_name) >= 0, f"Negative history {key_name}")
    _require(len(rows) == config["max_epochs"] or stale >= config["patience"], "Training ended before its stopping rule")
    for key, expected in (("best_epoch", best_epoch), ("epochs_run", len(rows)),
                          ("best_selection_key", list(best_key)),
                          ("stopped_early", len(rows) < config["max_epochs"])):
        _compare(expected, training[key], f"training.{key}")
    _compare(final_validation, training["best_validation_metrics"], "selected validation metrics")
    selected = rows[best_epoch - 1]
    scalar_metrics = {f"val_{key}": value for key, value in final_validation.items()
                      if isinstance(value, Real) and not isinstance(value, bool)}
    _compare(set(scalar_metrics), {key for key in selected if key.startswith("val_")}, "history validation columns")
    for key, value in scalar_metrics.items():
        _compare(value, _number(selected[key], key), f"selected history.{key}")
    _compare(_number(selected["train_online_bce"], "selected train BCE"),
             training["selected_epoch_online_train_bce"], "selected online training BCE")
    return {"best_epoch": best_epoch, "epochs_run": len(rows), "selection_key": list(best_key),
            "earliest_exact_tie_rule_verified": True}


def _verify_bootstrap(ci, metrics, patient_ids, config):
    _compare(CLASS_NAMES, ci["class_names"], "bootstrap class order")
    expected = {"method": "Patient-cluster nonparametric percentile bootstrap",
                "confidence_level": config["confidence_level"], "percentiles": [2.5, 97.5],
                "quantile_method": "linear", "n_resamples": config["resamples"], "seed": config["seed"],
                "decision_threshold": .5, "threshold_fitting_performed": False,
                "calibration_performed": False, "model_training_performed": False,
                "n_records": len(patient_ids), "n_unique_patients": len(set(patient_ids)),
                "patient_draws_per_resample": len(set(patient_ids))}
    for key, value in expected.items():
        _compare(value, ci[key], f"bootstrap.{key}")
    n = _integer(ci["n_resamples"], "bootstrap resamples", 1)
    cluster_sizes = Counter(patient_ids)
    distribution = {str(k): v for k, v in sorted(Counter(cluster_sizes.values()).items())}
    _compare(distribution, ci["patient_record_count_distribution"], "bootstrap patient cluster sizes")
    _compare(set(CI_METRICS), set(ci["metrics"]), "bootstrap metric names")
    for name in CI_METRICS:
        interval = ci["metrics"][name]
        _compare(metrics[name], interval["point_estimate"], f"bootstrap {name} point estimate")
        defined = _integer(interval["defined_resamples"], f"bootstrap {name} defined")
        undefined = _integer(interval["undefined_resamples"], f"bootstrap {name} undefined")
        _require(defined + undefined == n, f"Bootstrap replicate accounting mismatch: {name}")
        if defined >= 2:
            lo = _number(interval["ci_lower"], f"bootstrap {name} lower")
            hi = _number(interval["ci_upper"], f"bootstrap {name} upper")
            _require(0 <= lo <= hi <= 1, f"Invalid bootstrap CI bounds: {name}")
            _compare("defined", interval["interval_status"], f"bootstrap {name} status")
        else:
            _compare(None, interval["ci_lower"], f"bootstrap {name} lower")
            _compare(None, interval["ci_upper"], f"bootstrap {name} upper")
            _compare("insufficient_defined_resamples", interval["interval_status"], f"bootstrap {name} status")
    record_counts = ci["resampled_record_counts"]
    lo = _integer(record_counts["minimum"], "bootstrap record minimum", 1)
    hi = _integer(record_counts["maximum"], "bootstrap record maximum", 1)
    average = _number(record_counts["mean"], "bootstrap record mean")
    _require(len(cluster_sizes) * min(cluster_sizes.values()) <= lo <= average <= hi
             <= len(cluster_sizes) * max(cluster_sizes.values()), "Bootstrap record counts violate patient-cluster bounds")
    return {"point_estimates_verified": True, "parameters_and_bounds_verified": True,
            "intervals_recomputed": False, "resamples": n, "seed": ci["seed"],
            "n_unique_patients": len(cluster_sizes)}


def _verify_log(prior_path, log_path, run_id, cfg, run, metrics):
    old_bytes, current_bytes = prior_path.read_bytes(), log_path.read_bytes()
    _require(current_bytes.startswith(old_bytes), "Prior experiment log is not an exact byte prefix")
    old_fields, old_rows = _csv(prior_path)
    fields, rows = _csv(log_path)
    _compare(old_fields, fields, "experiment log schema")
    _compare(old_rows, rows[:len(old_rows)], "prior experiment log rows")
    ids = [row["experiment_id"] for row in rows]
    _require(len(ids) == len(set(ids)) and run_id not in [row["experiment_id"] for row in old_rows],
             "Duplicate experiment ID or current run present in prior log")
    matches = [row for row in rows if row["experiment_id"] == run_id]
    _require(len(matches) == 1, "Expected exactly one logged current experiment")
    row = matches[0]
    expected = {"status": "completed", "model": "PTBXL_Image_CNN_LSTM", "date": run["started_utc"],
                "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
                "random_seed": str(cfg["seed"]), "accuracy": "", "balanced_accuracy": "",
                "output_directory": f"results/ptbxl_cnn_lstm/{run_id}"}
    for key, value in expected.items():
        _compare(value, row[key], f"experiment log.{key}")
    _compare(cfg["training"], json.loads(row["hyperparameters"]), "logged training configuration")
    _compare(cfg["preprocessing"], json.loads(row["preprocessing"]), "logged preprocessing")
    for key in ("macro_f1", "weighted_f1"):
        _compare(metrics[key], _number(row[key], f"logged {key}"), f"experiment log.{key}")
    buffer = io.StringIO(newline="")
    csv.DictWriter(buffer, fieldnames=fields).writerow(row)
    completed_prefix = old_bytes + buffer.getvalue().encode("utf-8")
    _require(current_bytes.startswith(completed_prefix), "Current run is not the first appended row after its prior log snapshot")
    return {"prior_log_rows": len(old_rows), "appended_log_rows": len(rows) - len(old_rows),
            "current_run_occurrences": len(matches), "prior_byte_prefix_preserved": True,
            "single_label_accuracy_fields_blank": True, "current_log_sha256": _sha(log_path),
            "completed_run_log_sha256": hashlib.sha256(completed_prefix).hexdigest()}


def _verify_cache(root, cfg, provenance):
    cache, source = _safe_path(root, cfg["cache_root"]), _safe_path(root, cfg["source_dataset_root"])
    _compare({"protocol.json", "metadata.json", "cache_manifest.csv"}, set(provenance["cache_sha256"]), "cache hash names")
    for name, digest in provenance["cache_sha256"].items():
        _hash(cache / name, digest)
    protocol, metadata = _json(cache / "protocol.json"), _json(cache / "metadata.json")
    _compare("complete", metadata["status"], "cache completion")
    _require(not (cache / "failure.json").exists(), "Cache has a failure artifact")
    _compare(INPUT_SHAPE, metadata["input_shape"], "cache input shape")
    _compare(CLASS_NAMES, metadata["class_names"], "cache target order")
    _compare(COUNTS, metadata["counts"], "cache partition sizes")
    _compare(sum(COUNTS.values()), metadata["images"], "cache record count")
    binding = protocol["binding"]
    _compare(binding, metadata["binding"], "cache protocol binding")
    # The cache recorded the absolute folder of the computer that built it; map it into this project.
    _compare(str(source), str(relocate(binding["source_root"], root)), "cache source root")
    _compare(CLASS_NAMES, binding["class_names"], "cache binding target order")
    _compare(binding["preprocessing"], metadata["preprocessing"], "cache preprocessing")
    _compare(INPUT_SHAPE, binding["preprocessing"]["input_shape"], "cache preprocessing input shape")
    _compare(provenance["source_sha256"]["src/ptbxl_learning_data.py"], binding["code_sha256"], "cache source code binding")
    source_names = {"classification_manifest.csv", "protocol.json", "generation_summary.json", "verification.json", "generation_state.json"}
    _compare(source_names, set(binding["source_files_sha256"]), "cache source hash names")
    for name, digest in binding["source_files_sha256"].items():
        _hash(source / name, digest)
    _compare(cfg["source_manifest_sha256"], metadata["source_manifest_sha256"], "cache source manifest binding")
    _hash(cache / "cache_manifest.csv", metadata["cache_manifest_sha256"])
    _compare(sum(COUNTS.values()), metadata["source_png_hashes_checked_before_decoding"], "cache source hash bookkeeping")
    fields, rows = _csv(cache / "cache_manifest.csv")
    source_fields, source_rows = _csv(source / "classification_manifest.csv")
    _compare([*source_fields, "cache_path", "cache_sha256", "cache_pixel_sha256"], fields, "cache manifest columns")
    _require(len(rows) == len(source_rows) == sum(COUNTS.values()), "Cache/source cohort size mismatch")
    _compare(source_rows, [{k: row[k] for k in source_fields} for row in rows], "cache identities and source targets")
    split_rows, patient_folds, ids, bytes_checked = {name: [] for name in COUNTS}, defaultdict(set), [], 0
    for row in rows:
        ecg = _integer(row["ecg_id"], "ECG ID", 1)
        _integer(row["patient_id"], "patient ID", 1)
        fold = _integer(row["official_fold"], "official fold", 1)
        _require(fold <= 10, "Official fold exceeds 10")
        split = "train" if fold <= 8 else "validation" if fold == 9 else "test"
        _compare(split, row["split"], "official partition")
        labels = []
        for name in CLASS_NAMES:
            _require(row[f"label_{name}"] in ("True", "False"), "Invalid source label boolean")
            if row[f"label_{name}"] == "True":
                labels.append(name)
        _require(labels and row["eligible_for_classification"] == "True" and not row["exclusion_reason"], "Ineligible record entered cache")
        _compare(labels, json.loads(row["labels_json"]), "cache target labels")
        _compare(f"images/{split}/ecg_{ecg:05d}.png", row["image_path"], "source image path")
        _compare(f"cachedimages/{split}/ecg_{ecg:05d}.png", row["cache_path"], "cache image path")
        _require(re.fullmatch(r"[0-9a-f]{64}", row["cache_pixel_sha256"]), "Invalid cached pixel hash")
        image_path = _safe_path(cache, row["cache_path"])
        _hash(image_path, row["cache_sha256"])
        bytes_checked += image_path.stat().st_size
        ids.append(ecg)
        patient_folds[row["patient_id"]].add(fold)
        split_rows[split].append(row)
    _require(len(set(ids)) == len(ids) and ids == sorted(ids), "Cache ECG IDs are duplicated or out of source order")
    _require(all(len(folds) == 1 for folds in patient_folds.values()), "Patient crosses official folds")
    _compare(COUNTS, {name: len(part) for name, part in split_rows.items()}, "cache split counts")
    expected_cache_paths = {row["cache_path"] for row in rows}
    _compare(expected_cache_paths, {path.relative_to(cache).as_posix() for path in (cache / "cachedimages").rglob("*.png")}, "cache image inventory")
    norm = metadata["normalization"]
    _compare("train", norm["fit_partition"], "normalization fit partition")
    _compare(COUNTS["train"], norm["training_images"], "normalization training images")
    _compare(COUNTS["train"] * math.prod(INPUT_SHAPE), norm["training_pixels"], "normalization pixel count")
    _require(0 <= _number(norm["mean"], "normalization mean") <= 1 and _number(norm["raw_std"], "normalization raw std") >= 0,
             "Invalid normalization moments")
    _compare(1e-6, norm["std_floor"], "normalization std floor")
    _compare(max(norm["raw_std"], norm["std_floor"]), norm["std"], "normalization std")
    _compare(norm, provenance["normalization"], "run normalization")
    return split_rows, norm, {"encoded_images_hashed": len(rows), "encoded_image_bytes_hashed": bytes_checked,
                             "decoded_pixels_recomputed": False, "metadata_hashes_verified": 3,
                             "partition_records": COUNTS, "unique_patients": len(patient_folds)}


def _verify_protection(root, out, source):
    prior = _json(root / PRIOR_PATH)["sha256"]
    log_name = "experiments/experiment_log.csv"
    old = {name: digest for name, digest in prior.items() if name != log_name}
    _require(len(old) == 79, "Expected the 79 protected artifacts of the three previous model runs")
    expected = {**old, **{(source / name).relative_to(root).as_posix(): _sha(source / name) for name in SOURCE_FILES}}
    protected = _json(out / "protected_artifacts.json")
    _compare(expected, protected, "protected original model and dataset artifact declarations")
    for name, digest in protected.items():
        _hash(_safe_path(root, name), current_source_sha256(name, digest))
    prior_log = out / "prior_experiment_log.csv"
    _hash(prior_log, prior[log_name])
    lock = _json(root / "experiments/reviews/2026-09-24/baseline_lock.json")["files"]
    for name, digest in lock.items():
        if name != log_name:
            _hash(_safe_path(root, name), current_source_sha256(name, digest))
    baseline_log = root / "experiments/protocols/cnn_lstm_image_v1/baseline_log_before.csv"
    _hash(baseline_log, lock[log_name])
    _require(prior_log.read_bytes().startswith(baseline_log.read_bytes()), "Earlier baseline log prefix changed")
    preservation = _json(out / "preservation.json")
    for key, value in {"protected_artifacts_checked": 89, "changed_artifacts": [],
                       "prior_three_model_rows_preserved_as_exact_byte_prefix": True,
                       "prior_log_sha256": _sha(prior_log)}.items():
        _compare(value, preservation[key], f"preservation.{key}")
    baseline = preservation["baseline_preservation"]
    for key, value in {"immutable_baseline_artifacts_checked": len(lock) - 1,
                       "changed_baseline_artifacts": [], "original_log_sha256": lock[log_name],
                       "original_log_byte_prefix_preserved": True,
                       "original_log_rows_preserved": len(_csv(baseline_log)[1])}.items():
        _compare(value, baseline[key], f"preservation.baseline.{key}")
    current = (root / log_name).read_bytes()
    _require(current.startswith(prior_log.read_bytes()), "Prior log byte prefix changed")
    return {"previous_model_artifacts_hashed": len(old), "source_dataset_artifacts_hashed": len(SOURCE_FILES),
            "original_baseline_artifacts_hashed": len(lock) - 1, "changed_artifacts": []}


def _timestamp(value, context):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp: {context}") from exc
    _require(result.tzinfo is not None, f"Timestamp lacks timezone: {context}")
    return result


def verify(run_id, *, root=ROOT):
    """Verify completed saved artifacts; return a JSON-safe report without writes."""
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id), "Invalid run ID")
    root = Path(root).resolve()
    out = root / "results/ptbxl_cnn_lstm" / run_id
    model_dir = root / "models/ptbxl_cnn_lstm" / run_id
    run, cfg, provenance = _json(out / "run.json"), _json(out / "config.json"), _json(out / "provenance.json")
    _compare("completed", run["status"], "run status")
    _compare(run_id, run["experiment_id"], "run identity")
    for parent, names in ((out, ["failure.json", "logging_failure.json"]),
                          (model_dir, [".training.lock", ".best.pt.tmp"])):
        _require(not any((parent / name).exists() for name in names), f"Failure/unfinished artifacts remain in {parent}")
    for name in ("best.pt", "config.json", "architecture.txt", "history.csv", "training_curve.png", "metadata.json", "progress.json"):
        _require((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0,
                 f"Missing/empty completed model artifact: {name}")
    for key, value in {"model": "ptbxl_image_cnn_lstm", "class_names": CLASS_NAMES,
                       "input_shape": INPUT_SHAPE, "threshold": .5, "seed": 42}.items():
        _compare(value, cfg[key], f"configuration.{key}")
    _compare(42, cfg["training"]["seed"], "training seed")
    _compare(_json(root / CONFIG_PATH), cfg, "saved and original configuration")
    _hash(root / CONFIG_PATH, provenance["configuration_sha256"])
    pretraining = _json(root / PRETRAINING_PATH)
    sources = provenance["source_sha256"]
    _compare(set(CODE_FILES), set(sources), "declared training source files")
    _compare({**sources, CONFIG_PATH: provenance["configuration_sha256"]},
             pretraining["source_and_config_sha256"], "pretraining source/configuration freeze")
    _compare(False, pretraining["test_predictions_accessed"], "pretraining test-access record")
    snapshots = {path.relative_to(out / "source_snapshot").as_posix() for path in (out / "source_snapshot").rglob("*") if path.is_file()}
    _compare(set(CODE_FILES), snapshots, "source snapshot inventory")
    for name, digest in sources.items():
        _hash(root / name, current_source_sha256(name, digest))
        _hash(out / "source_snapshot" / name, digest)
    for key in ("selection_rule", "test_policy"):
        _compare(cfg[key], provenance[key], f"provenance.{key}")
    _compare(run["started_utc"], provenance["started_utc"], "run start timestamp")
    source = _safe_path(root, cfg["source_dataset_root"])
    for name, key in [("classification_manifest.csv", "source_manifest_sha256"),
                      ("verification.json", "source_verification_sha256"), ("protocol.json", "source_protocol_sha256")]:
        _hash(source / name, cfg[key])
    for key in ("model", "dataset_id", "split_version", "threshold", "uncertainty"):
        _compare(cfg[key], run[key], f"run.{key}")
    _compare(cfg["sequence_interpretation"], run["representation"], "run representation")
    _compare(cfg["comparison_limit"], run["comparison_limit"], "run comparison limit")
    _compare(model_dir.relative_to(root).as_posix(), run["model_directory"], "model directory")
    split_rows, normalization, cache_report = _verify_cache(root, cfg, provenance)
    _compare(split_rows["train"] + split_rows["validation"], _csv(out / "development_split_used.csv")[1], "development manifest")
    metrics, probabilities, truths = {}, {}, {}
    _compare(set(COUNTS), set(run["metrics"]), "run metric partitions")
    for split, rows in split_rows.items():
        y, p = _verify_predictions(out / f"{split}_predictions.csv", rows)
        metrics[split] = evaluate_multilabel(y, p, CLASS_NAMES, threshold=.5)
        truths[split], probabilities[split] = y, p
        _compare(metrics[split], _json(out / f"{split}_metrics.json"), f"{split} saved metrics")
        _compare(metrics[split], run["metrics"][split], f"run {split} metrics")
        _verify_per_label(out / f"{split}_per_label.csv", metrics[split])
        _require((out / f"{split}_confusion_matrices.png").is_file(), f"Missing {split} confusion plot")
    training = _json(out / "training_metadata.json")
    _compare(training, run["training"], "run training metadata")
    _compare(training, _json(model_dir / "metadata.json"), "model training metadata")
    _compare(cfg["training"], _json(model_dir / "config.json"), "model training configuration")
    train_counts = truths["train"].sum(axis=0).tolist()
    training_expected = {"model": "LeadImageCNNLSTM", "config": cfg["training"], "input_shape": INPUT_SHAPE,
                         "class_names": CLASS_NAMES, "normalization": normalization,
                         "training_records": COUNTS["train"], "validation_records": COUNTS["validation"],
                         "class_counts_train": train_counts, "test_data_accessed": False,
                         "class_weighting": "none", "augmentation": "none", "pretrained_weights": False,
                         "resume_supported": False, "device": "cpu",
                         "checkpoint_path": located(training["checkpoint_path"], model_dir / "best.pt", root),
                         "history_path": located(training["history_path"], model_dir / "history.csv", root)}
    for key, value in training_expected.items():
        _compare(value, training[key], f"training.{key}")
    history_report = _verify_history(_csv(model_dir / "history.csv")[1], training, metrics["validation"], cfg["training"])
    progress = _json(model_dir / "progress.json")
    for key, value in {"status": "complete", "phase": "complete", "epochs_run": training["epochs_run"],
                       "best_epoch": training["best_epoch"], "test_data_accessed": False}.items():
        _compare(value, progress[key], f"training progress.{key}")
    checkpoint_hash = run["checkpoint_sha256"]
    _hash(model_dir / "best.pt", checkpoint_hash)
    # Metadata/tensor inspection only: do not construct a model, loader or predictor.
    import torch
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    checkpoint_expected = {"format_version": 1, "model_name": "LeadImageCNNLSTM", "class_names": CLASS_NAMES,
                           "input_shape": INPUT_SHAPE, "config": cfg["training"], "normalization": normalization,
                           "training_records": COUNTS["train"], "class_counts_train": train_counts,
                           "best_epoch": training["best_epoch"], "validation_selection_key": training["best_selection_key"],
                           "best_validation_metrics": metrics["validation"], "selection_rule": training["checkpoint_selection"],
                           "sequence_interpretation": training["sequence_interpretation"], "resume_supported": False}
    for key, value in checkpoint_expected.items():
        _compare(value, checkpoint[key], f"checkpoint.{key}")
    state = checkpoint["state_dict"]
    _require(isinstance(state, dict) and state and all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
             for value in state.values()), "Checkpoint has missing/nonfinite tensors")
    _compare(training["num_parameters"], sum(value.numel() for value in state.values()), "checkpoint parameter count")
    frozen, replay, test_state = (_json(out / name) for name in
                                  ("selection_frozen.json", "checkpoint_replay.json", "test_inference_state.json"))
    for key, value in {"checkpoint_sha256": checkpoint_hash, "configuration_sha256": provenance["configuration_sha256"],
                       "selection_rule": cfg["selection_rule"], "threshold": .5, "training": training,
                       "test_inference_started": False}.items():
        _compare(value, frozen[key], f"selection freeze.{key}")
    _compare({"split": "train", "records": 32, "maximum_absolute_probability_difference": 0.0,
              "passed": True, "test_replay_performed": False, "checkpoint_sha256": checkpoint_hash}, replay, "checkpoint replay")
    _require(replay["maximum_absolute_probability_difference"] == 0.0,
             "Checkpoint replay difference must be exactly zero")
    for key, value in {"status": "completed", "checkpoint_sha256": checkpoint_hash, "threshold": .5,
                       "inference_calls": 1, "records": COUNTS["test"]}.items():
        _compare(value, test_state[key], f"test inference state.{key}")
    _compare(1, run["test_inference_calls"], "run test inference count")
    times = [(run["started_utc"], "run start"), (frozen["frozen_utc"], "checkpoint freeze"),
             (test_state["started_utc"], "test start"), (test_state["completed_utc"], "test complete"),
             (run["completed_utc"], "run complete")]
    parsed_times = [_timestamp(value, context) for value, context in times]
    _require(parsed_times == sorted(parsed_times), "Checkpoint freeze/test/run timestamps are out of order")
    _require(_number(run["total_runtime_seconds"], "run duration") > 0, "Run runtime must be positive")
    bootstrap = _verify_bootstrap(_json(out / "test_patient_bootstrap.json"), metrics["test"],
                                  [row["patient_id"] for row in split_rows["test"]], cfg["uncertainty"])
    protection = _verify_protection(root, out, source)
    log = _verify_log(out / "prior_experiment_log.csv", root / "experiments/experiment_log.csv",
                      run_id, cfg, run, metrics["test"])
    baseline_preservation = _json(out / "preservation.json")["baseline_preservation"]
    _compare(log["completed_run_log_sha256"], baseline_preservation["current_log_sha256"],
             "historical completed-run log hash")
    _compare(log["prior_log_rows"] + 1 - baseline_preservation["original_log_rows_preserved"],
             baseline_preservation["appended_log_rows"], "historical appended log rows")
    summary_fields, summary_rows = _csv(out / "run_summary.csv")
    _compare(["run_id", "dataset", "task", *CI_METRICS], summary_fields, "run summary columns")
    _require(len(summary_rows) == 1, "Expected one run summary row")
    for key, value in {"run_id": run_id, "dataset": cfg["dataset_id"], "task": "five-label"}.items():
        _compare(value, summary_rows[0][key], f"run summary.{key}")
    for key in CI_METRICS:
        _compare(metrics["test"][key], _number(summary_rows[0][key], f"run summary {key}"), f"run summary.{key}")
    # Recheck immutable training bindings after the potentially longer cache audit.
    _hash(root / CONFIG_PATH, provenance["configuration_sha256"])
    _hash(model_dir / "best.pt", checkpoint_hash)
    for name, digest in sources.items():
        _hash(root / name, current_source_sha256(name, digest))
    run_hashes = {path.relative_to(out).as_posix(): _sha(path) for path in sorted(out.rglob("*"))
                  if path.is_file() and path.name != "verification.json"}
    return {"status": "passed", "run_id": run_id, "verified_utc": datetime.now(timezone.utc).isoformat(),
            "verifier_sha256": _sha(Path(__file__)), "model_inference_performed": False,
            "bootstrap_resampling_performed": False, "metric_absolute_tolerance": TOLERANCE,
            "failures": [], "class_names": CLASS_NAMES, "checkpoint_sha256": checkpoint_hash,
            "configuration_sha256": provenance["configuration_sha256"],
            "source_files_verified": len(sources), "source_snapshots_verified": len(sources),
            "cache": cache_report, "prediction_records_verified": COUNTS,
            "metrics_recomputed": True, "confusions_and_per_label_metrics_verified": True,
            "fixed_threshold_decisions_verified": True, "patient_fold_disjointness_verified": True,
            "selection": history_report, "checkpoint_replay": replay,
            "test_inference_bookkeeping_verified": {"status": "completed", "inference_calls": 1},
            "uncertainty": bootstrap, "preservation": protection, "experiment_log": log,
            "run_artifact_sha256": run_hashes,
            "scope": "Saved-artifact consistency and current file integrity. No model inference or new clinical validation. "
                     "Inference counts are checked against saved bookkeeping; unrecorded activity cannot be ruled out. "
                     "Bootstrap intervals have structural/point-estimate checks and are not numerically recomputed. "
                     "Cached PNG encoded hashes are rechecked; their pixel arrays are not decoded again."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    report = verify(arguments.run_id)
    destination = ROOT / "results/ptbxl_cnn_lstm" / arguments.run_id / "verification.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "run_id": arguments.run_id,
                      "verification": str(destination), "prediction_records_verified": report["prediction_records_verified"]}, indent=2))


if __name__ == "__main__":
    main()
