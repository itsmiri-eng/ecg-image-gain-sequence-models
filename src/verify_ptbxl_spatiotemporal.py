"""Offline verification of the separate PTB-XL CNN + causal-TCN image run.

The read-only verify() function checks saved evidence; only the CLI writes a
new verification.json. It never trains, constructs a model, evaluates a
checkpoint, decodes image pixels, or repeats bootstrap resampling. Shared
stateless readers/metric checks are reused without changing the earlier
verifier or running its model-specific verification entry point.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

from src.approved_source_revisions import current_source_sha256
from src.portable_cache import located
from src.evaluation.multilabel import evaluate_multilabel
from src.verify_ptbxl_run import (
    CLASS_NAMES, COUNTS, INPUT_SHAPE, SOURCE_FILES, CI_METRICS, TOLERANCE,
    _require, _sha, _hash, _safe_path, _json, _csv, _compare, _integer, _number,
    _timestamp, _verify_predictions, _verify_per_label, _verify_history,
    _verify_bootstrap, _verify_cache,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "experiments/configs/ptbxl_spatiotemporal_image_v1.json"
PRETRAINING_PATH = "experiments/protocols/ptbxl_spatiotemporal_image_v1/pretraining_checks.json"
PRIOR_PATH = "experiments/protocols/ptbxl_spatiotemporal_image_v1/prior_artifacts.json"
PRIOR_CONFIG_PATH = "experiments/configs/ptbxl_cnn_lstm_image_v1.json"
PRIOR_RUN_ID = "ptbxl_cnn_lstm_image_v1_seed42"
CODE_FILES = [
    "src/run_ptbxl_spatiotemporal.py", "src/models/ptbxl_spatiotemporal.py",
    "src/ptbxl_learning_data.py", "src/evaluation/multilabel.py",
    "src/evaluation/ptbxl_uncertainty.py", "src/common.py", "src/baseline_preservation.py",
    "src/models/ptbxl_cnn_lstm.py",
]
HEAD_CONFIGURATION = {"hidden_channels": 32, "kernel_size": 3, "dilations": [1, 2],
                      "convolutions_per_block": 2, "feature_receptive_field": 13}


def _verify_protocol(cfg, root):
    """The next architecture keeps the earlier task/input/selection protocol."""
    old = _json(Path(root) / PRIOR_CONFIG_PATH)
    for key in ("dataset_id", "source_dataset_root", "cache_root", "split_version", "class_names",
                "input_shape", "source_manifest_sha256", "source_verification_sha256",
                "source_protocol_sha256", "seed", "preprocessing", "loss", "selection_rule",
                "threshold", "threshold_policy", "uncertainty"):
        _compare(old[key], cfg[key], f"unchanged comparison protocol.{key}")
    old_shared = {key: value for key, value in old["training"].items() if key != "hidden_size"}
    new_shared = {key: value for key, value in cfg["training"].items()
                  if key not in {"hidden_channels", "dilations", "kernel_size"}}
    _compare(old_shared, new_shared, "unchanged training hyperparameters")
    for key, value in {"seed": 42, "max_epochs": 20, "patience": 4, "hidden_channels": 32,
                       "dilations": [1, 2], "kernel_size": 3}.items():
        _compare(value, cfg["training"][key], f"prespecified training.{key}")
    for key, value in {"feature_sequence": [70, 96], "head_receptive_field_columns": 13,
                       "streaming_causality": False}.items():
        _compare(value, cfg["architecture"][key], f"image-feature architecture.{key}")


def _verify_log(prior_path, log_path, run_id, cfg, run, metrics):
    old_bytes, current_bytes = prior_path.read_bytes(), log_path.read_bytes()
    _require(current_bytes.startswith(old_bytes), "Prior four-model log is not an exact byte prefix")
    old_fields, old_rows = _csv(prior_path)
    fields, rows = _csv(log_path)
    _require(len(old_rows) == 4, "Expected four prior model rows")
    _compare(old_fields, fields, "experiment log schema")
    _compare(old_rows, rows[:4], "prior four experiment rows")
    _require(all(row["status"] == "completed" for row in old_rows), "Prior model run is not completed")
    ids = [row["experiment_id"] for row in rows]
    _require(len(ids) == len(set(ids)) and run_id not in ids[:4], "Duplicate experiment ID")
    matches = [row for row in rows if row["experiment_id"] == run_id]
    _require(len(matches) == 1, "Expected exactly one logged current experiment")
    row = matches[0]
    expected = {"status": "completed", "model": "PTBXL_Image_Spatiotemporal", "date": run["started_utc"],
                "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
                "random_seed": str(cfg["seed"]), "accuracy": "", "balanced_accuracy": "",
                "output_directory": f"results/ptbxl_spatiotemporal/{run_id}"}
    for key, value in expected.items():
        _compare(value, row[key], f"experiment log.{key}")
    _compare(cfg["training"], json.loads(row["hyperparameters"]), "logged training configuration")
    _compare(cfg["preprocessing"], json.loads(row["preprocessing"]), "logged preprocessing")
    for key in ("macro_f1", "weighted_f1"):
        _compare(metrics[key], _number(row[key], f"logged {key}"), f"experiment log.{key}")
    buffer = io.StringIO(newline="")
    csv.DictWriter(buffer, fieldnames=fields).writerow(row)
    completed_prefix = old_bytes + buffer.getvalue().encode("utf-8")
    _require(current_bytes.startswith(completed_prefix), "Current run is not the first append after four prior rows")
    return {"prior_log_rows": 4, "appended_log_rows": len(rows) - 4,
            "current_run_occurrences": 1, "prior_byte_prefix_preserved": True,
            "single_label_accuracy_fields_blank": True, "current_log_sha256": _sha(log_path),
            "completed_run_log_sha256": hashlib.sha256(completed_prefix).hexdigest()}


def _verify_protection(root, out, source):
    prior = _json(root / PRIOR_PATH)["sha256"]
    log_name = "experiments/experiment_log.csv"
    _require(log_name in prior, "Prior-artifact lock omits the four-row experiment log")
    old = {name: digest for name, digest in prior.items() if name != log_name}
    _require(bool(old), "Empty prior-artifact inventory")
    first_three = _json(root / "experiments/protocols/ptbxl_images_v1/prior_model_artifacts.json")["sha256"]
    for name, digest in first_three.items():
        if name != log_name:
            _compare(digest, old.get(name), f"original three-model artifact declaration.{name}")
    previous_out = root / "results/ptbxl_cnn_lstm" / PRIOR_RUN_ID
    previous_model = root / "models/ptbxl_cnn_lstm" / PRIOR_RUN_ID
    required_previous = {path.relative_to(root).as_posix() for directory in (previous_out, previous_model)
                         for path in directory.rglob("*") if path.is_file()}
    _require(bool(required_previous) and required_previous <= set(old),
             "Prior lock omits existing CNN-LSTM result/model artifacts")
    previous_provenance = _json(previous_out / "provenance.json")
    for name, digest in previous_provenance["source_sha256"].items():
        _compare(digest, old.get(name), f"prior frozen source declaration.{name}")
    _compare(previous_provenance["configuration_sha256"], old.get(PRIOR_CONFIG_PATH), "prior model config declaration")
    previous_cache = _safe_path(root, _json(root / PRIOR_CONFIG_PATH)["cache_root"])
    for name, digest in previous_provenance["cache_sha256"].items():
        relative = (previous_cache / name).relative_to(root).as_posix()
        _compare(digest, old.get(relative), f"prior cache declaration.{name}")
    expected = {**old, **{(source / name).relative_to(root).as_posix(): _sha(source / name) for name in SOURCE_FILES}}
    protected = _json(out / "protected_artifacts.json")
    _compare(expected, protected, "protected prior model and source artifact declarations")
    for name, digest in protected.items():
        _hash(_safe_path(root, name), current_source_sha256(name, digest))
    prior_log = out / "prior_experiment_log.csv"
    _hash(prior_log, prior[log_name])
    _require(len(_csv(prior_log)[1]) == 4, "Prior log snapshot must contain four model rows")
    lock = _json(root / "experiments/reviews/2026-09-24/baseline_lock.json")["files"]
    for name, digest in lock.items():
        if name != log_name:
            _hash(_safe_path(root, name), current_source_sha256(name, digest))
    baseline_log = root / "experiments/protocols/cnn_lstm_image_v1/baseline_log_before.csv"
    _hash(baseline_log, lock[log_name])
    _require(prior_log.read_bytes().startswith(baseline_log.read_bytes()), "Original baseline log prefix changed")
    preservation = _json(out / "preservation.json")
    for key, value in {"protected_artifacts_checked": len(expected), "changed_artifacts": [],
                       "prior_four_model_rows_preserved_as_exact_byte_prefix": True,
                       "prior_log_sha256": _sha(prior_log)}.items():
        _compare(value, preservation[key], f"preservation.{key}")
    baseline = preservation["baseline_preservation"]
    for key, value in {"immutable_baseline_artifacts_checked": len(lock) - 1,
                       "changed_baseline_artifacts": [], "original_log_sha256": lock[log_name],
                       "original_log_byte_prefix_preserved": True,
                       "original_log_rows_preserved": len(_csv(baseline_log)[1])}.items():
        _compare(value, baseline[key], f"preservation.baseline.{key}")
    _require((root / log_name).read_bytes().startswith(prior_log.read_bytes()), "Prior log byte prefix changed")
    return {"previous_model_artifacts_hashed": len(old), "source_dataset_artifacts_hashed": len(SOURCE_FILES),
            "original_baseline_artifacts_hashed": len(lock) - 1, "changed_artifacts": []}


def verify(run_id, *, root=ROOT):
    """Verify completed saved artifacts; return a JSON-safe report without writes."""
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id), "Invalid run ID")
    root = Path(root).resolve()
    out = root / "results/ptbxl_spatiotemporal" / run_id
    model_dir = root / "models/ptbxl_spatiotemporal" / run_id
    run, cfg, provenance = _json(out / "run.json"), _json(out / "config.json"), _json(out / "provenance.json")
    _compare("completed", run["status"], "run status")
    _compare(run_id, run["experiment_id"], "run identity")
    for parent, names in ((out, ["failure.json", "logging_failure.json"]),
                          (model_dir, [".training.lock", ".best.pt.tmp"])):
        _require(not any((parent / name).exists() for name in names), f"Failure/unfinished artifacts remain in {parent}")
    for name in ("best.pt", "config.json", "architecture.txt", "history.csv", "training_curve.png", "metadata.json", "progress.json"):
        _require((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0,
                 f"Missing/empty completed model artifact: {name}")
    for key, value in {"model": "ptbxl_image_spatiotemporal", "class_names": CLASS_NAMES,
                       "input_shape": INPUT_SHAPE, "threshold": .5, "seed": 42}.items():
        _compare(value, cfg[key], f"configuration.{key}")
    _verify_protocol(cfg, root)
    _compare(_json(root / CONFIG_PATH), cfg, "saved and original configuration")
    _hash(root / CONFIG_PATH, provenance["configuration_sha256"])
    pretraining = _json(root / PRETRAINING_PATH)
    sources = provenance["source_sha256"]
    _compare(set(CODE_FILES), set(sources), "declared training source files")
    _compare({**sources, CONFIG_PATH: provenance["configuration_sha256"]},
             pretraining["source_and_config_sha256"], "pretraining source/configuration freeze")
    _compare(False, pretraining["test_predictions_accessed"], "pretraining test-access record")
    _hash(root / PRIOR_PATH, pretraining["prior_artifacts_sha256"])
    _compare(cfg["prior_artifacts_sha256"], pretraining["prior_artifacts_sha256"], "frozen prior-artifact inventory")
    _compare(cfg["prior_artifacts_sha256"], provenance["prior_artifacts_sha256"], "run prior-artifact inventory")
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
    training_expected = {"model": "LeadImageSpatiotemporal", "config": cfg["training"], "input_shape": INPUT_SHAPE,
                         "class_names": CLASS_NAMES, "normalization": normalization,
                         "training_records": COUNTS["train"], "validation_records": COUNTS["validation"],
                         "class_counts_train": train_counts, "test_data_accessed": False,
                         "class_weighting": "none", "augmentation": "none", "pretrained_weights": False,
                         "resume_supported": False, "device": "cpu",
                         "checkpoint_path": located(training["checkpoint_path"], model_dir / "best.pt", root),
                         "history_path": located(training["history_path"], model_dir / "history.csv", root)}
    for key, value in training_expected.items():
        _compare(value, training[key], f"training.{key}")
    for key, value in {"sequence_shape": [70, 96], "temporal_blocks": 2,
                       **HEAD_CONFIGURATION}.items():
        _compare(value, training[key], f"training image-feature head.{key}")
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
    checkpoint_expected = {"format_version": 1, "model_name": "LeadImageSpatiotemporal", "class_names": CLASS_NAMES,
                           "input_shape": INPUT_SHAPE, "config": cfg["training"], "normalization": normalization,
                           "training_records": COUNTS["train"], "class_counts_train": train_counts,
                           "best_epoch": training["best_epoch"], "validation_selection_key": training["best_selection_key"],
                           "best_validation_metrics": metrics["validation"], "selection_rule": training["checkpoint_selection"],
                           "sequence_interpretation": training["sequence_interpretation"], "resume_supported": False,
                           "head_configuration": HEAD_CONFIGURATION}
    for key, value in checkpoint_expected.items():
        _compare(value, checkpoint[key], f"checkpoint.{key}")
    state = checkpoint["state_dict"]
    _require(isinstance(state, dict) and state and all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
             for value in state.values()), "Checkpoint has missing/nonfinite tensors")
    _compare(17161, training["num_parameters"], "prespecified parameter count")
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
            "verifier_sha256": _sha(Path(__file__)),
            "verification_helper_sha256": _sha(root / "src/verify_ptbxl_run.py"), "model_inference_performed": False,
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
    destination = ROOT / "results/ptbxl_spatiotemporal" / arguments.run_id / "verification.json"
    if destination.exists():
        _require(_json(destination).get("status") == "passed", "Existing verification is not a passed report")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "run_id": arguments.run_id,
                      "verification": str(destination), "prediction_records_verified": report["prediction_records_verified"]}, indent=2))


if __name__ == "__main__":
    main()
