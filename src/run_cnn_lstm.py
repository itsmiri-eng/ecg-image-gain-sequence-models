"""Execute one frozen image CNN-LSTM experiment; test inference follows selection."""
import argparse
import csv
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from src.baseline_preservation import append_experiment_log, verify_baseline_preservation
from src.common import ROOT, environment, read_json, sha256_file, source_hashes, utc_now, write_csv, write_json
from src.evaluation.metrics import save_evaluation
from src.verify import load_verified_data


def run(config_path, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid experiment ID")
    config_path = Path(config_path)
    config = read_json(config_path)
    config_sha = sha256_file(config_path)
    if config["model"] != "cnn_lstm" or config["seed"] != config["cnn_lstm"]["seed"]:
        raise ValueError("Inconsistent model or seed")
    before = verify_baseline_preservation()
    with (ROOT / "experiments/experiment_log.csv").open() as f:
        if any(r["experiment_id"] == run_id for r in csv.DictReader(f)):
            raise ValueError("Experiment ID already logged")
    out = ROOT / "results/cnn_lstm" / run_id
    model_dir = ROOT / "models/cnn_lstm" / run_id
    if out.exists() or model_dir.exists():
        raise FileExistsError("Run already exists; use a new ID")
    out.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    started, start = utc_now(), time.perf_counter()
    log = {"experiment_id": run_id, "date": started, "model": "CNN_LSTM", "status": "failed",
           "dataset_version": "", "split_version": "", "preprocessing": "", "random_seed": config["seed"],
           "hyperparameters": json.dumps(config["cnn_lstm"], sort_keys=True), "accuracy": "",
           "balanced_accuracy": "", "macro_f1": "", "weighted_f1": "", "notes": "",
           "output_directory": str(out.relative_to(ROOT))}
    logged = False
    try:
        x, y, rows, parts, frozen = load_verified_data()
        audit, baseline_config = frozen["audit"], frozen["config"]
        for key in ["dataset_version", "split_version", "split_sha256", "cache_sha256"]:
            if config[key] != audit[key]:
                raise ValueError(f"Frozen comparison mismatch: {key}")
        if config["class_names"] != baseline_config["class_names"] or list(x.shape[1:]) != config["input_shape"]:
            raise ValueError("Class order or input shape mismatch")
        log.update(dataset_version=audit["dataset_version"], split_version=audit["split_version"],
                   preprocessing=json.dumps(baseline_config["preprocessing"], sort_keys=True))
        write_json(out / "config.json", config)
        write_json(out / "preprocessing.json", baseline_config["preprocessing"])
        initial_sources = source_hashes()
        write_json(out / "provenance.json", {"started_utc": started, "environment": environment(),
                   "source_sha256": initial_sources, "configuration_sha256": config_sha,
                   "dataset_version": audit["dataset_version"], "split_version": audit["split_version"],
                   "split_sha256": audit["split_sha256"], "cache_sha256": audit["cache_sha256"],
                   "baseline_preservation_before": before})
        write_csv(out / "split_used.csv", rows)
        from src.models.cnn_lstm import train_cnn_lstm
        predictor, training = train_cnn_lstm(x[parts["train"]], y[parts["train"]],
                                           x[parts["validation"]], y[parts["validation"]],
                                           config["cnn_lstm"], model_dir)
        write_json(out / "training_metadata.json", training)
        if config_sha != sha256_file(config_path) or initial_sources != source_hashes():
            raise ValueError("Configuration or source changed during training; test inference refused")
        checkpoint_sha = sha256_file(model_dir / "best.pt")
        write_json(out / "selection_frozen.json", {"frozen_utc": utc_now(),
                   "snapshot_kind": "immutable pre-inference selection snapshot",
                   "checkpoint_sha256": checkpoint_sha, "configuration_sha256": config_sha,
                   "selected_epoch": training["best_epoch"], "selection_rule": config["selection_rule"],
                   "test_inference_started": False, "test_policy": config["test_policy"]})
        metrics, probability_checks = {}, {}
        for subset in ["train", "validation", "test"]:
            idx = parts[subset]
            if sha256_file(model_dir / "best.pt") != checkpoint_sha:
                raise ValueError("Checkpoint changed after selection")
            if subset == "test":
                test_started = utc_now()
                write_json(out / "test_inference_state.json", {"status": "started", "started_utc": test_started,
                           "checkpoint_sha256": checkpoint_sha, "configuration_sha256": config_sha})
            p = np.asarray(predictor.predict_proba(x[idx]), dtype=np.float64)
            sums = p.sum(axis=1, keepdims=True)
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(sums, 1.0, atol=1e-6):
                raise ValueError("Invalid prediction probabilities")
            probability_checks[subset] = {"maximum_float32_sum_error": float(np.abs(sums - 1).max()),
                                          "normalization": "float64 row sum normalization of softmax roundoff only"}
            p = p / sums
            metrics[subset] = save_evaluation(out, subset, [rows[i] for i in idx], y[idx], p, config["class_names"])
            if subset == "test":
                write_json(out / "test_inference_state.json", {"status": "completed", "started_utc": test_started,
                           "completed_utc": utc_now(), "checkpoint_sha256": checkpoint_sha,
                           "configuration_sha256": config_sha, "inference_calls": 1, "records": len(idx)})
        if initial_sources != source_hashes() or config_sha != sha256_file(config_path):
            raise ValueError("Source or configuration changed during evaluation")
        verify_baseline_preservation()
        elapsed = time.perf_counter() - start
        result = {"status": "completed", "experiment_id": run_id, "model": "cnn_lstm",
                  "started_utc": started, "completed_utc": utc_now(), "total_runtime_seconds": elapsed,
                  "dataset_version": audit["dataset_version"], "split_version": audit["split_version"],
                  "training": training, "metrics": metrics, "test_evaluation_policy": config["test_policy"],
                  "test_inference_calls": 1, "checkpoint_sha256": checkpoint_sha,
                  "probability_checks": probability_checks, "representation": config["representation"],
                  "comparison_limit": config["comparison_limit"], "model_directory": str(model_dir.relative_to(ROOT))}
        log.update(status="completed", notes="Spatial image sequence; validation macro F1 selection; previously observed test exploratory; no test tuning; not an isolated architecture ablation")
        log.update({k: metrics["test"][k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]})
        append_experiment_log(log)
        logged = True
        write_json(out / "baseline_preservation.json", verify_baseline_preservation())
        write_json(out / "run.json", result)
        print(json.dumps({"run_id": run_id, "runtime_seconds": elapsed,
                          "test": {k: metrics["test"][k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}}, indent=2))
        return result
    except BaseException as exc:
        failure = {"status": "failed", "experiment_id": run_id,
                   "started_utc": started, "failed_utc": utc_now(), "error": repr(exc),
                   "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter() - start,
                   "log_record_already_appended": logged}
        write_json(out / "failure.json", failure)
        write_json(out / "run.json", failure)
        if not logged:
            log.update(status="failed", notes=repr(exc))
            try:
                append_experiment_log(log)
            except BaseException as log_exc:
                write_json(out / "logging_failure.json", {"original_error": repr(exc),
                           "logging_error": repr(log_exc), "record_not_appended": log})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/cnn_lstm_image_v1.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.config, args.run_id)
