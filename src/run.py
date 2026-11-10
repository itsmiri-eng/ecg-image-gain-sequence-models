"""Run one prespecified baseline. Test data are scored only after fitting/model selection."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import traceback

from src.common import ROOT, environment, sha256_file, source_hashes, utc_now, write_csv, write_json
from src.evaluation.metrics import save_evaluation
from src.verify import load_verified_data


def append_log(record):
    path = ROOT / "experiments/experiment_log.csv"
    existing = []
    if path.exists():
        with path.open() as f:
            existing = list(csv.DictReader(f))
    if any(r["experiment_id"] == record["experiment_id"] for r in existing):
        raise ValueError("Experiment ID already logged")
    write_csv(path, existing + [record], list(record))


def run(model_name, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Use an experiment ID containing only letters, numbers, underscores, and hyphens")
    log_path = ROOT / "experiments/experiment_log.csv"
    if log_path.exists():
        with log_path.open() as f:
            if any(r["experiment_id"] == run_id for r in csv.DictReader(f)):
                raise ValueError("Experiment ID already exists in the central log")
    out = ROOT / "results" / model_name / run_id
    model_dir = ROOT / "models" / model_name / run_id
    if out.exists() or model_dir.exists():
        raise FileExistsError(f"Run {run_id} already exists; use a new ID, do not overwrite completed evidence")
    out.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    started = utc_now()
    start = time.perf_counter()
    audit, config = {}, {}
    log = {"experiment_id": run_id, "date": started, "model": model_name.upper(), "status": "failed",
           "dataset_version": "", "split_version": "", "preprocessing": "", "random_seed": "",
           "hyperparameters": "", "accuracy": "", "balanced_accuracy": "", "macro_f1": "", "weighted_f1": "",
           "notes": "", "output_directory": str(out.relative_to(ROOT))}
    try:
        x, y, rows, parts, protocol = load_verified_data()
        audit, config = protocol["audit"], protocol["config"]
        log.update(dataset_version=audit["dataset_version"], split_version=audit["split_version"],
                   preprocessing=json.dumps(config["preprocessing"], sort_keys=True), random_seed=config["seed"],
                   hyperparameters=json.dumps(config[model_name], sort_keys=True))
        write_json(out / "config.json", config)
        write_json(out / "provenance.json", {"started_utc": started, "environment": environment(),
                    "source_sha256": source_hashes(), "dataset_version": audit["dataset_version"],
                    "split_version": audit["split_version"], "split_sha256": audit["split_sha256"],
                    "cache_sha256": audit["cache_sha256"]})
        write_csv(out / "split_used.csv", rows)
        train, val = parts["train"], parts["validation"]
        if model_name == "lda":
            from src.models.lda import train_lda
            predictor, training = train_lda(x[train], y[train], config["lda"], config["seed"], model_dir)
        else:
            from src.models.cnn import train_cnn
            predictor, training = train_cnn(x[train], y[train], x[val], y[val], config["cnn"], model_dir)
        write_json(out / "training_metadata.json", training)
        # Both configurations are frozen. No fitting or tuning below this point.
        all_metrics = {}
        for subset in ["train", "validation", "test"]:
            idx = parts[subset]
            probabilities = predictor.predict_proba(x[idx])
            all_metrics[subset] = save_evaluation(out, subset, [rows[i] for i in idx], y[idx],
                                                  probabilities, config["class_names"])
        elapsed = time.perf_counter() - start
        result = {"status": "completed", "experiment_id": run_id, "model": model_name,
                  "started_utc": started, "completed_utc": utc_now(), "total_runtime_seconds": elapsed,
                  "dataset_version": audit["dataset_version"], "split_version": audit["split_version"],
                  "training": training, "metrics": all_metrics,
                  "test_evaluation_policy": "One final test evaluation after all model settings/weights were frozen",
                  "model_directory": str(model_dir.relative_to(ROOT))}
        write_json(out / "run.json", result)
        log.update(status="completed", notes="Fixed baseline; no test tuning; printed-ID groups are not verified patient IDs")
        log.update({k: all_metrics["test"][k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]})
        append_log(log)
        print(json.dumps({"run_id": run_id, "runtime_seconds": elapsed, "test": all_metrics["test"]}, indent=2))
    except BaseException as exc:
        failure = {"status": "failed", "experiment_id": run_id, "started_utc": started,
                   "failed_utc": utc_now(), "runtime_seconds": time.perf_counter() - start,
                   "error": repr(exc), "traceback": traceback.format_exc()}
        write_json(out / "failure.json", failure)
        log["notes"] = repr(exc)
        append_log(log)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["lda", "cnn"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.model, args.run_id)
