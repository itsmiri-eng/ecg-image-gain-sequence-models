"""Write the missing run.json and experiment-log row for PTB-XL v2 runs stopped by the Windows log-check bug.

Usage:
    python -X utf8 -m src.recover_ptbxl_run_record --check-against ptbxl_v2_cnn_pool_seed42
    python -X utf8 -m src.recover_ptbxl_run_record RUN_ID [RUN_ID ...]            (dry run)
    python -X utf8 -m src.recover_ptbxl_run_record --apply RUN_ID [RUN_ID ...]

On the Windows PC (1-2 Nov 2026) src/baseline_preservation.py compared the lock key
"experiments/experiment_log.csv" with a backslash path, so every run stopped after its single test
inference and before its log row and run.json were written. This rebuilds both from the run's saved
files exactly as src/run_ptbxl_experiment.py builds them. No model is loaded and the test partition
is not read again. Only the run's own completion time and runtime are taken from failure.json, which
is then moved (not deleted) to experiments/record_recovery/<run_id>/.

--check-against rebuilds an already completed run and compares the result with its real log row and
run.json, to show the rebuild is faithful.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import shutil
import sys

from src.baseline_preservation import LOG, append_experiment_log
from src.common import ROOT, read_json, sha256_file, utc_now, write_json
from src.run_ptbxl_experiment import log_lock

BUG = "ValueError(\"Protected baseline artifacts changed: ['experiments/experiment_log.csv']\")"
FAMILIES = ("ptbxl_v2", "ptbxl_v2_fixedgain")
RECOVERY = ROOT / "experiments/record_recovery"
NOTE = (" Run record written afterwards by src/recover_ptbxl_run_record.py (Windows path bug in the "
        "experiment-log check, 2026-11-02); test inference not repeated.")
TIMING_KEYS = ("completed_utc", "total_runtime_seconds")


def find(run_id):
    matches = [ROOT / "results" / family / run_id for family in FAMILIES if (ROOT / "results" / family / run_id).is_dir()]
    if len(matches) != 1:
        raise ValueError(f"{run_id}: expected exactly one result folder, found {len(matches)}")
    return matches[0]


def logged_rows():
    with LOG.open(newline="") as handle:
        return {row["experiment_id"]: row for row in csv.DictReader(handle)}


def rebuild(out):
    """Return (log record, run.json without timing keys, model folder) as run_ptbxl_experiment.run writes them."""
    cfg = read_json(out / "config.json")
    run_id, training_cfg = cfg["run_id"], cfg["training"]
    if out.name != run_id or out.parent.name != cfg["result_family"]:
        raise ValueError(f"{run_id}: folder does not match its saved configuration")
    provenance, training = read_json(out / "provenance.json"), read_json(out / "training_metadata.json")
    frozen, state = read_json(out / "selection_frozen.json"), read_json(out / "test_inference_state.json")
    metrics = {split: read_json(out / f"{split}_metrics.json") for split in ("train", "validation", "test")}
    test = metrics["test"]
    model_dir = ROOT / "models" / cfg["result_family"] / run_id
    record = {"experiment_id": run_id, "date": provenance["started_utc"], "model": cfg["log_model_name"],
              "status": "completed", "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
              "preprocessing": json.dumps(cfg["preprocessing"], sort_keys=True), "random_seed": training_cfg["seed"],
              "hyperparameters": json.dumps(training_cfg, sort_keys=True), "accuracy": "", "balanced_accuracy": "",
              "macro_f1": test["macro_f1"], "weighted_f1": test["weighted_f1"],
              "notes": (f"Five-label PTB-XL protocol v2; test macro AUROC={test['macro_roc_auc']:.6f}; "
                        f"macro AP={test['macro_average_precision']:.6f}; "
                        f"subset_accuracy={test['subset_accuracy']:.6f}. Single-label accuracy columns "
                        "blank by design. Fold 10 previously observed: exploratory comparison."),
              "output_directory": out.relative_to(ROOT).as_posix()}
    result = {"status": "completed", "experiment_id": run_id, "model": training_cfg["model"],
              "seed": training_cfg["seed"], "started_utc": provenance["started_utc"],
              "dataset_id": cfg["dataset_id"], "split_version": cfg["split_version"], "training": training,
              "metrics": metrics, "checkpoint_sha256": frozen["checkpoint_sha256"], "test_inference_calls": 1,
              "threshold": 0.5, "model_directory": model_dir.relative_to(ROOT).as_posix()}
    if state.get("status") != "completed" or state.get("inference_calls") != 1:
        raise ValueError(f"{run_id}: test inference is not recorded as completed exactly once")
    if not (state["checkpoint_sha256"] == frozen["checkpoint_sha256"] == sha256_file(model_dir / "best.pt")):
        raise ValueError(f"{run_id}: checkpoint hash differs between freeze, test inference and model file")
    return record, result, model_dir


def as_log_row(record):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(record))
    writer.writeheader()
    writer.writerow(record)
    return next(csv.DictReader(io.StringIO(buffer.getvalue())))


def check_against(run_id):
    out = find(run_id)
    record, result, _ = rebuild(out)
    actual_run = read_json(out / "run.json")
    differences = [f"log.{key}" for key, value in as_log_row(record).items() if logged_rows()[run_id][key] != value]
    differences += [f"run.{key}" for key in set(actual_run) | set(result)
                    if key not in TIMING_KEYS and actual_run.get(key) != result.get(key)]
    print(json.dumps({"run_id": run_id, "differences": differences, "faithful": not differences}, indent=2))
    return not differences


def recover(run_id, apply):
    out = find(run_id)
    if (out / "run.json").exists():
        raise ValueError(f"{run_id}: run.json already exists; nothing to recover")
    failure = read_json(out / "failure.json")
    if failure.get("error") != BUG:
        raise ValueError(f"{run_id}: failed for a different reason, not recovered: {failure.get('error')}")
    if not (out / "test_patient_bootstrap.json").is_file():
        raise ValueError(f"{run_id}: bootstrap file missing; the run did not reach the logging step")
    if run_id in logged_rows():
        raise ValueError(f"{run_id}: already has an experiment-log row")
    record, result, _ = rebuild(out)
    destination = RECOVERY / run_id
    stopped = datetime.fromtimestamp((out / "failure.json").stat().st_mtime, timezone.utc).isoformat()
    record["notes"] += NOTE
    result.update(completed_utc=stopped, total_runtime_seconds=failure["runtime_seconds"], record_recovery={
        "recovered_utc": utc_now(), "tool": "src/recover_ptbxl_run_record.py",
        "tool_sha256": sha256_file(ROOT / "src/recover_ptbxl_run_record.py"),
        "reason": "Windows path bug in src/baseline_preservation.py stopped the run after test inference, "
                  "before the log row and run.json were written",
        "test_inference_repeated": False, "model_loaded": False,
        "completed_utc_source": "modification time of failure.json, written when the original process stopped",
        "total_runtime_seconds_source": "failure.json runtime_seconds",
        "original_failure_moved_to": (destination / "failure.json").relative_to(ROOT).as_posix()})
    summary = {"run_id": run_id, "test_macro_auc": result["metrics"]["test"]["macro_roc_auc"],
               "completed_utc": stopped, "applied": apply}
    if apply:
        destination.mkdir(parents=True, exist_ok=False)
        with log_lock():
            append_experiment_log(record)
        write_json(out / "run.json", result)
        shutil.move(out / "failure.json", destination / "failure.json")
        write_json(destination / "recovery.json", {"log_record": record, "run_json": result})
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="*")
    parser.add_argument("--apply", action="store_true", help="write the records; without it, only check")
    parser.add_argument("--check-against", metavar="COMPLETED_RUN_ID")
    arguments = parser.parse_args()
    if arguments.check_against and not check_against(arguments.check_against):
        sys.exit(1)
    for run_id in arguments.run_ids:
        recover(run_id, arguments.apply)


if __name__ == "__main__":
    main()
