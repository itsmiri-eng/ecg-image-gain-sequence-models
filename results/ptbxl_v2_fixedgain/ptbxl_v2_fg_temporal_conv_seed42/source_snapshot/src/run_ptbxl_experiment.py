"""Run one PTB-XL image network under protocol v2 (matched baselines and repeated seeds).

Usage:
    python -m src.run_ptbxl_experiment --config experiments/configs/ptbxl_v2_cnn_pool.json --seed 42 [--device cuda]

Every run uses the same cached images, official patient folds (1-8 train, 9
validation, 10 test), training-only normalization, five labels, fixed 0.5
threshold and validation-only checkpoint selection. The test partition is opened
once, after the selected checkpoint has been frozen and hashed. Completed runs
are never overwritten; prior completed run directories are hash-checked before
and after so that this run cannot silently alter earlier evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np

if os.name == "nt":  # Windows has no fcntl
    import msvcrt
else:
    import fcntl

from src.baseline_preservation import append_experiment_log
from src.common import ROOT, environment, read_json, sha256_file, utc_now, write_csv, write_json
from src.evaluation.multilabel import save_multilabel_evaluation
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.portable_cache import PortableCachedECGDataset
from src.ptbxl_learning_data import CLASS_NAMES
from src.training.ptbxl_trainer import get_config, load_model, predict_proba, train_model

CODE_FILES = ["src/run_ptbxl_experiment.py", "src/ptbxl_fixed_gain.py", "src/training/ptbxl_trainer.py", "src/models/ptbxl_cnn_pool.py",
              "src/models/ptbxl_cnn_lstm.py", "src/models/ptbxl_spatiotemporal.py", "src/models/ptbxl_hybrid_vit.py",
              "src/ptbxl_learning_data.py", "src/evaluation/multilabel.py", "src/evaluation/ptbxl_uncertainty.py",
              "src/common.py", "src/baseline_preservation.py"]
LOG = ROOT / "experiments/experiment_log.csv"


def make_dataset(cfg, split, **kwargs):
    """Dataset for the protocol's input cache: the v1 adaptive-gain cache by default.

    The v1 cache opens through PortableCachedECGDataset, which applies the same checks as
    CachedECGDataset but also works after the project moved to another computer.
    """
    if cfg.get("dataset_loader", "v1") == "fixed_gain":
        from src.ptbxl_fixed_gain import FixedGainDataset
        return FixedGainDataset(ROOT / cfg["cache_root"], split, **kwargs)
    return PortableCachedECGDataset(ROOT / cfg["cache_root"], split, **kwargs)


def labels(dataset):
    return np.asarray([[row[f"label_{name}"] for name in CLASS_NAMES] for row in dataset.rows], dtype=np.int8)


def completed_run_hashes(exclude=()):
    """Hash every file of every completed run (a run.json exists) in models/ and results/."""
    hashes = {}
    for run_file in sorted((ROOT / "results").glob("*/*/run.json")):
        if run_file.parent.name in exclude:
            continue
        for parent in (ROOT / "results", ROOT / "models"):
            directory = parent / run_file.parent.parent.name / run_file.parent.name
            for path in sorted(directory.rglob("*")) if directory.is_dir() else []:
                if path.is_file() and not path.name.startswith("."):
                    hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def check_unchanged(protected):
    changed = [name for name, digest in protected.items()
               if not (ROOT / name).is_file() or sha256_file(ROOT / name) != digest]
    if changed:
        raise ValueError(f"Previously completed artifacts changed: {changed[:10]}")


@contextmanager
def log_lock():
    """Serialize experiment-log appends when several runs execute concurrently."""
    with (ROOT / "experiments/.experiment_log.lock").open("w") as handle:
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def append_log_locked(record, original_log):
    """Append one row; earlier rows (including any appended by concurrent runs) must be untouched."""
    with log_lock():
        if not LOG.read_bytes().startswith(original_log):
            raise ValueError("Experiment log rows present before this run were changed")
        append_experiment_log(record)


def run(config_path, seed, run_id=None, device=None):
    config_path = Path(config_path)
    cfg = read_json(config_path)
    training_cfg = get_config({**cfg["training"], "seed": seed, **({"device": device} if device else {})})
    run_id = run_id or f"{cfg['protocol']}_seed{seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run ID")
    with LOG.open() as handle:
        if any(row["experiment_id"] == run_id for row in csv.DictReader(handle)):
            raise ValueError(f"Run ID {run_id} already in the experiment log")
    family = cfg["result_family"]
    out, model_dir = ROOT / "results" / family / run_id, ROOT / "models" / family / run_id
    if out.exists() or model_dir.exists():
        raise FileExistsError("Run already exists; choose a new ID, never repeat its test inference")
    cache = ROOT / cfg["cache_root"]
    cache_hashes = {name: sha256_file(cache / name) for name in ("protocol.json", "metadata.json", "cache_manifest.csv")}
    if cache_hashes != cfg["cache_sha256"]:
        raise ValueError("Cached input dataset differs from the frozen protocol binding")
    protected = completed_run_hashes()
    original_log = LOG.read_bytes()
    out.mkdir(parents=True)
    started, timer = utc_now(), time.perf_counter()
    source_hashes = {name: sha256_file(ROOT / name) for name in CODE_FILES}
    record = {"experiment_id": run_id, "date": started, "model": cfg["log_model_name"], "status": "failed",
              "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
              "preprocessing": json.dumps(cfg["preprocessing"], sort_keys=True), "random_seed": seed,
              "hyperparameters": json.dumps(training_cfg, sort_keys=True), "accuracy": "", "balanced_accuracy": "",
              "macro_f1": "", "weighted_f1": "", "notes": "", "output_directory": out.relative_to(ROOT).as_posix()}
    try:
        train, validation = make_dataset(cfg, "train"), make_dataset(cfg, "validation")
        if train.normalization != validation.normalization or train.normalization["fit_partition"] != "train":
            raise ValueError("Normalization must be fitted on training only and shared")
        write_json(out / "config.json", {**cfg, "training": training_cfg, "run_id": run_id})
        for relative in CODE_FILES:
            destination = out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        write_json(out / "provenance.json", {"started_utc": started, "environment": environment(),
                   "configuration_sha256": sha256_file(config_path), "source_sha256": source_hashes,
                   "cache_sha256": cache_hashes, "normalization": train.normalization,
                   "protected_prior_files": len(protected), "test_policy": cfg["test_policy"],
                   "cache_binding_check": getattr(train, "binding_check", None)})
        model, training = train_model(train, validation, training_cfg, model_dir, train.normalization)
        write_json(out / "training_metadata.json", training)
        checkpoint_hash = sha256_file(model_dir / "best.pt")
        write_json(out / "selection_frozen.json", {"frozen_utc": utc_now(), "checkpoint_sha256": checkpoint_hash,
                   "best_epoch": training["best_epoch"], "threshold": 0.5, "test_inference_started": False})
        metrics, saved = {}, {}
        for split, dataset in (("train", train), ("validation", validation)):
            saved[split] = predict_proba(model, dataset, training_cfg["batch_size"], split)
            metrics[split] = save_multilabel_evaluation(out, split, dataset.rows, labels(dataset), saved[split], CLASS_NAMES)
        if abs(metrics["validation"]["macro_roc_auc"] - training["best_validation_metrics"]["macro_roc_auc"]) > 1e-9:
            raise ValueError("Restored checkpoint does not reproduce the selected validation result")
        replay_model, _ = load_model(model_dir / "best.pt", device=training_cfg["device"])
        replay = predict_proba(replay_model, validation, training_cfg["batch_size"], "replay")
        if float(np.abs(replay - saved["validation"]).max()) > 1e-6:
            raise ValueError("Checkpoint reload does not reproduce validation probabilities")
        if sha256_file(model_dir / "best.pt") != checkpoint_hash:
            raise ValueError("Selected checkpoint changed before test inference")
        check_unchanged(protected)
        test = make_dataset(cfg, "test")
        write_json(out / "test_inference_state.json", {"status": "started", "started_utc": utc_now(),
                   "checkpoint_sha256": checkpoint_hash, "inference_calls": 1})
        probability = predict_proba(model, test, training_cfg["batch_size"], "test")
        truth = labels(test)
        metrics["test"] = save_multilabel_evaluation(out, "test", test.rows, truth, probability, CLASS_NAMES)
        write_json(out / "test_inference_state.json", {"status": "completed", "completed_utc": utc_now(),
                   "checkpoint_sha256": checkpoint_hash, "inference_calls": 1, "records": len(test)})
        bootstrap = patient_cluster_bootstrap(truth, probability, [row["patient_id"] for row in test.rows], CLASS_NAMES,
                                              n_resamples=cfg["uncertainty"]["resamples"], seed=cfg["uncertainty"]["seed"])
        write_json(out / "test_patient_bootstrap.json", bootstrap)
        check_unchanged(protected)
        test_metrics = metrics["test"]
        record.update(status="completed", macro_f1=test_metrics["macro_f1"], weighted_f1=test_metrics["weighted_f1"],
                      notes=(f"Five-label PTB-XL protocol v2; test macro AUROC={test_metrics['macro_roc_auc']:.6f}; "
                             f"macro AP={test_metrics['macro_average_precision']:.6f}; "
                             f"subset_accuracy={test_metrics['subset_accuracy']:.6f}. Single-label accuracy columns "
                             "blank by design. Fold 10 previously observed: exploratory comparison."))
        append_log_locked(record, original_log)
        result = {"status": "completed", "experiment_id": run_id, "model": training_cfg["model"], "seed": seed,
                  "started_utc": started, "completed_utc": utc_now(),
                  "total_runtime_seconds": time.perf_counter() - timer, "dataset_id": cfg["dataset_id"],
                  "split_version": cfg["split_version"], "training": training, "metrics": metrics,
                  "checkpoint_sha256": checkpoint_hash, "test_inference_calls": 1, "threshold": 0.5,
                  "model_directory": model_dir.relative_to(ROOT).as_posix()}
        write_json(out / "run.json", result)
        print(json.dumps({"run_id": run_id, "status": "completed",
                          "test_macro_auc": test_metrics["macro_roc_auc"], "test_macro_f1": test_metrics["macro_f1"],
                          "runtime_seconds": result["total_runtime_seconds"]}, indent=2), flush=True)
        return result
    except BaseException as exc:
        write_json(out / "failure.json", {"status": "failed", "experiment_id": run_id, "error": repr(exc),
                   "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter() - timer})
        with LOG.open() as handle:
            logged = any(row["experiment_id"] == run_id for row in csv.DictReader(handle))
        if not logged:
            record.update(notes=repr(exc)[:500])
            append_log_locked(record, original_log)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="override the config's training device (recorded)")
    arguments = parser.parse_args()
    run(arguments.config, arguments.seed, arguments.run_id, arguments.device)
