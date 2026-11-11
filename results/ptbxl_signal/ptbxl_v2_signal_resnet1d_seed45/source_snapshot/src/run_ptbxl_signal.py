"""Run the signal-domain reference model (protocol ptbxl_v2_signal_resnet1d) for one seed.

Usage:
    python -X utf8 -m src.run_ptbxl_signal --seed 42 [--device cuda]

Same steps as src/run_ptbxl_experiment.py: validation-only checkpoint selection with the shared
training loop, checkpoint freeze and reload check, one test inference, patient-cluster bootstrap,
append-only experiment log, completed runs hash-checked before and after. Only the input differs:
raw 100 Hz waveforms (src/ptbxl_signals.py) in the image cache's cohort and order.

The shared loop (src/training/ptbxl_trainer.py) is used unchanged. During a run, signal_trainer()
registers the 1D model in its model table and points its batch-shape check at 12 x 1000, and restores
both afterwards; this is recorded in provenance.json.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np

from src.common import ROOT, environment, read_json, sha256_file, utc_now, write_json
from src.evaluation.multilabel import save_multilabel_evaluation
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.models.ptbxl_resnet1d import ResNet1dWang
from src.ptbxl_learning_data import CLASS_NAMES
from src.ptbxl_signals import CACHE, SHAPE, SignalDataset
from src.run_ptbxl_experiment import LOG, append_log_locked, check_unchanged, completed_run_hashes, labels
from src.training import ptbxl_trainer as trainer

CONFIG = ROOT / "experiments/configs/ptbxl_v2_signal_resnet1d.json"
CODE_FILES = ["src/run_ptbxl_signal.py", "src/ptbxl_signals.py", "src/models/ptbxl_resnet1d.py",
              "src/training/ptbxl_trainer.py", "src/run_ptbxl_experiment.py", "src/evaluation/multilabel.py",
              "src/evaluation/ptbxl_uncertainty.py", "src/common.py", "src/baseline_preservation.py"]

@contextmanager
def signal_trainer():
    """Register the 1D model and the 12 x 1000 batch shape in the shared loop; restore both afterwards."""
    previous_shape = trainer.INPUT_SHAPE
    trainer.MODELS["resnet1d_wang"] = ResNet1dWang
    trainer.INPUT_SHAPE = SHAPE
    try:
        yield
    finally:
        trainer.INPUT_SHAPE = previous_shape
        trainer.MODELS.pop("resnet1d_wang", None)


def run(seed, run_id=None, device=None, config=CONFIG):
    with signal_trainer():
        return _run(seed, run_id, device, config)


def _run(seed, run_id, device, config):
    cfg = read_json(config)
    training_cfg = trainer.get_config({**cfg["training"], "seed": seed, **({"device": device} if device else {})})
    run_id = run_id or f"{cfg['protocol']}_seed{seed}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run ID")
    with LOG.open() as handle:
        if any(row["experiment_id"] == run_id for row in csv.DictReader(handle)):
            raise ValueError(f"Run ID {run_id} already in the experiment log")
    out, model_dir = ROOT / "results" / cfg["result_family"] / run_id, ROOT / "models" / cfg["result_family"] / run_id
    if out.exists() or model_dir.exists():
        raise FileExistsError("Run already exists; choose a new ID, never repeat its test inference")
    protected = completed_run_hashes()
    original_log = LOG.read_bytes()
    out.mkdir(parents=True)
    started, timer = utc_now(), time.perf_counter()
    record = {"experiment_id": run_id, "date": started, "model": cfg["log_model_name"], "status": "failed",
              "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
              "preprocessing": json.dumps(cfg["preprocessing"], sort_keys=True), "random_seed": seed,
              "hyperparameters": json.dumps(training_cfg, sort_keys=True), "accuracy": "", "balanced_accuracy": "",
              "macro_f1": "", "weighted_f1": "", "notes": "", "output_directory": out.relative_to(ROOT).as_posix()}
    try:
        train, validation = SignalDataset("train"), SignalDataset("validation")
        write_json(out / "config.json", {**cfg, "training": training_cfg, "run_id": run_id})
        for relative in CODE_FILES:
            destination = out / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        write_json(out / "provenance.json", {
            "started_utc": started, "environment": environment(), "configuration_sha256": sha256_file(config),
            "source_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
            "signal_cache_metadata_sha256": sha256_file(CACHE / "metadata.json"), "normalization": train.normalization,
            "protected_prior_files": len(protected), "test_policy": cfg["test_policy"],
            "runtime_registration": {"trainer.MODELS['resnet1d_wang']": "src.models.ptbxl_resnet1d.ResNet1dWang",
                                     "trainer.INPUT_SHAPE": list(SHAPE)}})
        model, training = trainer.train_model(train, validation, training_cfg, model_dir, train.normalization)
        write_json(out / "training_metadata.json", training)
        checkpoint_hash = sha256_file(model_dir / "best.pt")
        write_json(out / "selection_frozen.json", {"frozen_utc": utc_now(), "checkpoint_sha256": checkpoint_hash,
                   "best_epoch": training["best_epoch"], "threshold": 0.5, "test_inference_started": False})
        metrics, saved = {}, {}
        for split, dataset in (("train", train), ("validation", validation)):
            saved[split] = trainer.predict_proba(model, dataset, training_cfg["batch_size"], split)
            metrics[split] = save_multilabel_evaluation(out, split, dataset.rows, labels(dataset), saved[split], CLASS_NAMES)
        if abs(metrics["validation"]["macro_roc_auc"] - training["best_validation_metrics"]["macro_roc_auc"]) > 1e-9:
            raise ValueError("Restored checkpoint does not reproduce the selected validation result")
        replay_model, _ = trainer.load_model(model_dir / "best.pt", device=training_cfg["device"])
        replay = trainer.predict_proba(replay_model, validation, training_cfg["batch_size"], "replay")
        if float(np.abs(replay - saved["validation"]).max()) > 1e-6:
            raise ValueError("Checkpoint reload does not reproduce validation probabilities")
        if sha256_file(model_dir / "best.pt") != checkpoint_hash:
            raise ValueError("Selected checkpoint changed before test inference")
        check_unchanged(protected)
        test = SignalDataset("test")
        write_json(out / "test_inference_state.json", {"status": "started", "started_utc": utc_now(),
                   "checkpoint_sha256": checkpoint_hash, "inference_calls": 1})
        probability = trainer.predict_proba(model, test, training_cfg["batch_size"], "test")
        truth = labels(test)
        metrics["test"] = save_multilabel_evaluation(out, "test", test.rows, truth, probability, CLASS_NAMES)
        write_json(out / "test_inference_state.json", {"status": "completed", "completed_utc": utc_now(),
                   "checkpoint_sha256": checkpoint_hash, "inference_calls": 1, "records": len(test)})
        write_json(out / "test_patient_bootstrap.json", patient_cluster_bootstrap(
            truth, probability, [row["patient_id"] for row in test.rows], CLASS_NAMES,
            n_resamples=cfg["uncertainty"]["resamples"], seed=cfg["uncertainty"]["seed"]))
        check_unchanged(protected)
        t = metrics["test"]
        record.update(status="completed", macro_f1=t["macro_f1"], weighted_f1=t["weighted_f1"],
                      notes=(f"Signal-domain reference (100 Hz waveforms, ResNet1D Wang); test macro AUROC={t['macro_roc_auc']:.6f}; "
                             f"macro AP={t['macro_average_precision']:.6f}; subset_accuracy={t['subset_accuracy']:.6f}. "
                             "Same cohort, folds and protocol as the image models. Fold 10 previously observed: exploratory."))
        append_log_locked(record, original_log)
        result = {"status": "completed", "experiment_id": run_id, "model": training_cfg["model"], "seed": seed,
                  "started_utc": started, "completed_utc": utc_now(), "total_runtime_seconds": time.perf_counter() - timer,
                  "dataset_id": cfg["dataset_id"], "split_version": cfg["split_version"], "training": training,
                  "metrics": metrics, "checkpoint_sha256": checkpoint_hash, "test_inference_calls": 1, "threshold": 0.5,
                  "model_directory": model_dir.relative_to(ROOT).as_posix()}
        write_json(out / "run.json", result)
        print(json.dumps({"run_id": run_id, "status": "completed", "test_macro_auc": t["macro_roc_auc"],
                          "test_macro_f1": t["macro_f1"], "runtime_seconds": result["total_runtime_seconds"]}, indent=2), flush=True)
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    run(arguments.seed, arguments.run_id, arguments.device)
