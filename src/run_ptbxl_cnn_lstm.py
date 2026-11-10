"""Run the prespecified five-label image experiment without changing prior runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np
from torch.utils.data import Subset

from src.baseline_preservation import append_experiment_log, verify_baseline_preservation
from src.common import ROOT, environment, read_json, sha256_file, utc_now, write_csv, write_json
from src.evaluation.multilabel import evaluate_multilabel, save_multilabel_evaluation
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.ptbxl_learning_data import CachedECGDataset, CLASS_NAMES, INPUT_SHAPE
from src.models.ptbxl_cnn_lstm import train_model, load_checkpoint

CONFIG = ROOT / "experiments/configs/ptbxl_cnn_lstm_image_v1.json"
CODE_FILES = ["src/run_ptbxl_cnn_lstm.py", "src/models/ptbxl_cnn_lstm.py",
              "src/ptbxl_learning_data.py", "src/evaluation/multilabel.py",
              "src/evaluation/ptbxl_uncertainty.py", "src/common.py", "src/baseline_preservation.py"]


def _source_hashes():
    return {name: sha256_file(ROOT/name) for name in CODE_FILES}


def _labels(dataset):
    return np.asarray([[row[f"label_{name}"] for name in CLASS_NAMES] for row in dataset.rows], dtype=np.int8)


def _check_frozen(config_path, config_hash, sources, cache, cache_hashes, checkpoint=None, checkpoint_hash=None):
    if sha256_file(config_path) != config_hash or _source_hashes() != sources:
        raise ValueError("Experiment configuration or source changed after freezing")
    if any(sha256_file(cache/name) != digest for name, digest in cache_hashes.items()):
        raise ValueError("Input cache manifest, normalization or protocol changed")
    if checkpoint is not None and sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("Selected checkpoint changed")


def _prior_artifacts():
    saved = read_json(ROOT/"experiments/protocols/ptbxl_images_v1/prior_model_artifacts.json")["sha256"]
    # Preserve the current three model runs. The central log may only be appended.
    protected = {name: digest for name, digest in saved.items() if name != "experiments/experiment_log.csv"}
    for name in ["protocol.json", "generation_state.json", "generation_summary.json", "verification.json",
                 "all_images.csv", "classification_manifest.csv", "classification_exclusions.csv",
                 "splits/train.csv", "splits/validation.csv", "splits/test.csv"]:
        relative = f"data/derived/ptbxl_images_v1/{name}"
        protected[relative] = sha256_file(ROOT/relative)
    _check_prior(protected)
    return protected


def _check_prior(protected):
    changed = [name for name, digest in protected.items() if sha256_file(ROOT/name) != digest]
    if changed:
        raise ValueError(f"Previously completed artifacts changed: {changed}")
    return verify_baseline_preservation()


def run(config_path, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid experiment ID")
    config_path = Path(config_path)
    cfg, config_hash = read_json(config_path), sha256_file(config_path)
    if (cfg["model"] != "ptbxl_image_cnn_lstm" or cfg["class_names"] != list(CLASS_NAMES)
            or cfg["input_shape"] != list(INPUT_SHAPE) or cfg["threshold"] != 0.5
            or cfg["seed"] != cfg["training"]["seed"]):
        raise ValueError("Incompatible model, input, target or threshold protocol")
    source, cache = ROOT/cfg["source_dataset_root"], ROOT/cfg["cache_root"]
    for name, key in [("classification_manifest.csv", "source_manifest_sha256"),
                      ("verification.json", "source_verification_sha256"), ("protocol.json", "source_protocol_sha256")]:
        if sha256_file(source/name) != cfg[key]:
            raise ValueError(f"Source dataset binding changed: {name}")
    log_path = ROOT/"experiments/experiment_log.csv"
    with log_path.open() as handle:
        if any(row["experiment_id"] == run_id for row in csv.DictReader(handle)):
            raise ValueError("Experiment ID already logged")
    out, model_dir = ROOT/"results/ptbxl_cnn_lstm"/run_id, ROOT/"models/ptbxl_cnn_lstm"/run_id
    if out.exists() or model_dir.exists():
        raise FileExistsError("Run already exists; choose a new ID, never overwrite or repeat its test inference")
    protected = _prior_artifacts()
    original_log = log_path.read_bytes()
    out.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    started, timer, logged = utc_now(), time.perf_counter(), False
    record = {"experiment_id": run_id, "date": started, "model": "PTBXL_Image_CNN_LSTM", "status": "failed",
              "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
              "preprocessing": json.dumps(cfg["preprocessing"], sort_keys=True), "random_seed": cfg["seed"],
              "hyperparameters": json.dumps(cfg["training"], sort_keys=True),
              "accuracy": "", "balanced_accuracy": "", "macro_f1": "", "weighted_f1": "",
              "notes": "", "output_directory": str(out.relative_to(ROOT))}
    try:
        train, validation = CachedECGDataset(cache, "train"), CachedECGDataset(cache, "validation")
        if train.normalization != validation.normalization or train.normalization["fit_partition"] != "train":
            raise ValueError("Normalization must be fitted on training only and shared")
        sources = _source_hashes()
        cache_hashes = {name: sha256_file(cache/name) for name in ["protocol.json", "metadata.json", "cache_manifest.csv"]}
        write_json(out/"config.json", cfg)
        (out/"prior_experiment_log.csv").write_bytes(original_log)
        write_json(out/"protected_artifacts.json", protected)
        for relative in CODE_FILES:
            destination = out/"source_snapshot"/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, destination)
        write_json(out/"provenance.json", {"started_utc": started, "environment": environment(),
                   "source_sha256": sources, "configuration_sha256": config_hash,
                   "cache_sha256": cache_hashes, "normalization": train.normalization,
                   "test_policy": cfg["test_policy"], "selection_rule": cfg["selection_rule"],
                   "input_modality": "generated ECG page pixels; no native waveform values or numeric metadata features"})
        write_csv(out/"development_split_used.csv", train.rows + validation.rows)
        _check_frozen(config_path, config_hash, sources, cache, cache_hashes)
        predictor, training = train_model(train, validation, cfg["training"], model_dir,
                                          train.normalization, _labels(train).sum(axis=0).tolist())
        write_json(out/"training_metadata.json", training)
        checkpoint = model_dir/"best.pt"
        checkpoint_hash = sha256_file(checkpoint)
        _check_frozen(config_path, config_hash, sources, cache, cache_hashes, checkpoint, checkpoint_hash)
        write_json(out/"selection_frozen.json", {"frozen_utc": utc_now(), "checkpoint_sha256": checkpoint_hash,
                   "configuration_sha256": config_hash, "selection_rule": cfg["selection_rule"],
                   "threshold": cfg["threshold"], "training": training, "test_inference_started": False})
        metrics, saved_predictions = {}, {}
        for split, dataset in [("train", train), ("validation", validation)]:
            p = predictor.predict_proba(dataset)
            metrics[split] = save_multilabel_evaluation(out, split, dataset.rows, _labels(dataset), p, CLASS_NAMES)
            saved_predictions[split] = p
        for key in ("macro_roc_auc", "macro_average_precision", "binary_cross_entropy"):
            if abs(metrics["validation"][key] - training["best_validation_metrics"][key]) > 1e-12:
                raise ValueError(f"Restored validation result differs from selected epoch: {key}")
        # Reconstruct the checkpoint and replay training examples only, before opening test inputs.
        replay_data = Subset(train, range(min(32, len(train))))
        replay_data.normalization = train.normalization
        replay_data.normalize = True
        replay = load_checkpoint(checkpoint).predict_proba(replay_data)
        difference = float(np.abs(replay - saved_predictions["train"][:len(replay)]).max())
        if difference != 0:
            raise ValueError(f"Checkpoint reconstruction differs on training examples: {difference}")
        write_json(out/"checkpoint_replay.json", {"split": "train", "records": len(replay),
                   "maximum_absolute_probability_difference": difference, "passed": True,
                   "test_replay_performed": False, "checkpoint_sha256": checkpoint_hash})
        _check_prior(protected)
        _check_frozen(config_path, config_hash, sources, cache, cache_hashes, checkpoint, checkpoint_hash)
        test = CachedECGDataset(cache, "test")
        test_started = utc_now()
        write_json(out/"test_inference_state.json", {"status": "started", "started_utc": test_started,
                   "checkpoint_sha256": checkpoint_hash, "threshold": 0.5, "inference_calls": 1})
        probability = predictor.predict_proba(test)
        truth = _labels(test)
        metrics["test"] = save_multilabel_evaluation(out, "test", test.rows, truth, probability, CLASS_NAMES)
        write_json(out/"test_inference_state.json", {"status": "completed", "started_utc": test_started,
                   "completed_utc": utc_now(), "checkpoint_sha256": checkpoint_hash,
                   "threshold": 0.5, "inference_calls": 1, "records": len(test)})
        ci = patient_cluster_bootstrap(truth, probability, [row["patient_id"] for row in test.rows], CLASS_NAMES,
                                       n_resamples=cfg["uncertainty"]["resamples"], seed=cfg["uncertainty"]["seed"])
        write_json(out/"test_patient_bootstrap.json", ci)
        _check_frozen(config_path, config_hash, sources, cache, cache_hashes, checkpoint, checkpoint_hash)
        _check_prior(protected)
        if log_path.read_bytes() != original_log:
            raise ValueError("Central log changed during this experiment")
        result = {"status": "completed", "experiment_id": run_id, "model": cfg["model"],
                  "started_utc": started, "completed_utc": utc_now(), "total_runtime_seconds": time.perf_counter()-timer,
                  "dataset_id": cfg["dataset_id"], "split_version": cfg["split_version"],
                  "training": training, "metrics": metrics, "checkpoint_sha256": checkpoint_hash,
                  "test_inference_calls": 1, "threshold": .5, "representation": cfg["sequence_interpretation"],
                  "comparison_limit": cfg["comparison_limit"], "model_directory": str(model_dir.relative_to(ROOT)),
                  "uncertainty": cfg["uncertainty"]}
        record.update(status="completed", macro_f1=metrics["test"]["macro_f1"], weighted_f1=metrics["test"]["weighted_f1"],
                      notes=f"Five-label PTB-XL task; primary macro AUROC={metrics['test']['macro_roc_auc']:.6f}; "
                            f"subset_accuracy={metrics['test']['subset_accuracy']:.6f}; label_accuracy={metrics['test']['label_accuracy']:.6f}. "
                            "Single-label accuracy/balanced_accuracy columns intentionally blank; no direct Mendeley ranking. "
                            "One seed/config, fixed0.5threshold, one final test inference.")
        append_experiment_log(record)
        logged = True
        if not log_path.read_bytes().startswith(original_log):
            raise ValueError("Prior three experiment rows changed")
        write_json(out/"preservation.json", {"protected_artifacts_checked": len(protected), "changed_artifacts": [],
                   "prior_three_model_rows_preserved_as_exact_byte_prefix": True,
                   "prior_log_sha256": sha256_file(out/"prior_experiment_log.csv"),
                   "baseline_preservation": _check_prior(protected)})
        write_json(out/"run.json", result)
        write_csv(out/"run_summary.csv", [{"run_id": run_id, "dataset": cfg["dataset_id"], "task": "five-label",
                  **{key: metrics["test"][key] for key in ["macro_roc_auc", "macro_average_precision", "macro_f1", "micro_f1", "subset_accuracy", "label_accuracy"]}}])
        print(json.dumps({"run_id": run_id, "status": "completed", "test": metrics["test"],
                          "total_runtime_seconds": result["total_runtime_seconds"]}, indent=2), flush=True)
        return result
    except BaseException as exc:
        # Appending may succeed before a post-write check raises. Reconcile the
        # actual log before deciding whether another append is permissible.
        try:
            with log_path.open() as handle:
                logged = any(row["experiment_id"] == run_id for row in csv.DictReader(handle))
        except (OSError, csv.Error, KeyError):
            pass
        failure = {"status": "failed", "experiment_id": run_id, "started_utc": started, "failed_utc": utc_now(),
                   "error": repr(exc), "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter()-timer,
                   "central_log_already_appended": logged}
        write_json(out/"failure.json", failure)
        write_json(out/"run.json", failure)
        if not logged:
            record.update(status="failed", notes=repr(exc))
            try:
                append_experiment_log(record)
            except BaseException as log_exc:
                write_json(out/"logging_failure.json", {"error": repr(log_exc), "record_not_appended": record})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.run_id)
