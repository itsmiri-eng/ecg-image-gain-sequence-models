"""Read-only saved-artifact verifier for the separate hybrid ViT experiment.

No model construction, inference, data fitting or bootstrap resampling occurs.
The optional CLI writes a new verification file exclusively; it never overwrites.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

from src.approved_source_revisions import current_source_sha256
from src.portable_cache import located, relocate
from src.evaluation.multilabel import evaluate_multilabel
from src.models.ptbxl_hybrid_vit import HEAD_CONFIGURATION
from src.verify_ptbxl_run import (
    CLASS_NAMES, COUNTS, INPUT_SHAPE, SOURCE_FILES, CI_METRICS, TOLERANCE,
    _require, _sha, _hash, _safe_path, _json, _csv, _compare, _integer, _number,
    _timestamp, _verify_predictions, _verify_per_label, _verify_history,
    _verify_bootstrap,
)
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "experiments/configs/ptbxl_hybrid_vit_image_v1.json"
PRETRAINING_PATH = "experiments/protocols/ptbxl_hybrid_vit_image_v1/pretraining_freeze.json"
PRIOR_PATH = "experiments/protocols/ptbxl_hybrid_vit_image_v1/prior_artifacts.json"
PRIOR_CONFIG_PATH = "experiments/configs/ptbxl_cnn_lstm_image_v1.json"
CODE_FILES = [
    "src/run_ptbxl_hybrid_vit.py", "src/models/ptbxl_hybrid_vit.py",
    "src/training/__init__.py", "src/training/ptbxl_hybrid_vit.py",
    "src/ptbxl_learning_data.py", "src/evaluation/multilabel.py",
    "src/evaluation/ptbxl_uncertainty.py", "src/common.py", "src/baseline_preservation.py",
]


def _verify_log(prior_path, log_path, run_id, cfg, run, metrics):
    old_bytes, current_bytes = prior_path.read_bytes(), log_path.read_bytes()
    _require(current_bytes.startswith(old_bytes), "Prior experiment log is not an exact byte prefix")
    old_fields, old_rows = _csv(prior_path)
    _require(len(old_rows) >= 5, "Prior experiment log must contain at least five completed model rows")
    _require(all(row["status"] == "completed" for row in old_rows[:5]), "Original five runs must be complete")
    fields, rows = _csv(log_path)
    _compare(old_fields, fields, "experiment log schema")
    _compare(old_rows, rows[:len(old_rows)], "prior experiment log rows")
    ids = [row["experiment_id"] for row in rows]
    _require(len(ids) == len(set(ids)) and run_id not in [row["experiment_id"] for row in old_rows],
             "Duplicate experiment ID or current run present in prior log")
    matches = [row for row in rows if row["experiment_id"] == run_id]
    _require(len(matches) == 1, "Expected exactly one logged current experiment")
    row = matches[0]
    expected = {"status": "completed", "model": "PTBXL_Image_HybridViT", "date": run["started_utc"],
                "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
                "random_seed": str(cfg["seed"]), "accuracy": "", "balanced_accuracy": "",
                "output_directory": f"results/ptbxl_hybrid_vit/{run_id}"}
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


def _verify_cache(root, cfg, provenance, *, full_cache=True):
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
        _require(re.fullmatch(r"[0-9a-f]{64}", row["cache_sha256"]), "Invalid cached encoded hash")
        if full_cache:
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
    return split_rows, norm, {"encoded_images_hashed": len(rows) if full_cache else 0, "full_cache": full_cache, "encoded_image_bytes_hashed": bytes_checked,
                             "decoded_pixels_recomputed": False, "metadata_hashes_verified": 3,
                             "partition_records": COUNTS, "unique_patients": len(patient_folds)}


def _verify_protocol(cfg, root):
    previous = _json(root / PRIOR_CONFIG_PATH)
    for key in ("dataset_id", "source_dataset_root", "cache_root", "split_version", "class_names",
                "input_shape", "source_manifest_sha256", "source_verification_sha256",
                "source_protocol_sha256", "seed", "preprocessing", "loss", "selection_rule",
                "threshold", "threshold_policy", "uncertainty"):
        _compare(previous[key], cfg[key], f"unchanged comparison protocol.{key}")
    old_training = {k: v for k, v in previous["training"].items() if k not in ("hidden_size", "device")}
    new_training = {k: v for k, v in cfg["training"].items()
                    if k not in ("embed_dim", "num_heads", "num_layers", "feedforward_dim", "device")}
    _compare(old_training, new_training, "unchanged training budget and optimizer")
    for key, value in {"seed": 42, "max_epochs": 20, "patience": 4,
                       "embed_dim": 32, "num_heads": 4, "num_layers": 2, "feedforward_dim": 64}.items():
        _compare(value, cfg["training"][key], f"prespecified training.{key}")
    _require(cfg["training"]["device"] in ("auto", "cpu", "cuda", "mps"), "Unsupported requested device")
    for key, value in {"feature_sequence": [70, 96], "head_configuration": HEAD_CONFIGURATION,
                       "trainable_parameters": 23881, "pretraining": False, "streaming_causality": False}.items():
        _compare(value, cfg["architecture"][key], f"hybrid architecture.{key}")


def _verify_protection(root, out, source):
    lock = _json(root / PRIOR_PATH)
    baseline_path = root / "experiments/protocols/ptbxl_hybrid_vit_image_v1/baseline_experiment_log.csv"
    _hash(baseline_path, lock["baseline_log_sha256"])
    baseline_rows = _csv(baseline_path)[1]
    _require(len(baseline_rows) == 5 and all(row["status"] == "completed" for row in baseline_rows),
             "Protocol baseline must contain five completed runs")
    prior_path = out / "prior_experiment_log.csv"
    _require(prior_path.read_bytes().startswith(baseline_path.read_bytes()), "Prior log lost original five-run prefix")
    protected = _json(out / "protected_artifacts.json")
    _require(isinstance(protected, dict) and protected, "Missing protected-artifact snapshot")
    for name, digest in lock["sha256"].items():
        _compare(digest, protected.get(name), f"frozen prior-artifact declaration.{name}")
    for row in _csv(prior_path)[1]:
        directory = _safe_path(root, row["output_directory"])
        run_path = directory / "run.json"
        if run_path.is_file():
            prior_run = _json(run_path)
            directories = [directory]
            if prior_run.get("model_directory"):
                directories.append(_safe_path(root, prior_run["model_directory"]))
            else:
                # Failed attempts may have model artifacts but no model_directory field.
                for candidate in (root / "models").glob(f"*/{row['experiment_id']}"):
                    if candidate.is_dir():
                        directories.append(candidate)
            required = {p.relative_to(root).as_posix() for parent in directories for p in parent.rglob("*") if p.is_file()}
            _require(required <= set(protected), f"Protected snapshot omits artifacts for {row['experiment_id']}")
    for name in SOURCE_FILES:
        path = source / name
        _compare(_sha(path), protected.get(path.relative_to(root).as_posix()), f"protected dataset.{name}")
    for name, digest in protected.items():
        _hash(_safe_path(root, name), current_source_sha256(name, digest))
    preserved = _json(out / "preservation.json")
    for key, value in {"protected_artifacts_checked": len(protected), "changed_artifacts": [],
                       "prior_model_rows_preserved_as_exact_byte_prefix": True,
                       "prior_log_sha256": _sha(prior_path)}.items():
        _compare(value, preserved[key], f"preservation.{key}")
    original = _json(root / "experiments/reviews/2026-09-24/baseline_lock.json")["files"]
    for name, digest in original.items():
        if name != "experiments/experiment_log.csv":
            _hash(_safe_path(root, name), current_source_sha256(name, digest))
    original_log = root / "experiments/protocols/cnn_lstm_image_v1/baseline_log_before.csv"
    _hash(original_log, original["experiments/experiment_log.csv"])
    _require(prior_path.read_bytes().startswith(original_log.read_bytes()), "Original two-run log prefix changed")
    for key, value in {"immutable_baseline_artifacts_checked": len(original) - 1,
                       "changed_baseline_artifacts": [], "original_log_sha256": _sha(original_log),
                       "original_log_byte_prefix_preserved": True,
                       "original_log_rows_preserved": len(_csv(original_log)[1])}.items():
        _compare(value, preserved["baseline_preservation"][key], f"baseline preservation.{key}")
    return {"protected_artifacts_hashed": len(protected), "baseline_five_run_artifacts": len(lock["sha256"]),
            "original_baseline_artifacts_hashed": len(original) - 1, "changed_artifacts": []}


def _verify_device(training, cfg):
    probe = training["device_probe"]
    _compare(cfg["training"]["device"], probe["requested"], "device requested")
    _compare(training["device"], probe["resolved"], "device resolved")
    _require(training["device"] in ("cpu", "cuda", "mps"), "Unsupported resolved device")
    _compare(True, probe["synthetic_only"], "device probe synthetic scope")
    _compare(False, probe["ecg_data_accessed"], "device probe ECG access")
    if probe["requested"] != "auto":
        _compare(probe["requested"], probe["resolved"], "explicit device cannot silently fall back")
    _require(isinstance(probe["attempts"], list) and probe["attempts"], "Missing device probe attempts")
    final = probe["attempts"][-1]
    for key, value in {"device": training["device"], "status": "passed", "output_shape": [1, 5],
                       "finite_forward_backward": True, "synthetic_records": 1}.items():
        _compare(value, final[key], f"successful device probe.{key}")
    _compare(True, probe["available"][training["device"]], "resolved device available")
    _require(all(a["status"] in ("failed", "unavailable") and isinstance(a.get("reason"), str)
                 for a in probe["attempts"][:-1]), "Invalid device fallback evidence")
    _compare([a["reason"] for a in probe["attempts"][:-1]], probe["fallback_reasons"], "device fallback reasons")


def _expected_state_shapes():
    result = {"features.0.weight": (12, 1, 5, 5), "features.1.weight": (8, 12, 1, 1),
              "features.4.weight": (8, 1, 3, 3), "features.5.weight": (16, 8, 1, 1),
              "features.8.weight": (16, 1, 3, 3), "features.9.weight": (24, 16, 1, 1),
              "projection.weight": (32, 96), "projection.bias": (32,),
              "position_embedding": (1, 70, 32), "classifier.weight": (5, 32), "classifier.bias": (5,)}
    for prefix, channels in (("features.2", 8), ("features.6", 16), ("features.10", 24), ("final_norm", 32)):
        for suffix in ("weight", "bias"):
            result[f"{prefix}.{suffix}"] = (channels,)
    for layer in range(2):
        prefix = f"transformer_layers.{layer}"
        for suffix, shape in {"self_attn.in_proj_weight": (96, 32), "self_attn.in_proj_bias": (96,),
                              "self_attn.out_proj.weight": (32, 32), "self_attn.out_proj.bias": (32,),
                              "linear1.weight": (64, 32), "linear1.bias": (64,),
                              "linear2.weight": (32, 64), "linear2.bias": (32,),
                              "norm1.weight": (32,), "norm1.bias": (32,),
                              "norm2.weight": (32,), "norm2.bias": (32,)}.items():
            result[f"{prefix}.{suffix}"] = shape
    return result


def verify(run_id, *, root=ROOT, full_cache=True):
    """Verify completed saved artifacts; return a JSON-safe report without writes."""
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id), "Invalid run ID")
    _require(type(full_cache) is bool, "full_cache must be boolean")
    root = Path(root).resolve()
    out = root / "results/ptbxl_hybrid_vit" / run_id
    model_dir = root / "models/ptbxl_hybrid_vit" / run_id
    run, cfg, provenance = _json(out / "run.json"), _json(out / "config.json"), _json(out / "provenance.json")
    _compare("completed", run["status"], "run status")
    _compare(run_id, run["experiment_id"], "run identity")
    for parent, names in ((out, ["failure.json", "logging_failure.json"]),
                          (model_dir, [".training.lock", ".best.pt.tmp"])):
        _require(not any((parent / name).exists() for name in names), f"Failure/unfinished artifacts remain in {parent}")
    for name in ("best.pt", "config.json", "architecture.txt", "history.csv", "training_curve.png", "metadata.json", "progress.json"):
        _require((model_dir / name).is_file() and (model_dir / name).stat().st_size > 0,
                 f"Missing/empty completed model artifact: {name}")
    for key, value in {"model": "ptbxl_image_hybrid_vit", "class_names": CLASS_NAMES,
                       "input_shape": INPUT_SHAPE, "threshold": .5, "seed": 42}.items():
        _compare(value, cfg[key], f"configuration.{key}")
    _verify_protocol(cfg, root)
    _compare(_json(root / CONFIG_PATH), cfg, "saved and original configuration")
    _hash(root / CONFIG_PATH, provenance["configuration_sha256"])
    pretraining = _json(root / PRETRAINING_PATH)
    _compare(False, pretraining["ecg_training_started"], "pretraining ECG fit declaration")
    _compare(False, pretraining["test_outcome_used_for_design"], "pretraining test-outcome declaration")
    sources = provenance["source_sha256"]
    _compare(set(CODE_FILES), set(sources), "declared training source files")
    _compare(provenance["configuration_sha256"], pretraining["configuration_sha256"], "pretraining configuration freeze")
    _compare(sources, pretraining["source_sha256"], "pretraining source freeze")
    _hash(root / PRETRAINING_PATH, provenance["pretraining_freeze_sha256"])
    _require(isinstance(pretraining["evidence_sha256"], dict) and pretraining["evidence_sha256"], "Missing pretraining evidence")
    for name, digest in pretraining["evidence_sha256"].items():
        _hash(_safe_path(root, name), current_source_sha256(name, digest))
    _require(_timestamp(pretraining["created_utc"], "pretraining freeze") <= _timestamp(run["started_utc"], "run start"),
             "Pretraining freeze postdates the run start")
    _hash(root / PRIOR_PATH, cfg["prior_artifacts_sha256"])
    _compare(cfg["prior_artifacts_sha256"], provenance["prior_artifacts_sha256"], "run prior-artifact inventory")
    snapshots = {path.relative_to(out / "source_snapshot").as_posix() for path in (out / "source_snapshot").rglob("*") if path.is_file()}
    _compare(set(CODE_FILES), snapshots, "source snapshot inventory")
    for name, digest in sources.items():
        _hash(root / name, current_source_sha256(name, digest))
        _hash(out / "source_snapshot" / name, digest)
    for key in ("selection_rule", "test_policy"):
        _compare(cfg[key], provenance[key], f"provenance.{key}")
    _compare(run["started_utc"], provenance["started_utc"], "run start timestamp")
    _compare(provenance["git_commit"], run["git_commit"], "Git provenance")
    _require(run["git_commit"] is None or (isinstance(run["git_commit"], str) and re.fullmatch(r"[0-9a-f]{40,64}", run["git_commit"])),
             "Git commit must be an explicit null or hexadecimal commit ID")
    source = _safe_path(root, cfg["source_dataset_root"])
    for name, key in [("classification_manifest.csv", "source_manifest_sha256"),
                      ("verification.json", "source_verification_sha256"), ("protocol.json", "source_protocol_sha256")]:
        _hash(source / name, cfg[key])
    for key in ("model", "dataset_id", "split_version", "threshold", "uncertainty"):
        _compare(cfg[key], run[key], f"run.{key}")
    _compare(cfg["sequence_interpretation"], run["representation"], "run representation")
    _compare(cfg["comparison_limit"], run["comparison_limit"], "run comparison limit")
    _compare(model_dir.relative_to(root).as_posix(), run["model_directory"], "model directory")
    split_rows, normalization, cache_report = _verify_cache(root, cfg, provenance, full_cache=full_cache)
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
    training_expected = {"model": "LeadImageHybridViT", "config": cfg["training"], "input_shape": INPUT_SHAPE,
                         "class_names": CLASS_NAMES, "normalization": normalization,
                         "training_records": COUNTS["train"], "validation_records": COUNTS["validation"],
                         "class_counts_train": train_counts, "test_data_accessed": False,
                         "class_weighting": "none", "augmentation": "none", "pretrained_weights": False,
                         "resume_supported": False,
                         "checkpoint_path": located(training["checkpoint_path"], model_dir / "best.pt", root),
                         "history_path": located(training["history_path"], model_dir / "history.csv", root)}
    for key, value in training_expected.items():
        _compare(value, training[key], f"training.{key}")
    _verify_device(training, cfg)
    for key, value in {"sequence_shape": [70, 96], "head_configuration": HEAD_CONFIGURATION,
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
    checkpoint_expected = {"format_version": 1, "model_name": "LeadImageHybridViT", "class_names": CLASS_NAMES,
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
    _compare(23881, training["num_parameters"], "prespecified parameter count")
    _compare(training["num_parameters"], sum(value.numel() for value in state.values()), "checkpoint parameter count")
    _compare(_expected_state_shapes(), {key: tuple(value.shape) for key, value in state.items()}, "checkpoint tensor shapes")
    _require(all(value.dtype == torch.float32 for value in state.values()), "Checkpoint weights must be float32")
    frozen, replay, test_state = (_json(out / name) for name in
                                  ("selection_frozen.json", "checkpoint_replay.json", "test_inference_state.json"))
    for key, value in {"checkpoint_sha256": checkpoint_hash, "configuration_sha256": provenance["configuration_sha256"],
                       "selection_rule": cfg["selection_rule"], "threshold": .5, "training": training,
                       "test_inference_started": False}.items():
        _compare(value, frozen[key], f"selection freeze.{key}")
    _compare({"split": "train", "records": 32, "maximum_absolute_probability_difference": 0.0,
              "passed": True, "test_replay_performed": False, "checkpoint_sha256": checkpoint_hash,
              "device": training["device"]}, replay, "checkpoint replay")
    _require(replay["maximum_absolute_probability_difference"] == 0.0,
             "Checkpoint replay difference must be exactly zero")
    for key, value in {"status": "completed", "checkpoint_sha256": checkpoint_hash, "threshold": .5,
                       "inference_calls": 1, "records": COUNTS["test"]}.items():
        _compare(value, test_state[key], f"test inference state.{key}")
    _compare(1, run["test_inference_calls"], "run test inference count")
    times = [(pretraining["created_utc"], "pretraining freeze"), (run["started_utc"], "run start"), (frozen["frozen_utc"], "checkpoint freeze"),
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
            "pretraining_freeze_sha256": provenance["pretraining_freeze_sha256"], "git_commit": run["git_commit"],
            "pretraining_evidence_files_verified": len(pretraining["evidence_sha256"]),
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
                     "Cache manifest integrity is always checked; encoded PNG bytes are hashed only when full_cache is true. Pixels are not decoded again."}


def write_verification(report, destination, *, root=ROOT):
    """Write one new audit artifact inside this run or the review directory."""
    _compare("passed", report["status"], "verification status")
    run_id = report["run_id"]
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id), "Invalid run ID")
    root, destination = Path(root).resolve(), Path(destination).resolve()
    own_run = root / "results/ptbxl_hybrid_vit" / run_id
    reviews = root / "experiments/reviews"
    _require(destination.suffix == ".json" and (destination.is_relative_to(own_run) or destination.is_relative_to(reviews)),
             "Verification output must be a new JSON inside this hybrid run or experiments/reviews")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, help="New JSON under this run or experiments/reviews; never overwritten")
    parser.add_argument("--skip-cache-hashes", action="store_true", help="Repeat-report mode: verify manifests but skip PNG byte hashing")
    args = parser.parse_args(argv)
    destination = args.output or ROOT / "results/ptbxl_hybrid_vit" / args.run_id / "verification.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {destination}")
    report = verify(args.run_id, full_cache=not args.skip_cache_hashes)
    write_verification(report, destination)
    print(json.dumps({"status": report["status"], "run_id": args.run_id,
                      "verification": str(destination), "full_cache": report["cache"]["full_cache"]}, indent=2))


if __name__ == "__main__":
    main()
