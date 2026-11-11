"""CPU CNN+LSTM for twelve aligned rendered ECG lead images.

Datasets supply images already standardized with training-only statistics.
Five sigmoid targets overlap: probabilities must never be row-normalized.
Training accepts training and validation datasets only, with no test argument.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.multilabel import evaluate_multilabel


INPUT_SHAPE = (12, 120, 1120)
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
DEFAULT_CONFIG = {
    "seed": 42, "num_threads": 4, "batch_size": 16, "num_workers": 0,
    "max_epochs": 20, "patience": 4, "learning_rate": 0.001,
    "weight_decay": 0.0001, "dropout": 0.2, "hidden_size": 32,
    "gradient_clip_norm": 1.0, "device": "cpu", "scheduler": "cosine",
    "eta_min": 0.00001,
}
SELECTION_RULE = "Maximize validation macro ROC-AUC, then macro average precision, then minimize probability BCE; earliest exact tie"
SEQUENCE_INTERPRETATION = (
    "Left-to-right spatial image-feature columns across aligned leads, including calibration; "
    "not recovered calibrated waveform samples"
)
ARTIFACT_NAMES = ("best.pt", ".best.pt.tmp", "config.json", "architecture.txt", "history.csv",
                  "training_curve.png", "metadata.json", "progress.json", ".training.lock")


def spatial_feature_sequence(feature_map):
    """N,C,R,W -> N,W,C*R: column order, then channels and vertical bands."""
    if feature_map.ndim != 4:
        raise ValueError("Expected feature map with dimensions N,C,R,W")
    return feature_map.permute(0, 3, 1, 2).contiguous().flatten(2)


class LeadImageCNNLSTM(nn.Module):
    """Depthwise lead filtering, joint lead features, then one forward LSTM."""
    def __init__(self, dropout=0.2, hidden_size=32):
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1)")
        self.features = nn.Sequential(
            nn.Conv2d(12, 12, 5, stride=(2, 4), padding=2, groups=12, bias=False),
            nn.Conv2d(12, 8, 1, bias=False), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False),
            nn.Conv2d(8, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 24, 1, bias=False), nn.GroupNorm(4, 24), nn.ReLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 70))
        self.lstm = nn.LSTM(96, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 5)

    def extract_sequence(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected image batch N x {INPUT_SHAPE}")
        return spatial_feature_sequence(self.spatial_pool(self.features(images)))

    def forward(self, images):
        outputs, _ = self.lstm(self.extract_sequence(images))
        return self.classifier(self.dropout(outputs.mean(dim=1)))


def selection_key(metrics):
    values = [metrics[key] for key in ("macro_roc_auc", "macro_average_precision", "binary_cross_entropy")]
    if any(value is None for value in values) or not np.isfinite(values).all():
        raise ValueError("Checkpoint selection requires finite full-five-label validation AUROC, AP and BCE")
    return float(values[0]), float(values[1]), -float(values[2])


def _get_config(config):
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown PTB-XL CNN+LSTM configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **config}
    for key in ("num_threads", "batch_size", "max_epochs", "patience", "hidden_size"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if cfg["device"] != "cpu" or cfg["num_workers"] != 0:
        raise ValueError("This deterministic CPU protocol requires device='cpu' and num_workers=0")
    if cfg["scheduler"] not in ("cosine", "none"):
        raise ValueError("scheduler must be 'cosine' or 'none'")
    for key in ("learning_rate", "gradient_clip_norm"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "eta_min"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["eta_min"] > cfg["learning_rate"]:
        raise ValueError("eta_min cannot exceed learning_rate")
    if not isinstance(cfg["dropout"], (int, float)) or not math.isfinite(cfg["dropout"]) or not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout must be finite and in [0,1)")
    return cfg


def _seed_everything(seed, num_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def _normalization(value):
    result = json.loads(json.dumps(value, allow_nan=False))
    if (result.get("fit_partition") != "train" or not np.isfinite([result.get("mean", np.nan), result.get("std", np.nan)]).all()
            or result["std"] <= 0):
        raise ValueError("Require finite training-only normalization with positive std")
    return result


def _check_dataset(dataset, normalization, split=None):
    if split is not None and hasattr(dataset, "split") and dataset.split != split:
        raise ValueError(f"Expected {split} dataset, got {dataset.split}")
    if hasattr(dataset, "normalize") and dataset.normalize is not True:
        raise ValueError("Dataset must apply the stored training normalization")
    if hasattr(dataset, "normalization"):
        supplied = _normalization(dataset.normalization)
        if any(supplied[key] != normalization[key] for key in ("mean", "std", "fit_partition")):
            raise ValueError("Dataset normalization differs from the checkpoint/training normalization")
    if hasattr(dataset, "metadata") and "class_names" in dataset.metadata:
        if list(dataset.metadata["class_names"]) != list(CLASS_NAMES):
            raise ValueError("Dataset target order differs from the five fixed PTB-XL labels")


def _check_batch(images, targets):
    if (not isinstance(images, torch.Tensor) or images.dtype != torch.float32 or images.device.type != "cpu"
            or images.ndim != 4 or tuple(images.shape[1:]) != INPUT_SHAPE or not torch.isfinite(images).all()):
        raise ValueError("Require finite CPU float32 image batches with shape N x 12 x 120 x 1120")
    if (not isinstance(targets, torch.Tensor) or tuple(targets.shape) != (len(images), 5)
            or targets.dtype != torch.float32 or targets.device.type != "cpu" or not torch.isfinite(targets).all()
            or not ((targets == 0) | (targets == 1)).all()):
        raise ValueError("Require CPU float32 five-column binary targets")


def _class_counts(value, n_train):
    if isinstance(value, dict):
        if set(value) != set(CLASS_NAMES):
            raise ValueError("class_counts must contain the five fixed labels")
        value = [value[name] for name in CLASS_NAMES]
    counts = np.asarray(value)
    if (counts.shape != (5,) or counts.dtype.kind not in "iu" or np.any(counts <= 0)
            or np.any(counts >= n_train)):
        raise ValueError("class_counts must be five positive integer training supports, each below n_train")
    return counts.astype(np.int64).tolist()


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class LeadImagePredictor:
    def __init__(self, model, normalization, config):
        self.model = model.cpu().eval()
        self.normalization = _normalization(normalization)
        self.config = _get_config(config)
        self.class_names = list(CLASS_NAMES)
        self.classes_ = np.asarray(CLASS_NAMES)

    @torch.inference_mode()
    def predict_proba(self, dataset):
        """Independent sigmoid scores in dataset order; inputs already normalized."""
        _check_dataset(dataset, self.normalization)
        if len(dataset) == 0:
            return np.empty((0, 5), dtype=np.float32)
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=0)
        self.model.eval()
        probabilities = []
        for images, targets in loader:
            _check_batch(images, targets)
            p = torch.sigmoid(self.model(images))
            if not torch.isfinite(p).all():
                raise RuntimeError("Nonfinite prediction probabilities")
            probabilities.append(p.numpy())
        return np.concatenate(probabilities, axis=0)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "LeadImageCNNLSTM"
            or checkpoint.get("input_shape") != list(INPUT_SHAPE)
            or checkpoint.get("class_names") != list(CLASS_NAMES)):
        raise ValueError("Unsupported PTB-XL CNN+LSTM checkpoint shape/model/labels")
    cfg = _get_config(checkpoint["config"])
    normalization = _normalization(checkpoint["normalization"])
    _seed_everything(cfg["seed"], cfg["num_threads"])
    model = LeadImageCNNLSTM(dropout=cfg["dropout"], hidden_size=cfg["hidden_size"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains nonfinite model parameters")
    return LeadImagePredictor(model, normalization, cfg)


@torch.inference_mode()
def _evaluate_validation(model, loader):
    model.eval()
    truths, probabilities = [], []
    for images, targets in loader:
        _check_batch(images, targets)
        truths.append(targets.numpy())
        probabilities.append(torch.sigmoid(model(images)).numpy())
    return evaluate_multilabel(np.concatenate(truths), np.concatenate(probabilities), CLASS_NAMES)


def _plot_history(history, best_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_online_bce"] for row in history], label="Train: online loss")
    axes[0].plot(epochs, [row["val_binary_cross_entropy"] for row in history], label="Validation: epoch-end")
    axes[0].set_ylabel("Unweighted binary cross-entropy")
    for axis, key, title in zip(axes[1:], ("macro_roc_auc", "macro_average_precision"), ("Macro ROC-AUC", "Macro average precision")):
        axis.plot(epochs, [row[f"val_{key}"] for row in history], label="Validation")
        axis.set(ylabel=title, ylim=(0, 1))
    for axis in axes:
        axis.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1, label="Selected epoch")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("PTB-XL lead-image CNN+LSTM | validation selection | five overlapping labels")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(train_dataset, val_dataset, config, output_dir, normalization, class_counts):
    """Fresh training only. Dataset objects own train-fitted input normalization.

    No resume is offered: existing run artifacts are rejected. Online training
    loss averages batch losses by record count; only validation gets a complete
    evaluation pass after each epoch. No test dataset is accepted or opened.
    """
    started = time.perf_counter()
    cfg = _get_config(config)
    norm = _normalization(normalization)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation datasets must be nonempty")
    _check_dataset(train_dataset, norm, "train")
    _check_dataset(val_dataset, norm, "validation")
    if "training_images" in norm and norm["training_images"] != len(train_dataset):
        raise ValueError("Normalization training count differs from the training dataset")
    counts = _class_counts(class_counts, len(train_dataset))
    out = Path(output_dir)
    existing = [str(out / name) for name in ARTIFACT_NAMES if (out / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite an existing PTB-XL CNN+LSTM run: {existing}")
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / ".training.lock"
    with lock_path.open("x") as handle:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, handle)
    progress = {"status": "running", "phase": "initializing", "epoch": 0,
                "max_epochs": cfg["max_epochs"], "training_records": len(train_dataset),
                "validation_records": len(val_dataset), "test_data_accessed": False}
    try:
        _atomic_json(out / "progress.json", progress)
        _seed_everything(cfg["seed"], cfg["num_threads"])
        model = LeadImageCNNLSTM(dropout=cfg["dropout"], hidden_size=cfg["hidden_size"])
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        generator = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"], eta_min=cfg["eta_min"])
                     if cfg["scheduler"] == "cosine" else None)
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        _atomic_json(out / "config.json", cfg)
        (out / "architecture.txt").write_text(
            f"{model}\n\nParameters: {parameter_count:,}\nInput: N x 12 x 120 x 1120, already train-standardized.\n"
            "Features: N x 24 x 15 x 70; pooled: N x 24 x 4 x 70; sequence: N x 70 x 96.\n"
            "First convolution filters each lead separately; pointwise convolutions combine aligned leads.\n"
            "Each sequence step flattens channels then vertical bands; mean LSTM outputs -> dropout -> five logits.\n"
            f"{SEQUENCE_INTERPRETATION}\nIndependent sigmoid probabilities, never row-normalized.\n"
            f"{SELECTION_RULE}\nNo class weighting, augmentation, pretrained weights, or test access.\n"
        )
        history, best_key, best_epoch, stale = [], None, 0, 0
        checkpoint_path = out / "best.pt"
        for epoch in range(1, cfg["max_epochs"] + 1):
            epoch_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train()
            weighted_loss, seen, max_gradient_norm = 0.0, 0, 0.0
            observed_counts = torch.zeros(5, dtype=torch.int64)
            for batch, (images, targets) in enumerate(train_loader, start=1):
                _check_batch(images, targets)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss at epoch {epoch}, batch {batch}")
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
                optimizer.step()
                seen += len(images)
                weighted_loss += float(loss.detach()) * len(images)
                observed_counts += targets.sum(dim=0).to(torch.int64)
                if batch == 1 or batch % 100 == 0 or batch == len(train_loader):
                    progress.update(status="running", phase="training", epoch=epoch, batch=batch,
                                    batches_in_epoch=len(train_loader), records_seen_this_epoch=seen,
                                    train_online_bce=weighted_loss / seen, learning_rate=learning_rate,
                                    elapsed_seconds=time.perf_counter() - started,
                                    epoch_seconds=time.perf_counter() - epoch_started,
                                    best_epoch=best_epoch, best_selection_key=list(best_key) if best_key is not None else None)
                    _atomic_json(out / "progress.json", progress)
                    print(f"PTB-XL CNN+LSTM epoch {epoch:02d}/{cfg['max_epochs']} batch {batch}/{len(train_loader)} "
                          f"train_online_bce={weighted_loss / seen:.5f} seconds={progress['epoch_seconds']:.1f}", flush=True)
            if seen != len(train_dataset) or observed_counts.tolist() != counts:
                raise ValueError("Observed training record/label counts differ from the supplied frozen counts")
            progress.update(phase="validation", elapsed_seconds=time.perf_counter() - started)
            _atomic_json(out / "progress.json", progress)
            validation = _evaluate_validation(model, val_loader)
            key = selection_key(validation)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_epoch, stale = key, epoch, 0
                checkpoint = {
                    "format_version": 1, "model_name": "LeadImageCNNLSTM", "input_shape": list(INPUT_SHAPE),
                    "class_names": list(CLASS_NAMES), "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": cfg, "normalization": norm, "class_counts_train": counts,
                    "training_records": len(train_dataset), "best_epoch": best_epoch,
                    "validation_selection_key": list(best_key), "best_validation_metrics": validation,
                    "selection_rule": SELECTION_RULE, "sequence_interpretation": SEQUENCE_INTERPRETATION,
                    "resume_supported": False,
                }
                temporary = out / ".best.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
            else:
                stale += 1
            row = {"epoch": epoch, "learning_rate": learning_rate, "train_online_bce": weighted_loss / seen,
                   **{f"val_{k}": v for k, v in validation.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                   "checkpoint_improved": improved, "epochs_without_selection_improvement": stale,
                   "max_gradient_norm_before_clip": max_gradient_norm, "epoch_seconds": time.perf_counter() - epoch_started}
            history.append(row)
            with (out / "history.csv").open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
            if scheduler is not None:
                scheduler.step()
            progress.update(phase="epoch_complete", best_epoch=best_epoch, best_selection_key=list(best_key),
                            epochs_without_selection_improvement=stale, validation_metrics=validation,
                            elapsed_seconds=time.perf_counter() - started)
            _atomic_json(out / "progress.json", progress)
            print(f"PTB-XL CNN+LSTM epoch {epoch:02d}/{cfg['max_epochs']} "
                  f"train_online_bce={row['train_online_bce']:.5f} val_bce={validation['binary_cross_entropy']:.5f} "
                  f"val_macro_auc={validation['macro_roc_auc']:.5f} val_macro_ap={validation['macro_average_precision']:.5f} "
                  f"selected={improved} seconds={row['epoch_seconds']:.1f}", flush=True)
            if stale >= cfg["patience"]:
                break
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        predictor = LeadImagePredictor(model, norm, cfg)
        metadata = {
            "model": "LeadImageCNNLSTM", "config": cfg, "input_shape": list(INPUT_SHAPE),
            "class_names": list(CLASS_NAMES), "num_parameters": parameter_count,
            "normalization": norm, "normalization_application": "Dataset applies stored training statistics; model/predictor does not normalize again",
            "class_counts_train": counts, "class_weighting": "none",
            "training_records": len(train_dataset), "validation_records": len(val_dataset),
            "test_data_accessed": False, "augmentation": "none", "pretrained_weights": False,
            "sequence_shape": [70, 96], "sequence_interpretation": SEQUENCE_INTERPRETATION,
            "sequence_order": "width left-to-right; channels then vertical bands within each step",
            "lead_processing": "Depthwise input convolution, then pointwise channel mixing across aligned leads",
            "lstm_layers": 1, "bidirectional": False, "sequence_aggregation": "mean of LSTM output steps",
            "loss": "Unweighted BCEWithLogitsLoss, mean over all record-label pairs",
            "training_metric_definition": "Online pre-update batch BCE, averaged by record count; not a full epoch-end training evaluation",
            "validation_metric_definition": "Full validation partition after each epoch with dropout disabled; BCE from independent sigmoid probabilities",
            "checkpoint_selection": SELECTION_RULE, "best_selection_key": list(best_key),
            "best_epoch": best_epoch, "epochs_run": len(history), "stopped_early": len(history) < cfg["max_epochs"],
            "best_validation_metrics": checkpoint["best_validation_metrics"],
            "selected_epoch_online_train_bce": history[best_epoch - 1]["train_online_bce"],
            "early_stopping": "Patience counts epochs without strict lexicographic validation improvement",
            "scheduler": "CosineAnnealingLR stepped after each epoch, T_max=max_epochs" if scheduler is not None else "none",
            "training_seconds": time.perf_counter() - started, "device": "cpu", "resume_supported": False,
            "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                            "torch": str(torch.__version__), "numpy": np.__version__,
                            "num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "reproducibility_limit": "CPU determinism configured; cross-version/hardware identity is not guaranteed",
            "checkpoint_path": str(checkpoint_path.resolve()), "history_path": str((out / "history.csv").resolve()),
        }
        _plot_history(history, best_epoch, out / "training_curve.png")
        metadata["runtime_seconds"] = time.perf_counter() - started
        _atomic_json(out / "metadata.json", metadata)
        progress.update(status="complete", phase="complete", epochs_run=len(history), best_epoch=best_epoch,
                        elapsed_seconds=time.perf_counter() - started)
        _atomic_json(out / "progress.json", progress)
        return predictor, metadata
    except BaseException as error:
        progress.update(status="failed", phase="failed", error_type=type(error).__name__, error=str(error),
                        elapsed_seconds=time.perf_counter() - started)
        _atomic_json(out / "progress.json", progress)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
