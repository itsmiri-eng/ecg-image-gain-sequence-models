"""Shared training loop for every PTB-XL image network (protocol v2, November 2026).

One loop serves the CNN-only baseline, CNN-LSTM, temporal-convolution and hybrid
ViT models, so that matched comparisons and repeated seeds differ only in the
model factory. The loop mirrors the frozen v1 runners: AdamW, cosine schedule,
unweighted BCE, gradient clipping, validation-only checkpoint selection on
(macro AUROC, macro AP, -BCE), patience early stopping. No test data is accepted.
Inputs carry frozen training-only normalization from CachedECGDataset.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.multilabel import evaluate_multilabel
from src.models.ptbxl_cnn_lstm import LeadImageCNNLSTM
from src.models.ptbxl_cnn_pool import LeadImageCNNPool
from src.models.ptbxl_hybrid_vit import LeadImageHybridViT
from src.models.ptbxl_spatiotemporal import LeadImageSpatiotemporal

INPUT_SHAPE = (12, 120, 1120)
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
MODELS = {
    "cnn_pool": LeadImageCNNPool,
    "cnn_lstm": LeadImageCNNLSTM,
    "temporal_conv": LeadImageSpatiotemporal,
    "hybrid_vit": LeadImageHybridViT,
}
DEFAULT_CONFIG = {
    "model": None, "seed": 42, "num_threads": 4, "batch_size": 16, "max_epochs": 20, "patience": 4,
    "learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.2, "gradient_clip_norm": 1.0,
    "device": "cpu", "scheduler": "cosine", "eta_min": 0.00001,
}
SELECTION_RULE = ("Maximize validation macro ROC-AUC, then macro average precision, then minimize "
                  "probability BCE; earliest exact tie")
ARTIFACT_NAMES = ("best.pt", ".best.pt.tmp", "config.json", "architecture.txt", "history.csv",
                  "training_curve.png", "metadata.json", "progress.json", ".training.lock")


def get_config(config):
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **config}
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    for key in ("num_threads", "batch_size", "max_epochs", "patience"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if cfg["device"] not in ("cpu", "mps", "cuda") or cfg["scheduler"] not in ("cosine", "none"):
        raise ValueError("device must be cpu/mps/cuda and scheduler cosine/none")
    for key in ("learning_rate", "gradient_clip_norm", "weight_decay", "eta_min", "dropout"):
        if not isinstance(cfg[key], (int, float)) or isinstance(cfg[key], bool) or not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite numeric")
    if cfg["learning_rate"] <= 0 or cfg["eta_min"] > cfg["learning_rate"] or not 0 <= cfg["dropout"] < 1:
        raise ValueError("Invalid learning rate, eta_min or dropout")
    return cfg


def build_model(cfg):
    return MODELS[cfg["model"]](dropout=cfg["dropout"])


def seed_everything(seed, num_threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def selection_key(metrics):
    values = [metrics[key] for key in ("macro_roc_auc", "macro_average_precision", "binary_cross_entropy")]
    if any(value is None for value in values) or not np.isfinite(values).all():
        raise ValueError("Checkpoint selection requires finite validation AUROC, AP and BCE")
    return float(values[0]), float(values[1]), -float(values[2])


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _check_batch(images, targets):
    if images.dtype != torch.float32 or tuple(images.shape[1:]) != INPUT_SHAPE or not torch.isfinite(images).all():
        raise ValueError("Require finite float32 image batches N x 12 x 120 x 1120")
    if tuple(targets.shape) != (len(images), 5) or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("Require five-column binary targets")


@torch.inference_mode()
def predict_proba(model, dataset, batch_size=16, label="dataset"):
    """Independent sigmoid probabilities in dataset order (never row-normalized)."""
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    started, out = time.perf_counter(), []
    for batch, (images, targets) in enumerate(loader, start=1):
        _check_batch(images, targets)
        p = torch.sigmoid(model(images.to(device))).cpu().numpy()
        if not np.isfinite(p).all():
            raise RuntimeError("Nonfinite prediction probabilities")
        out.append(p)
        if batch == 1 or batch % 200 == 0 or batch == len(loader):
            print(f"{_utc_now()} predict:{label} batch {batch}/{len(loader)} "
                  f"seconds={time.perf_counter() - started:.1f}", flush=True)
    return np.concatenate(out, axis=0)


def _plot_history(history, best_epoch, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_online_bce"] for row in history], label="Train (online)")
    axes[0].plot(epochs, [row["val_binary_cross_entropy"] for row in history], label="Validation")
    axes[0].set_ylabel("Binary cross-entropy")
    for axis, key, name in zip(axes[1:], ("macro_roc_auc", "macro_average_precision"),
                               ("Validation macro ROC-AUC", "Validation macro AP")):
        axis.plot(epochs, [row[f"val_{key}"] for row in history])
        axis.set_ylabel(name)
    for axis in axes:
        axis.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1, label="Selected epoch")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(train_dataset, val_dataset, config, output_dir, normalization):
    """Fresh training only; refuses to overwrite an existing model directory."""
    started = time.perf_counter()
    cfg = get_config(config)
    if normalization.get("fit_partition") != "train":
        raise ValueError("Normalization must be fitted on the training partition only")
    for dataset, split in ((train_dataset, "train"), (val_dataset, "validation")):
        if getattr(dataset, "split", split) != split or getattr(dataset, "normalization", normalization) != normalization:
            raise ValueError(f"{split} dataset has the wrong partition or normalization")
    out = Path(output_dir)
    existing = [name for name in ARTIFACT_NAMES if (out / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite an existing run: {existing}")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".training.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    progress = {"status": "running", "epoch": 0, "max_epochs": cfg["max_epochs"], "test_data_accessed": False}
    try:
        seed_everything(cfg["seed"], cfg["num_threads"])
        device = torch.device(cfg["device"])
        model = build_model(cfg).to(device)
        parameters = sum(p.numel() for p in model.parameters())
        generator = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
                                  generator=generator)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"], eta_min=cfg["eta_min"])
                     if cfg["scheduler"] == "cosine" else None)
        criterion = nn.BCEWithLogitsLoss()
        _atomic_json(out / "config.json", cfg)
        (out / "architecture.txt").write_text(f"{model}\n\nParameters: {parameters:,}\n{SELECTION_RULE}\n"
                                              "Unweighted BCE; no class weighting, augmentation or pretrained weights.\n")
        val_truth = np.asarray([[row[f"label_{n}"] for n in CLASS_NAMES] for row in val_dataset.rows], dtype=np.int8)
        history, best_key, best_epoch, stale = [], None, 0, 0
        for epoch in range(1, cfg["max_epochs"] + 1):
            epoch_started = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train()
            loss_sum, seen, max_grad = 0.0, 0, 0.0
            for batch, (images, targets) in enumerate(train_loader, start=1):
                _check_batch(images, targets)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images.to(device)), targets.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss at epoch {epoch}, batch {batch}")
                loss.backward()
                grad = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                max_grad = max(max_grad, float(grad))
                optimizer.step()
                seen += len(images)
                loss_sum += float(loss.detach()) * len(images)
                if batch == 1 or batch % 200 == 0 or batch == len(train_loader):
                    progress.update(phase="training", epoch=epoch, batch=batch, batches=len(train_loader),
                                    train_online_bce=loss_sum / seen, best_epoch=best_epoch,
                                    elapsed_seconds=time.perf_counter() - started, updated_utc=_utc_now())
                    _atomic_json(out / "progress.json", progress)
                    print(f"{_utc_now()} {cfg['model']} epoch {epoch}/{cfg['max_epochs']} batch {batch}/{len(train_loader)} "
                          f"train_bce={loss_sum / seen:.5f} seconds={time.perf_counter() - epoch_started:.1f}", flush=True)
            if seen != len(train_dataset):
                raise ValueError("Not every training record was seen in the epoch")
            validation = evaluate_multilabel(val_truth, predict_proba(model, val_dataset, cfg["batch_size"], "validation"),
                                             CLASS_NAMES)
            key = selection_key(validation)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_epoch, stale = key, epoch, 0
                temporary = out / ".best.pt.tmp"
                torch.save({"format_version": 2, "model": cfg["model"], "config": cfg, "normalization": normalization,
                            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "best_epoch": epoch, "validation_selection_key": list(key),
                            "best_validation_metrics": validation, "selection_rule": SELECTION_RULE}, temporary)
                temporary.replace(out / "best.pt")
            else:
                stale += 1
            row = {"epoch": epoch, "learning_rate": learning_rate, "train_online_bce": loss_sum / seen,
                   **{f"val_{k}": v for k, v in validation.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                   "checkpoint_improved": improved, "epochs_without_improvement": stale,
                   "max_gradient_norm_before_clip": max_grad, "epoch_seconds": time.perf_counter() - epoch_started}
            history.append(row)
            with (out / "history.csv").open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
            if scheduler is not None:
                scheduler.step()
            print(f"{_utc_now()} {cfg['model']} epoch {epoch}/{cfg['max_epochs']} train_bce={row['train_online_bce']:.5f} "
                  f"val_bce={validation['binary_cross_entropy']:.5f} val_macro_auc={validation['macro_roc_auc']:.5f} "
                  f"val_macro_ap={validation['macro_average_precision']:.5f} selected={improved} "
                  f"seconds={row['epoch_seconds']:.1f}", flush=True)
            if stale >= cfg["patience"]:
                break
        checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        metadata = {
            "model": cfg["model"], "model_class": type(model).__name__, "config": cfg, "num_parameters": parameters,
            "normalization": normalization, "class_weighting": "none", "augmentation": "none",
            "pretrained_weights": False, "loss": "Unweighted BCEWithLogitsLoss",
            "training_records": len(train_dataset), "validation_records": len(val_dataset), "test_data_accessed": False,
            "checkpoint_selection": SELECTION_RULE, "best_epoch": best_epoch, "best_selection_key": list(best_key),
            "epochs_run": len(history), "stopped_early": len(history) < cfg["max_epochs"],
            "best_validation_metrics": checkpoint["best_validation_metrics"],
            "selected_epoch_online_train_bce": history[best_epoch - 1]["train_online_bce"],
            "early_stopping": f"patience {cfg['patience']} epochs without strict lexicographic validation improvement",
            "scheduler": "CosineAnnealingLR stepped per epoch, T_max=max_epochs" if scheduler else "none",
            "training_seconds": time.perf_counter() - started, "device": str(device),
            "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                            "numpy": np.__version__, "num_threads": torch.get_num_threads(),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        }
        _plot_history(history, best_epoch, out / "training_curve.png",
                      f"PTB-XL {cfg['model']} seed {cfg['seed']} | validation selection")
        _atomic_json(out / "metadata.json", metadata)
        progress.update(status="complete", phase="complete", epochs_run=len(history), best_epoch=best_epoch)
        _atomic_json(out / "progress.json", progress)
        return model, metadata
    except BaseException as error:
        progress.update(status="failed", error=repr(error))
        _atomic_json(out / "progress.json", progress)
        raise
    finally:
        lock.unlink(missing_ok=True)


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = get_config(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint
