"""Small, deterministic four-class ECG image CNN baseline.

The caller owns the audited, frozen split and image preprocessing. This module
accepts training and validation arrays only; test data cannot influence fitting
or checkpoint selection. Inputs are grayscale ink arrays (N, 1, 128, 256).
"""

from __future__ import annotations

import csv
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_CONFIG = {
    "seed": 42,
    "max_epochs": 30,
    "patience": 7,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.25,
    "device": "auto",
    "num_threads": 4,
    "min_delta": 0.0,
    "num_classes": 4,
}
INPUT_SHAPE = (1, 128, 256)
ARTIFACT_NAMES = (
    "best.pt",
    "architecture.txt",
    "history.csv",
    "training_curve.png",
    "metadata.json",
)


class SmallECGCNN(nn.Module):
    """Three convolutions and a 2x4 spatial grid retain coarse lead layout.

    GroupNorm avoids estimating running statistics from the small minibatches.
    Dropout and weight decay regularize this deliberately small first baseline.
    No data augmentation is applied: ECG orientation and lead identity matter.
    """

    def __init__(self, dropout: float = 0.25, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 2 * 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _seed_everything(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _validate_x(x: np.ndarray, name: str, allow_empty: bool = False) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"{name} must have shape (N, 1, 128, 256); got {x.shape}")
    if not allow_empty and len(x) == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
        raise ValueError(f"{name} must contain finite floating-point ink values")
    if x.size and (x.min() < -1e-6 or x.max() > 1.0 + 1e-6):
        raise ValueError(f"{name} ink values must be in [0, 1]")
    return np.ascontiguousarray(x, dtype=np.float32)


def _validate_y(y: np.ndarray, n: int, name: str) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector with {n} entries")
    if (y < 0).any() or (y >= 4).any():
        raise ValueError(f"{name} labels must be in 0, 1, 2, 3")
    return np.ascontiguousarray(y, dtype=np.int64)


def _get_config(config: dict[str, Any]) -> dict[str, Any]:
    unknown = set(config).difference(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown CNN configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **config}
    for key in ("max_epochs", "patience", "batch_size", "num_threads"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if cfg["num_classes"] != 4:
        raise ValueError("This fixed baseline requires num_classes=4")
    if cfg["device"] not in ("auto", "cpu", "mps"):
        raise ValueError("device must be 'auto', 'cpu', or 'mps'")
    if not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    for key in ("learning_rate", "weight_decay", "min_delta"):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if cfg["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    return cfg


def _select_device(cfg: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    """Probe an actual deterministic MPS forward/backward step before fitting."""
    probe = {"requested": cfg["device"], "mps_available": False, "fallback_reason": None}
    if cfg["device"] == "cpu":
        return torch.device("cpu"), probe
    probe["mps_available"] = bool(torch.backends.mps.is_available())
    if not probe["mps_available"]:
        probe["fallback_reason"] = "MPS is unavailable in this Python/PyTorch runtime."
        return torch.device("cpu"), probe
    try:
        device = torch.device("mps")
        model = SmallECGCNN(cfg["dropout"], cfg["num_classes"]).to(device)
        batch = torch.zeros((cfg["batch_size"], *INPUT_SHAPE), device=device)
        labels = torch.arange(cfg["batch_size"], device=device) % cfg["num_classes"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        loss = F.cross_entropy(model(batch), labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("MPS smoke test returned a non-finite loss")
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        del optimizer, loss, model, batch, labels
        torch.mps.empty_cache()
        probe["smoke_test"] = "deterministic full-batch forward/backward/AdamW passed"
        return device, probe
    except (RuntimeError, NotImplementedError) as error:
        probe["fallback_reason"] = f"MPS smoke test failed: {type(error).__name__}: {error}"
        return torch.device("cpu"), probe


def _classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(
            recall_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y, pred, labels=range(4), average="weighted", zero_division=0)
        ),
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    targets, predictions = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        weighted_loss_sum += float(
            F.cross_entropy(logits, yb, weight=class_weights, reduction="sum").item()
        )
        weight_sum += float(class_weights[yb].sum().item())
        targets.append(yb.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    metrics = _classification_metrics(np.concatenate(targets), np.concatenate(predictions))
    metrics["loss"] = weighted_loss_sum / weight_sum
    return metrics


class TorchPredictor:
    """Apply saved training normalization and predict in bounded-size batches."""

    def __init__(
        self,
        model: SmallECGCNN,
        mean: float,
        std: float,
        device: torch.device,
        batch_size: int,
    ):
        self.model = model.to(device).eval()
        self.mean = float(mean)
        self.std = float(std)
        self.device = device
        self.batch_size = batch_size
        self.classes_ = np.arange(4, dtype=np.int64)

    @torch.inference_mode()
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = _validate_x(x, "prediction inputs", allow_empty=True)
        if not len(x):
            return np.empty((0, 4), dtype=np.float32)
        self.model.eval()
        probabilities = []
        for start in range(0, len(x), self.batch_size):
            batch = (x[start : start + self.batch_size] - self.mean) / self.std
            logits = self.model(torch.from_numpy(batch).to(self.device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)


def load_checkpoint(path: str | Path, device: str = "cpu") -> TorchPredictor:
    """Load this module's checkpoint without pickle-enabled arbitrary objects."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint["input_shape"] != list(INPUT_SHAPE):
        raise ValueError("Unsupported CNN checkpoint format or input shape")
    cfg = _get_config(checkpoint["config"])
    cfg["device"] = device
    cfg = _get_config(cfg)
    _seed_everything(cfg["seed"], cfg["num_threads"])
    selected_device, _ = _select_device(cfg)
    model = SmallECGCNN(cfg["dropout"], cfg["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return TorchPredictor(
        model,
        checkpoint["normalization"]["mean"],
        checkpoint["normalization"]["std"],
        selected_device,
        cfg["batch_size"],
    )


def _plot_history(history: list[dict[str, Any]], best_epoch: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for prefix, label in (("train", "Training"), ("val", "Validation")):
        axes[0].plot(epochs, [row[f"{prefix}_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row[f"{prefix}_macro_f1"] for row in history], label=label)
    for axis in axes:
        axis.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1, label="Selected epoch")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Class-weighted cross-entropy")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0, 1)
    fig.suptitle("Small ECG CNN: checkpoint selected only by validation loss")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    output_dir: str | Path,
) -> tuple[TorchPredictor, dict[str, Any]]:
    """Fit one prespecified baseline; restore the minimum-validation-loss epoch.

    Labels have the caller's stable 0..3 order. The model and all normalization
    and imbalance statistics are fitted exclusively to x_train/y_train.
    Validation determines early stopping only. Existing run files are protected.
    """
    started = time.perf_counter()
    cfg = _get_config(config)
    x_train = _validate_x(x_train, "x_train")
    x_val = _validate_x(x_val, "x_val")
    y_train = _validate_y(y_train, len(x_train), "y_train")
    y_val = _validate_y(y_val, len(x_val), "y_val")
    class_counts = np.bincount(y_train, minlength=4)
    if (class_counts == 0).any():
        raise ValueError("All four classes must occur in the training split")
    output_dir = Path(output_dir)
    existing = [str(output_dir / name) for name in ARTIFACT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite an existing CNN run: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg["seed"], cfg["num_threads"])
    device, device_probe = _select_device(cfg)
    # Reset RNG state after the optional disposable device probe.
    _seed_everything(cfg["seed"], cfg["num_threads"])
    mean = float(x_train.mean(dtype=np.float64))
    raw_std = float(x_train.std(dtype=np.float64))
    std = max(raw_std, 1e-6)
    normalization = {
        "method": "global training-pixel standardization",
        "fit_partition": "train",
        "mean": mean,
        "std": std,
        "raw_std": raw_std,
        "std_floor": 1e-6,
    }
    train_dataset = TensorDataset(
        torch.from_numpy((x_train - mean) / std), torch.from_numpy(y_train)
    )
    val_dataset = TensorDataset(torch.from_numpy((x_val - mean) / std), torch.from_numpy(y_val))
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    train_eval_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    class_weights_np = (len(y_train) / (4 * class_counts)).astype(np.float32)
    class_weights = torch.as_tensor(class_weights_np, device=device)
    model = SmallECGCNN(cfg["dropout"], cfg["num_classes"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    (output_dir / "architecture.txt").write_text(
        f"{model}\n\nTrainable parameters: {parameter_count:,}\n"
        f"Input: N x {INPUT_SHAPE}\nOutput: 4 logits in caller-defined class order\n"
        "Adaptive 2x4 spatial pooling preserves coarse ECG lead layout.\n"
        "No augmentation. GroupNorm; dropout; AdamW weight decay.\n"
        "Normalization and class weights use training data only.\n",
        encoding="utf-8",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    for epoch in range(1, cfg["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=class_weights)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        train_metrics = _evaluate(model, train_eval_loader, device, class_weights)
        val_metrics = _evaluate(model, val_loader, device, class_weights)
        if not np.isfinite(val_metrics["loss"]):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        # Always save the actual minimum loss; min_delta only controls patience.
        significant_improvement = val_metrics["loss"] < best_loss - cfg["min_delta"]
        is_best = val_metrics["loss"] < best_loss
        if is_best:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            checkpoint = {
                "format_version": 1,
                "model_name": "SmallECGCNN",
                "input_shape": list(INPUT_SHAPE),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "normalization": normalization,
                "config": cfg,
                "class_counts": class_counts.tolist(),
                "class_weights": class_weights_np.tolist(),
                "best_epoch": best_epoch,
                "validation_loss": best_loss,
            }
            temporary = output_dir / ".best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        stale_epochs = 0 if significant_improvement else stale_epochs + 1
        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "checkpoint_improved": is_best,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"CNN epoch {epoch:02d}/{cfg['max_epochs']} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if stale_epochs >= cfg["patience"]:
            break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    predictor = TorchPredictor(model, mean, std, device, cfg["batch_size"])
    best_row = history[best_epoch - 1]
    metadata = {
        "model": "SmallECGCNN",
        "config": cfg,
        "input_shape": list(INPUT_SHAPE),
        "num_parameters": parameter_count,
        "normalization": normalization,
        "augmentation": "none",
        "class_counts_train": class_counts.tolist(),
        "class_weights": class_weights_np.tolist(),
        "class_weight_formula": "n_train / (4 * training_class_count)",
        "loss": "class-weighted cross-entropy; aggregate divided by sum of target weights",
        "checkpoint_selection": "strict minimum validation weighted cross-entropy",
        "early_stopping": "validation loss; min_delta affects patience only",
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": len(history) < cfg["max_epochs"],
        "training_seconds": training_seconds,
        "best_train_metrics": {
            key.removeprefix("train_"): value for key, value in best_row.items() if key.startswith("train_")
        },
        "best_validation_metrics": {
            key.removeprefix("val_"): value for key, value in best_row.items() if key.startswith("val_")
        },
        "epoch_metric_definition": "Full-split evaluation after each epoch, with dropout disabled",
        "device": str(device),
        "device_probe": device_probe,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "reproducibility_limit": "Deterministic on this runtime; cross-device/version equality is not guaranteed.",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "history_path": str(history_path.resolve()),
    }
    _plot_history(history, best_epoch, output_dir / "training_curve.png")
    metadata["runtime_seconds"] = time.perf_counter() - started
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return predictor, metadata
