"""Small CNN+LSTM classifier of spatial columns in frozen ECG images.

The sequence axis follows image width. It is NOT calibrated time: the input
contains multiple ECG leads in a page layout and no waveform digitization.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import INPUT_SHAPE, _evaluate, _seed_everything, _validate_x, _validate_y


DEFAULT_CONFIG = {
    "seed": 42, "max_epochs": 30, "patience": 7, "batch_size": 16,
    "learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.25,
    "device": "cpu", "num_threads": 4, "num_classes": 4,
    "hidden_size": 32, "pool_rows": 2, "sequence_length": 16,
    "gradient_clip_norm": 1.0,
}
SEQUENCE_INTERPRETATION = "Left-to-right spatial feature columns; not calibrated time or digitized waveforms"
SELECTION_RULE = "Maximize validation macro F1, then balanced accuracy, then minimize weighted loss; earliest exact tie"
ARTIFACT_NAMES = ("best.pt", "architecture.txt", "config.json", "history.csv", "training_curve.png", "metadata.json")


def spatial_feature_sequence(feature_map: torch.Tensor) -> torch.Tensor:
    """N,C,R,W -> N,W,C*R, with channels then vertical bands within each step."""
    if feature_map.ndim != 4:
        raise ValueError("Expected feature map with dimensions N,C,R,W")
    return feature_map.permute(0, 3, 1, 2).contiguous().flatten(start_dim=2)


class SmallCNNLSTM(nn.Module):
    def __init__(self, hidden_size=32, pool_rows=2, sequence_length=16, dropout=0.25, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_rows, sequence_length))
        self.lstm = nn.LSTM(input_size=64 * pool_rows, hidden_size=hidden_size,
                            num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def extract_sequence(self, images):
        return spatial_feature_sequence(self.spatial_pool(self.features(images)))

    def forward(self, images):
        output, _ = self.lstm(self.extract_sequence(images))
        return self.classifier(self.dropout(output.mean(dim=1)))


def selection_key(metrics):
    """Larger tuples are better; Python's strict > retains earliest exact ties."""
    key = (float(metrics["macro_f1"]), float(metrics["balanced_accuracy"]), -float(metrics["loss"]))
    if not np.isfinite(key).all():
        raise ValueError("Checkpoint-selection metrics must be finite")
    return key


def _get_config(config):
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown CNN+LSTM configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **config}
    for key in ("max_epochs", "patience", "batch_size", "num_threads", "hidden_size", "pool_rows", "sequence_length"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg["seed"], int) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if cfg["device"] != "cpu":
        raise ValueError("This reproducible spatial CNN+LSTM baseline requires device='cpu'")
    if cfg["num_classes"] != 4 or cfg["pool_rows"] > 8 or cfg["sequence_length"] > 16:
        raise ValueError("Require four classes and pooling no larger than the 8x16 convolution feature map")
    if not np.isfinite(cfg["dropout"]) or not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout must be in [0,1)")
    for key in ("learning_rate", "gradient_clip_norm"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    return cfg


def _make_model(cfg):
    return SmallCNNLSTM(**{key: cfg[key] for key in ("hidden_size", "pool_rows", "sequence_length", "dropout", "num_classes")})


class CNNLSTMPredictor:
    def __init__(self, model, mean, std, batch_size):
        self.model = model.cpu().eval()
        self.mean, self.std, self.batch_size = float(mean), float(std), int(batch_size)
        self.classes_ = np.arange(4, dtype=np.int64)

    @torch.inference_mode()
    def predict_proba(self, images):
        images = _validate_x(images, "prediction inputs", allow_empty=True)
        if not len(images):
            return np.empty((0, 4), dtype=np.float32)
        self.model.eval()
        probabilities = []
        for start in range(0, len(images), self.batch_size):
            normalized = (images[start:start + self.batch_size] - self.mean) / self.std
            logits = self.model(torch.from_numpy(normalized))
            probabilities.append(torch.softmax(logits, dim=1).numpy())
        return np.concatenate(probabilities)

    def predict(self, images):
        return self.predict_proba(images).argmax(axis=1)


def load_checkpoint(path, device="cpu"):
    if device != "cpu":
        raise ValueError("This checkpoint loader supports CPU only")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version") != 1 or checkpoint.get("model_name") != "SmallCNNLSTM"
            or checkpoint.get("input_shape") != list(INPUT_SHAPE)):
        raise ValueError("Unsupported CNN+LSTM checkpoint")
    cfg = _get_config(checkpoint["config"])
    _seed_everything(cfg["seed"], cfg["num_threads"])
    model = _make_model(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    normalization = checkpoint["normalization"]
    if not np.isfinite([normalization["mean"], normalization["std"]]).all() or normalization["std"] <= 0:
        raise ValueError("Invalid checkpoint normalization")
    return CNNLSTMPredictor(model, normalization["mean"], normalization["std"], cfg["batch_size"])


def _plot_history(history, best_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    for axis, metric, label in zip(axes, ("loss", "macro_f1", "balanced_accuracy"),
                                   ("Weighted cross-entropy", "Macro F1", "Balanced accuracy")):
        for split in ("train", "val"):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=split)
        axis.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1, label="Selected epoch")
        axis.set(xlabel="Epoch", ylabel=label)
        if metric != "loss":
            axis.set_ylim(0, 1)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Spatial CNN+LSTM — select validation macro F1, then balanced accuracy, then loss")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_cnn_lstm(x_train, y_train, x_val, y_val, config, output_dir):
    """Train on supplied training/validation partitions, with no test argument."""
    started = time.perf_counter()
    cfg = _get_config(config)
    x_train, x_val = _validate_x(x_train, "x_train"), _validate_x(x_val, "x_val")
    y_train, y_val = _validate_y(y_train, len(x_train), "y_train"), _validate_y(y_val, len(x_val), "y_val")
    counts = np.bincount(y_train, minlength=4)
    if (counts == 0).any():
        raise ValueError("All four classes must occur in training")
    out = Path(output_dir)
    existing = [str(out / name) for name in ARTIFACT_NAMES if (out / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite an existing CNN+LSTM run: {existing}")
    out.mkdir(parents=True, exist_ok=True)
    _seed_everything(cfg["seed"], cfg["num_threads"])
    device = torch.device("cpu")
    mean, raw_std = float(x_train.mean(dtype=np.float64)), float(x_train.std(dtype=np.float64))
    std = max(raw_std, 1e-6)
    normalization = {"method": "global training-pixel standardization", "fit_partition": "train",
                     "mean": mean, "std": std, "raw_std": raw_std, "std_floor": 1e-6}
    training_data = TensorDataset(torch.from_numpy((x_train - mean) / std), torch.from_numpy(y_train))
    validation_data = TensorDataset(torch.from_numpy((x_val - mean) / std), torch.from_numpy(y_val))
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(training_data, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, generator=generator)
    train_eval_loader = DataLoader(training_data, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    val_loader = DataLoader(validation_data, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    weights_np = (len(y_train) / (4 * counts)).astype(np.float32)
    weights = torch.from_numpy(weights_np)
    model = _make_model(cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2, allow_nan=False) + "\n")
    (out / "architecture.txt").write_text(
        f"{model}\n\nTrainable parameters: {parameter_count:,}\nInput: N x 1 x 128 x 256\n"
        "Convolution output: N x 64 x 8 x 16\n"
        f"Pooled map: N x 64 x {cfg['pool_rows']} x {cfg['sequence_length']}\n"
        f"Sequence: N x {cfg['sequence_length']} x {64 * cfg['pool_rows']}\n"
        f"Single-layer unidirectional LSTM hidden size: {cfg['hidden_size']}\n"
        f"Mean across spatial sequence -> dropout -> four logits\n{SEQUENCE_INTERPRETATION}\n"
        f"{SELECTION_RULE}\nNo augmentation; no pretrained weights. CPU only.\n"
    )
    history, best_key, best_epoch, stale = [], None, 0, 0
    checkpoint_path = out / "best.pt"
    for epoch in range(1, cfg["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        maximum_gradient_norm = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
            optimizer.step()
        train_metrics = _evaluate(model, train_eval_loader, device, weights)
        val_metrics = _evaluate(model, val_loader, device, weights)
        key = selection_key(val_metrics)
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, stale = key, epoch, 0
            checkpoint = {
                "format_version": 1, "model_name": "SmallCNNLSTM", "input_shape": list(INPUT_SHAPE),
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "config": cfg, "normalization": normalization, "class_counts": counts.tolist(),
                "class_weights": weights_np.tolist(), "best_epoch": best_epoch,
                "validation_selection_key": list(best_key), "best_validation_metrics": val_metrics,
                "selection_rule": SELECTION_RULE, "sequence_interpretation": SEQUENCE_INTERPRETATION,
            }
            temporary = out / ".best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale += 1
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}, "checkpoint_improved": improved,
               "epochs_without_selection_improvement": stale, "max_gradient_norm_before_clip": maximum_gradient_norm,
               "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        with (out / "history.csv").open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(f"Spatial CNN+LSTM epoch {epoch:02d}/{cfg['max_epochs']} "
              f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
              f"selected={improved} seconds={row['epoch_seconds']:.2f}", flush=True)
        if stale >= cfg["patience"]:
            break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    predictor = CNNLSTMPredictor(model, mean, std, cfg["batch_size"])
    best_row = history[best_epoch - 1]
    metadata = {
        "model": "SmallCNNLSTM", "config": cfg, "input_shape": list(INPUT_SHAPE),
        "num_parameters": parameter_count, "normalization": normalization,
        "class_counts_train": counts.tolist(), "class_weights": weights_np.tolist(),
        "class_weight_formula": "n_train / (4 * training_class_count)",
        "sequence_interpretation": SEQUENCE_INTERPRETATION,
        "spatial_sequence_shape": [cfg["sequence_length"], 64 * cfg["pool_rows"]],
        "sequence_order": "width left-to-right; each step flattens channels then vertical bands",
        "lstm_layers": 1, "bidirectional": False, "sequence_aggregation": "mean of LSTM output steps",
        "augmentation": "none", "pretrained_weights": False,
        "loss": "Class-weighted cross-entropy; aggregate divided by sum of target weights",
        "checkpoint_selection": SELECTION_RULE, "best_selection_key": list(best_key),
        "best_epoch": best_epoch, "epochs_run": len(history), "stopped_early": len(history) < cfg["max_epochs"],
        "early_stopping": "Patience counts epochs without strict lexicographic validation improvement",
        "best_train_metrics": {k.removeprefix("train_"): v for k, v in best_row.items() if k.startswith("train_")},
        "best_validation_metrics": {k.removeprefix("val_"): v for k, v in best_row.items() if k.startswith("val_")},
        "epoch_metric_definition": "Full-partition evaluation after epoch with dropout disabled",
        "training_seconds": training_seconds, "device": "cpu",
        "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                        "torch": str(torch.__version__), "numpy": np.__version__,
                        "num_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "reproducibility_limit": "CPU determinism configured; identical results across software/hardware versions are not guaranteed",
        "checkpoint_path": str(checkpoint_path.resolve()), "history_path": str((out / "history.csv").resolve()),
    }
    _plot_history(history, best_epoch, out / "training_curve.png")
    metadata["runtime_seconds"] = time.perf_counter() - started
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return predictor, metadata
