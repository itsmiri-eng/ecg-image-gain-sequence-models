"""Bounded synthetic-only CPU feasibility probe; never reads ECG datasets."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import psutil
import torch
from torch import nn


class SharedLeadCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, (5, 9), stride=(2, 4), padding=(2, 4), bias=False),
            nn.GroupNorm(2, 4), nn.ReLU(),
            nn.Conv2d(4, 8, (3, 5), stride=2, padding=(1, 2), bias=False),
            nn.GroupNorm(2, 8), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 125)),
        )
        self.lstm = nn.LSTM(384, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(32, 5))

    def forward(self, x):
        batch = x.shape[0]
        z = self.cnn(x.reshape(batch * 12, 1, 48, 1000))
        z = z.reshape(batch, 12, 8, 4, 125).permute(0, 4, 1, 2, 3).reshape(batch, 125, 384)
        z, _ = self.lstm(z)
        return self.head(z.mean(dim=1))


class LeadChannelSeparableCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 12, 5, stride=(2, 4), padding=2, groups=12, bias=False),
            nn.Conv2d(12, 8, 1, bias=False), nn.GroupNorm(4, 8), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=8, bias=False),
            nn.Conv2d(8, 16, 1, bias=False), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16, bias=False),
            nn.Conv2d(16, 24, 1, bias=False), nn.GroupNorm(4, 24), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 70)),
        )
        self.lstm = nn.LSTM(96, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 5))

    def forward(self, x):
        z = self.cnn(x).permute(0, 3, 1, 2).flatten(2)
        z, _ = self.lstm(z)
        return self.head(z.mean(dim=1))


def benchmark(name, constructor, image_shape, sequence_shape):
    torch.manual_seed(42)
    model = constructor().cpu().train()
    batch_size = 16
    x = torch.rand((batch_size, *image_shape), dtype=torch.float32)
    y = torch.randint(0, 2, (batch_size, 5)).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.BCEWithLogitsLoss()
    process = psutil.Process()
    rss_samples = [process.memory_info().rss]
    times = []
    losses = []
    for step in range(12):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_function(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        elapsed = time.perf_counter() - start
        rss_samples.append(process.memory_info().rss)
        if step >= 2:
            times.append(elapsed)
            losses.append(float(loss.detach()))
        print(f"{name} step={step + 1}/12 seconds={elapsed:.4f} loss={float(loss.detach()):.5f}", flush=True)
    mean_seconds = statistics.mean(times)
    result = {
        "name": name,
        "batch_size": batch_size,
        "input_shape": [batch_size, *image_shape],
        "lstm_sequence_shape": [batch_size, *sequence_shape],
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "warmup_steps": 2,
        "measured_steps": 10,
        "step_seconds": times,
        "mean_step_seconds": mean_seconds,
        "median_step_seconds": statistics.median(times),
        "minimum_step_seconds": min(times),
        "maximum_step_seconds": max(times),
        "throughput_examples_per_second": batch_size / mean_seconds,
        "synthetic_losses_finite": bool(np.isfinite(losses).all()),
        "synthetic_losses": losses,
        "estimated_train_epoch_seconds_for_17084_eligible": int(np.ceil(17084 / batch_size)) * mean_seconds,
        "estimated_train_epoch_seconds_for_17418_source": int(np.ceil(17418 / batch_size)) * mean_seconds,
        "estimated_20_train_epochs_hours_for_17084_eligible": int(np.ceil(17084 / batch_size)) * mean_seconds * 20 / 3600,
        "estimated_30_train_epochs_hours_for_17084_eligible": int(np.ceil(17084 / batch_size)) * mean_seconds * 30 / 3600,
        "process_rss_bytes_before_steps": rss_samples[0],
        "process_rss_bytes_max_sampled": max(rss_samples),
        "memory_available_bytes_after_steps": psutil.virtual_memory().available,
        "architecture": str(model),
    }
    del model, x, y, optimizer, loss, logits
    gc.collect()
    return result


def main():
    output = Path(__file__).with_name("compute_probe.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    hardware = {
        "platform": platform.platform(), "machine": platform.machine(),
        "os_cpu_count": os.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "memory_available_bytes_before_probe": psutil.virtual_memory().available,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    variants = [
        benchmark("shared_per_lead_cnn_lstm_48x1000", SharedLeadCNNLSTM, (12, 48, 1000), (125, 384)),
        benchmark("lead_channel_separable_cnn_lstm_120x1120", LeadChannelSeparableCNNLSTM, (12, 120, 1120), (70, 96)),
    ]
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Synthetic-only bounded CPU forward/backward feasibility benchmark; no ECG files accessed.",
        "real_training_data_accessed": False, "validation_data_accessed": False, "test_data_accessed": False,
        "device": "cpu", "seed": 42, "hardware": hardware, "variants": variants,
        "timing_includes": ["zero_grad", "forward", "BCEWithLogitsLoss", "backward", "gradient clipping norm 1.0", "AdamW optimizer step"],
        "timing_excludes": ["real image decoding", "crop/pooling preprocessing", "data loading", "validation", "checkpoint writing", "logging/plotting"],
        "limitations": [
            "Synthetic random tensors and fixed targets are reused; losses do not measure ECG performance.",
            "Only 10 measured steps after 2 warmups; long-run thermal throttling and competing load can change estimates.",
            "Epoch and multi-epoch numbers extrapolate training-step mean only and are not total wall-time guarantees.",
            "RSS is sampled after steps and excludes instantaneous peaks and child processes; successive variants share a process/allocator.",
            "Batch 32 was not probed because this 8 GiB machine has limited available memory.",
            "Shared-lead variant concatenates per-lead vertical CNN features; separable variant mixes lead channels after its first depthwise convolution.",
            "Rendered-image sequence positions are spatial positions with documented image geometry; they are not recovered calibrated waveform samples.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
