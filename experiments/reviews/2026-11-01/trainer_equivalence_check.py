"""Check that the v2 shared trainer reproduces the v1 hybrid-ViT trainer exactly.

Both trainers fit one epoch on the same 64 training ECGs (validated on 32
validation ECGs) with seed 42; final weights must match. No test data is used.
Run: python experiments/reviews/2026-11-01/trainer_equivalence_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import Dataset

from src.ptbxl_learning_data import CachedECGDataset, CLASS_NAMES
from src.training import ptbxl_hybrid_vit as v1
from src.training import ptbxl_trainer as v2


class Head(Dataset):
    def __init__(self, base, n):
        self.base, self.n, self.split, self.normalization = base, n, base.split, base.normalization
        self.normalize, self.rows = True, base.rows[:n]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.base[i]


cache = ROOT / "data/processed/ptbxl_lead_images_v1"
train, val = Head(CachedECGDataset(cache, "train"), 64), Head(CachedECGDataset(cache, "validation"), 32)
norm = {k: v for k, v in train.normalization.items() if k != "training_images"}  # count check only
train.normalization = val.normalization = norm
counts = np.asarray([[r[f"label_{n}"] for n in CLASS_NAMES] for r in train.rows]).sum(0).tolist()
from src.models import ptbxl_cnn_lstm as v1_lstm, ptbxl_spatiotemporal as v1_tcn

V1 = {"hybrid_vit": (v1.train_model, {"device": "cpu"}), "cnn_lstm": (v1_lstm.train_model, {}),
      "temporal_conv": (v1_tcn.train_model, {})}
results = []
for model, (v1_train, extra) in V1.items():
    with tempfile.TemporaryDirectory() as tmp:
        common = {"max_epochs": 1, "seed": 42}
        v1_train(train, val, {**common, **extra}, Path(tmp) / "v1", norm, counts)
        v2.train_model(train, val, {**common, "model": model}, Path(tmp) / "v2", norm)
        a = torch.load(Path(tmp) / "v1/best.pt", weights_only=True)["state_dict"]
        b = torch.load(Path(tmp) / "v2/best.pt", weights_only=True)["state_dict"]
        difference = max(float((a[k] - b[k]).abs().max()) for k in a)
    results.append({"model": model, "max_absolute_weight_difference": difference, "identical": difference == 0.0})
result = {"purpose": "v1 model-specific trainers versus v2 shared trainer, seed 42, CPU",
          "training_records": 64, "validation_records": 32, "epochs": 1, "results": results}
(Path(__file__).with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
