"""Synthetic checks for the protocol-v2 shared trainer and CNN-only baseline (no ECG data)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.ptbxl_cnn_pool import LeadImageCNNPool
from src.training.ptbxl_trainer import MODELS, get_config, load_model, predict_proba, train_model

NORM = {"fit_partition": "train", "mean": 0.0, "std": 1.0}
NAMES = ("NORM", "MI", "STTC", "CD", "HYP")


class Synthetic(Dataset):
    def __init__(self, split, n):
        rng = np.random.default_rng(0 if split == "train" else 1)
        self.split, self.normalization = split, NORM
        self.y = np.zeros((n, 5), dtype=np.float32)
        self.y[np.arange(n), np.arange(n) % 5] = 1
        self.x = rng.normal(size=(n, 12, 120, 1120)).astype(np.float32) * 0.1
        self.rows = [{f"label_{name}": bool(self.y[i, j]) for j, name in enumerate(NAMES)} for i in range(n)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


class TestV2(unittest.TestCase):
    def test_cnn_pool_shape_and_shared_encoder(self):
        model = LeadImageCNNPool()
        self.assertEqual(tuple(model(torch.zeros(2, 12, 120, 1120)).shape), (2, 5))
        reference = MODELS["cnn_lstm"]()
        self.assertEqual([tuple(p.shape) for p in model.features.parameters()],
                         [tuple(p.shape) for p in reference.features.parameters()])

    def test_all_models_build(self):
        for name in MODELS:
            from src.training.ptbxl_trainer import build_model
            self.assertEqual(tuple(build_model(get_config({"model": name}))(torch.zeros(1, 12, 120, 1120)).shape), (1, 5))

    def test_train_reload_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"model": "cnn_pool", "max_epochs": 2, "batch_size": 5}
            train, val = Synthetic("train", 10), Synthetic("validation", 10)
            model, meta = train_model(train, val, cfg, Path(tmp) / "m", NORM)
            self.assertFalse(meta["test_data_accessed"])
            reloaded, _ = load_model(Path(tmp) / "m" / "best.pt")
            np.testing.assert_allclose(predict_proba(model, val, 5), predict_proba(reloaded, val, 5), atol=1e-6)
            with self.assertRaises(FileExistsError):
                train_model(train, val, cfg, Path(tmp) / "m", NORM)

    def test_rejects_non_training_normalization(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            train_model(Synthetic("train", 4), Synthetic("validation", 4), {"model": "cnn_pool"}, tmp,
                        {"fit_partition": "validation", "mean": 0, "std": 1})


if __name__ == "__main__":
    unittest.main()
