"""Signal-domain reference: model shape, shared-loop registration, and cache cohort/order/normalization."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.ptbxl_resnet1d import ResNet1dWang
from src.run_ptbxl_signal import signal_trainer
from src.ptbxl_signals import CACHE, SignalDataset, image_rows
from src.training import ptbxl_trainer as trainer


class _Toy(torch.utils.data.Dataset):
    def __init__(self, split, n, normalization):
        rng = np.random.default_rng(0 if split == "train" else 1)
        self.split, self.normalization = split, normalization
        self.x = rng.normal(size=(n, 12, 1000)).astype(np.float32)
        self.y = np.tile(np.eye(5, dtype=np.float32), (n // 5 + 1, 1))[:n]
        self.x[:, 0, :] += 3 * self.y[:, :1]  # learnable signal in lead I
        self.rows = [{f"label_{c}": int(v) for c, v in zip(trainer.CLASS_NAMES, y)} for y in self.y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])


def test_model_outputs_five_logits_per_ecg():
    model = ResNet1dWang().eval()
    assert model(torch.zeros(2, 12, 1000)).shape == (2, 5)
    assert 400_000 < sum(p.numel() for p in model.parameters()) < 600_000


def test_shared_loop_trains_the_registered_signal_model():
    norm = {"fit_partition": "train", "mean": [0.0] * 12, "std": [1.0] * 12}
    cfg = {"model": "resnet1d_wang", "max_epochs": 1, "batch_size": 4, "seed": 0}
    with tempfile.TemporaryDirectory() as directory, signal_trainer():
        model, metadata = trainer.train_model(_Toy("train", 20, norm), _Toy("validation", 10, norm), cfg,
                                              Path(directory) / "m", norm)
        restored, _ = trainer.load_model(Path(directory) / "m" / "best.pt")
    assert metadata["model_class"] == "ResNet1dWang" and not metadata["test_data_accessed"]
    assert sum(p.numel() for p in restored.parameters()) == metadata["num_parameters"]
    # The image networks' loop is left exactly as it was.
    assert trainer.INPUT_SHAPE == (12, 120, 1120) and "resnet1d_wang" not in trainer.MODELS


@pytest.mark.skipif(not (CACHE / "metadata.json").exists(), reason="signal cache not built")
def test_cache_matches_image_cohort_order_and_train_only_normalization():
    validation = SignalDataset("validation")
    assert [r["ecg_id"] for r in validation.rows] == [r["ecg_id"] for r in image_rows("validation")]
    assert len(validation) == 2146 and validation[0][0].shape == (12, 1000)
    train = np.load(CACHE / "train.npy", mmap_mode="r")
    assert np.allclose(train[:, 0, :].mean(dtype=np.float64), validation.normalization["mean"][0], atol=1e-6)
