"""Synthetic smoke checks; no research test data are used here."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.cnn import load_checkpoint, train_cnn


class CNNContractTest(unittest.TestCase):
    def test_fitting_is_reproducible_and_checkpoint_is_self_contained(self):
        rng = np.random.default_rng(7)
        x_train = rng.uniform(0, 0.2, size=(10, 1, 128, 256)).astype(np.float32)
        y_train = np.repeat(np.arange(4), [4, 3, 2, 1])
        # Deliberately different validation distribution detects pooled scaling.
        x_val = rng.uniform(0.7, 1.0, size=(4, 1, 128, 256)).astype(np.float32)
        y_val = np.arange(4)
        config = {"max_epochs": 2, "patience": 2, "batch_size": 4, "device": "cpu", "num_threads": 2}
        with tempfile.TemporaryDirectory() as directory:
            first_dir = Path(directory) / "first"
            first, meta = train_cnn(x_train, y_train, x_val, y_val, config, first_dir)
            second, _ = train_cnn(x_train, y_train, x_val, y_val, config, Path(directory) / "second")
            predictions = first.predict_proba(x_val)
            self.assertEqual(predictions.shape, (4, 4))
            np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-6)
            np.testing.assert_array_equal(predictions, second.predict_proba(x_val))
            self.assertAlmostEqual(meta["normalization"]["mean"], float(x_train.mean(dtype=np.float64)))
            self.assertAlmostEqual(meta["normalization"]["std"], float(x_train.std(dtype=np.float64)))
            np.testing.assert_allclose(meta["class_weights"], 10 / (4 * np.array([4, 3, 2, 1])))
            restored = load_checkpoint(first_dir / "best.pt")
            np.testing.assert_array_equal(predictions, restored.predict_proba(x_val))
            with (first_dir / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            best_row = min(rows, key=lambda row: float(row["val_loss"]))
            self.assertEqual(meta["best_epoch"], int(best_row["epoch"]))
            checkpoint = torch.load(first_dir / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["best_epoch"], meta["best_epoch"])
            self.assertGreater((first_dir / "training_curve.png").stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                train_cnn(x_train, y_train, x_val, y_val, config, first_dir)

    def test_missing_training_class_fails_before_artifacts_are_written(self):
        x = np.zeros((4, 1, 128, 256), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "All four classes"):
                train_cnn(x, np.zeros(4, dtype=np.int64), x, np.arange(4), {}, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
