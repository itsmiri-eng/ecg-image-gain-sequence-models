"""Focused synthetic checks; no PTB-XL images, source signals or splits read."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.ptbxl_cnn_lstm import (
    CLASS_NAMES, INPUT_SHAPE, LeadImageCNNLSTM, _evaluate_validation,
    load_checkpoint, selection_key, spatial_feature_sequence, train_model,
)


class SyntheticDataset(Dataset):
    def __init__(self, images, targets, normalization, split):
        self.normalization = normalization
        self.split = split
        self.normalize = True
        self.metadata = {"class_names": list(CLASS_NAMES)}
        self.images = torch.from_numpy((images - normalization["mean"]) / normalization["std"])
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.accesses = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        self.accesses += 1
        return self.images[index], self.targets[index]


def toy_datasets():
    rng = np.random.default_rng(19)
    images = rng.uniform(0, .25, size=(4, *INPUT_SHAPE)).astype(np.float32)
    validation = rng.uniform(.7, 1., size=(4, *INPUT_SHAPE)).astype(np.float32)
    targets = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=np.float32)
    normalization = {"fit_partition": "train", "mean": float(images.mean(dtype=np.float64)),
                     "std": float(images.std(dtype=np.float64)), "training_images": 4}
    return (SyntheticDataset(images, targets, normalization, "train"),
            SyntheticDataset(validation, targets, normalization, "validation"), normalization)


class ModelShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_column_order_lead_processing_shape_and_finite_gradients(self):
        feature_map = torch.tensor([[[[0., 1., 2.], [10., 11., 12.]],
                                     [[100., 101., 102.], [110., 111., 112.]]]])
        torch.testing.assert_close(spatial_feature_sequence(feature_map),
                                   torch.tensor([[[0., 10., 100., 110.], [1., 11., 101., 111.], [2., 12., 102., 112.]]]))
        torch.manual_seed(42)
        model = LeadImageCNNLSTM()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 18025)
        self.assertEqual(model.features[0].groups, 12)
        self.assertEqual(model.features[1].in_channels, 12)
        self.assertEqual(model.features[1].out_channels, 8)
        x = torch.rand((2, *INPUT_SHAPE))
        self.assertEqual(tuple(model.features(x).shape), (2, 24, 15, 70))
        self.assertEqual(tuple(model.extract_sequence(x).shape), (2, 70, 96))
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 5))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros(2, 5)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(model.lstm.num_layers, 1)
        self.assertFalse(model.lstm.bidirectional)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 12, 48, 1000))

    def test_selection_priority_ties_and_undefined_metrics(self):
        def key(auc, ap, bce):
            return selection_key({"macro_roc_auc": auc, "macro_average_precision": ap, "binary_cross_entropy": bce})
        self.assertGreater(key(.7, .2, 2.), key(.6, .9, .1))
        self.assertGreater(key(.7, .8, 2.), key(.7, .7, .1))
        self.assertGreater(key(.7, .8, .2), key(.7, .8, .3))
        self.assertFalse(key(.7, .8, .2) > key(.7, .8, .2))
        for value in (None, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                key(value, .8, .2)

    def test_full_validation_metrics_do_not_depend_on_batch_boundaries(self):
        _, val, _ = toy_datasets()
        torch.manual_seed(9)
        model = LeadImageCNNLSTM().eval()
        one = _evaluate_validation(model, DataLoader(val, batch_size=4, shuffle=False))
        uneven = _evaluate_validation(model, DataLoader(val, batch_size=3, shuffle=False))
        self.assertEqual(one["n_samples"], 4)
        for key in ("macro_roc_auc", "macro_average_precision", "macro_f1", "binary_cross_entropy"):
            self.assertAlmostEqual(one[key], uneven[key], places=6)


class TrainingContractTest(unittest.TestCase):
    config = {"max_epochs": 1, "patience": 1, "batch_size": 2, "num_threads": 2}

    def test_reproducibility_checkpoint_independent_probabilities_and_no_overwrite(self):
        train, val, norm = toy_datasets()
        with tempfile.TemporaryDirectory() as directory:
            first_dir, second_dir = Path(directory) / "first", Path(directory) / "second"
            predictor, metadata = train_model(train, val, self.config, first_dir, norm, [2] * 5)
            self.assertEqual(train.accesses, len(train))  # No full training evaluation pass.
            self.assertEqual(val.accesses, len(val))
            second, _ = train_model(train, val, self.config, second_dir, norm, dict(zip(CLASS_NAMES, [2] * 5)))
            probabilities = predictor.predict_proba(val)
            np.testing.assert_array_equal(probabilities, second.predict_proba(val))
            restored = load_checkpoint(first_dir / "best.pt")
            np.testing.assert_array_equal(probabilities, restored.predict_proba(val))
            self.assertEqual(probabilities.shape, (4, 5))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
            self.assertFalse(np.allclose(probabilities.sum(axis=1), 1.))
            # Reversing a dataset reverses prediction rows; no shuffle at inference. Batch order may change
            # float32 rounding on some CPU backends (seen on Windows: 6e-8), so use the runners' 1e-6 replay limit.
            reverse = torch.utils.data.Subset(val, list(reversed(range(len(val)))))
            np.testing.assert_allclose(probabilities[::-1], restored.predict_proba(reverse), rtol=0, atol=1e-6)
            self.assertEqual(metadata["normalization"], norm)
            self.assertEqual(metadata["class_counts_train"], [2] * 5)
            self.assertFalse(metadata["test_data_accessed"])
            self.assertEqual(metadata["num_parameters"], 18025)
            self.assertEqual(json.loads((first_dir / "progress.json").read_text())["status"], "complete")
            with self.assertRaises(FileExistsError):
                train_model(train, val, self.config, first_dir, norm, [2] * 5)
            self.assertGreater((first_dir / "training_curve.png").stat().st_size, 1000)
            val.normalization = {**norm, "mean": norm["mean"] + .1}
            with self.assertRaises(ValueError):
                restored.predict_proba(val)
            bad = torch.load(first_dir / "best.pt", weights_only=True)
            bad["input_shape"] = [12, 48, 1000]
            torch.save(bad, Path(directory) / "wrong_shape.pt")
            with self.assertRaises(ValueError):
                load_checkpoint(Path(directory) / "wrong_shape.pt")

    def test_selection_prefers_ap_over_bce_and_retains_earliest_exact_tie(self):
        train, val, norm = toy_datasets()
        first = {"macro_roc_auc": .7, "macro_average_precision": .5, "binary_cross_entropy": .3}
        second = {"macro_roc_auc": .7, "macro_average_precision": .6, "binary_cross_entropy": .8}
        with tempfile.TemporaryDirectory() as directory, patch(
                "src.models.ptbxl_cnn_lstm._evaluate_validation", side_effect=[first, second, second]):
            _, metadata = train_model(train, val, {**self.config, "max_epochs": 5}, directory, norm, [2] * 5)
            self.assertEqual(metadata["best_epoch"], 2)
            self.assertEqual(metadata["epochs_run"], 3)
            self.assertTrue(metadata["stopped_early"])
            checkpoint = torch.load(Path(directory) / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["best_epoch"], 2)
            self.assertEqual(checkpoint["validation_selection_key"], [.7, .6, -.8])
            with (Path(directory) / "history.csv").open() as handle:
                history = list(csv.DictReader(handle))
            self.assertEqual([row["checkpoint_improved"] for row in history], ["True", "True", "False"])
            self.assertGreater(float(history[0]["learning_rate"]), float(history[1]["learning_rate"]))
            self.assertGreater(float(history[1]["learning_rate"]), float(history[2]["learning_rate"]))

    def test_wrong_partition_and_observed_supports_fail(self):
        train, val, norm = toy_datasets()
        with tempfile.TemporaryDirectory() as directory:
            val.split = "test"
            with self.assertRaises(ValueError):
                train_model(train, val, self.config, Path(directory) / "wrong_split", norm, [2] * 5)
            self.assertFalse((Path(directory) / "wrong_split").exists())
            val.split = "validation"
            with self.assertRaisesRegex(ValueError, "Observed training"):
                train_model(train, val, self.config, Path(directory) / "wrong_counts", norm, [1] * 5)
            self.assertEqual(json.loads((Path(directory) / "wrong_counts/progress.json").read_text())["status"], "failed")
            self.assertFalse((Path(directory) / "wrong_counts/best.pt").exists())


if __name__ == "__main__":
    unittest.main()
