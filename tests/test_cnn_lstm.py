"""Small synthetic tests only; no research train/validation/test records."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.models.cnn_lstm import (SmallCNNLSTM, load_checkpoint, selection_key,
                                  spatial_feature_sequence, train_cnn_lstm)


class SpatialSequenceTest(unittest.TestCase):
    def test_width_order_and_vertical_bands_are_preserved(self):
        feature_map = torch.zeros((2, 2, 2, 3))
        for batch in range(2):
            for channel in range(2):
                for row in range(2):
                    for column in range(3):
                        feature_map[batch, channel, row, column] = 1000 * batch + 100 * channel + 10 * row + column
        sequence = spatial_feature_sequence(feature_map)
        self.assertEqual(tuple(sequence.shape), (2, 3, 4))
        torch.testing.assert_close(sequence[0], torch.tensor([[0., 10., 100., 110.], [1., 11., 101., 111.], [2., 12., 102., 112.]]))
        torch.testing.assert_close(sequence[1], sequence[0] + 1000)
        model = SmallCNNLSTM().eval()
        with torch.inference_mode():
            self.assertEqual(tuple(model.features(torch.zeros(2, 1, 128, 256)).shape), (2, 64, 8, 16))
            self.assertEqual(tuple(model.extract_sequence(torch.zeros(2, 1, 128, 256)).shape), (2, 16, 128))
            self.assertEqual(tuple(model(torch.zeros(2, 1, 128, 256)).shape), (2, 4))
        self.assertFalse(model.lstm.bidirectional)
        self.assertEqual(model.lstm.num_layers, 1)

    def test_lexicographic_selection_priority(self):
        key = lambda f1, balanced, loss: selection_key({"macro_f1": f1, "balanced_accuracy": balanced, "loss": loss})
        self.assertGreater(key(.5, .1, 9.), key(.4, .9, .1))
        self.assertGreater(key(.5, .6, 9.), key(.5, .5, .1))
        self.assertGreater(key(.5, .6, .8), key(.5, .6, .9))
        self.assertFalse(key(.5, .6, .8) > key(.5, .6, .8))


class TrainingContractTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        self.x = rng.uniform(0, .2, size=(10, 1, 128, 256)).astype(np.float32)
        self.y = np.repeat(np.arange(4), [4, 3, 2, 1])
        self.val = rng.uniform(.7, 1, size=(4, 1, 128, 256)).astype(np.float32)
        self.val_y = np.arange(4)
        self.config = {"max_epochs": 2, "patience": 2, "batch_size": 5, "num_threads": 2}

    def test_determinism_train_only_normalization_checkpoint_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first_dir, second_dir = Path(directory) / "first", Path(directory) / "second"
            first, metadata = train_cnn_lstm(self.x, self.y, self.val, self.val_y, self.config, first_dir)
            second, _ = train_cnn_lstm(self.x, self.y, self.val, self.val_y, self.config, second_dir)
            p = first.predict_proba(self.val)
            self.assertEqual(p.shape, (4, 4))
            np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
            np.testing.assert_array_equal(p, second.predict_proba(self.val))
            restored = load_checkpoint(first_dir / "best.pt")
            np.testing.assert_array_equal(p, restored.predict_proba(self.val))
            self.assertEqual(metadata["normalization"]["mean"], float(self.x.mean(dtype=np.float64)))
            self.assertEqual(metadata["normalization"]["std"], float(self.x.std(dtype=np.float64)))
            np.testing.assert_allclose(metadata["class_weights"], 10 / (4 * np.array([4, 3, 2, 1])))
            self.assertEqual(metadata["spatial_sequence_shape"], [16, 128])
            self.assertIn("not calibrated time", metadata["sequence_interpretation"])
            with self.assertRaises(FileExistsError):
                train_cnn_lstm(self.x, self.y, self.val, self.val_y, self.config, first_dir)
            self.assertGreater((first_dir / "training_curve.png").stat().st_size, 1000)

    def test_checkpoint_prefers_f1_over_loss_and_retains_earliest_exact_tie(self):
        def metrics(f1, balanced, loss):
            return {"accuracy": .5, "macro_f1": f1, "balanced_accuracy": balanced, "weighted_f1": .5, "loss": loss}
        train = metrics(.3, .4, 1.)
        val1, val2 = metrics(.2, .3, .9), metrics(.4, .4, 1.3)
        scripted_metrics = [train, val1, train, val2, train, val2]
        config = {**self.config, "max_epochs": 5, "patience": 1}
        with tempfile.TemporaryDirectory() as directory, patch("src.models.cnn_lstm._evaluate", side_effect=scripted_metrics):
            _, metadata = train_cnn_lstm(self.x, self.y, self.val, self.val_y, config, directory)
            self.assertEqual(metadata["best_epoch"], 2)
            self.assertEqual(metadata["epochs_run"], 3)
            self.assertTrue(metadata["stopped_early"])
            checkpoint = torch.load(Path(directory) / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["best_epoch"], 2)
            self.assertEqual(checkpoint["validation_selection_key"], [.4, .4, -1.3])
            with (Path(directory) / "history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["checkpoint_improved"] for row in rows], ["True", "True", "False"])


if __name__ == "__main__":
    unittest.main()
