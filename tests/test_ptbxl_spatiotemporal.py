"""Synthetic-only controls for the image CNN/causal feature-convolution model."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.ptbxl_cnn_lstm import LeadImageCNNLSTM
from src.models.ptbxl_spatiotemporal import (
    CLASS_NAMES, INPUT_SHAPE, CausalResidualBlock, LeadImageSpatiotemporal,
    _get_config, load_checkpoint, selection_key, train_model,
)


class SyntheticDataset(Dataset):
    def __init__(self, images, labels, normalization, split):
        self.normalization = normalization
        self.split = split
        self.normalize = True
        self.metadata = {"class_names": list(CLASS_NAMES)}
        self.images = torch.from_numpy((images - normalization["mean"]) / normalization["std"])
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.accesses = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        self.accesses += 1
        return self.images[index], self.labels[index]


def datasets():
    rng = np.random.default_rng(19)
    images = rng.uniform(0, .25, size=(4, *INPUT_SHAPE)).astype(np.float32)
    validation = rng.uniform(.7, 1., size=(4, *INPUT_SHAPE)).astype(np.float32)
    labels = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 1, 1, 0, 1]], dtype=np.float32)
    norm = {"fit_partition": "train", "mean": float(images.mean(dtype=np.float64)),
            "std": float(images.std(dtype=np.float64)), "training_images": 4}
    return SyntheticDataset(images, labels, norm, "train"), SyntheticDataset(validation, labels, norm, "validation"), norm


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identical_seeded_cnn_front_end_and_finite_gradients(self):
        torch.manual_seed(42)
        reference = LeadImageCNNLSTM()
        torch.manual_seed(42)
        model = LeadImageSpatiotemporal()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17161)
        self.assertEqual(model.feature_receptive_field, 13)
        self.assertEqual(str(reference.features), str(model.features))
        for name, value in reference.features.state_dict().items():
            torch.testing.assert_close(model.features.state_dict()[name], value, rtol=0, atol=0)
        images = torch.rand(2, *INPUT_SHAPE)
        self.assertEqual(tuple(model.features(images).shape), (2, 24, 15, 70))
        self.assertEqual(tuple(model.extract_sequence(images).shape), (2, 70, 96))
        output = model(images)
        self.assertEqual(tuple(output.shape), (2, 5))
        torch.nn.functional.binary_cross_entropy_with_logits(output, torch.zeros(2, 5)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 12, 48, 1000))

    def test_causal_block_and_head_do_not_read_future_feature_columns(self):
        torch.manual_seed(9)
        block = CausalResidualBlock(dilation=2).eval()
        x = torch.randn(2, 32, 70)
        changed = x.clone()
        changed[:, :, 31:] += 50 * torch.randn_like(changed[:, :, 31:])
        with torch.inference_mode():
            before, after = block(x), block(changed)
        self.assertEqual(tuple(before.shape), tuple(x.shape))
        torch.testing.assert_close(before[:, :, :31], after[:, :, :31], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[:, :, 31:], after[:, :, 31:]))
        model = LeadImageSpatiotemporal().eval()
        sequence = torch.randn(2, 96, 70)
        altered = sequence.clone()
        altered[:, :, 31:] += 50 * torch.randn_like(altered[:, :, 31:])
        with torch.inference_mode():
            before = model.temporal_blocks(model.projection(sequence))
            after = model.temporal_blocks(model.projection(altered))
        torch.testing.assert_close(before[:, :, :31], after[:, :, :31], rtol=0, atol=0)
        # This intentionally tests the head, not the noncausal CNN/whole-image classifier.

    def test_head_receptive_field_is_thirteen_supplied_columns(self):
        torch.manual_seed(23)
        model = LeadImageSpatiotemporal().eval()
        sequence = torch.randn(1, 96, 70, requires_grad=True)
        output = model.temporal_blocks(model.projection(sequence))
        output[:, :, 30].sum().backward()
        influence = sequence.grad.abs().sum(dim=(0, 1))
        self.assertEqual(float(influence[:18].sum()), 0.)
        self.assertEqual(float(influence[31:].sum()), 0.)
        self.assertGreater(float(influence[18]), 0.)
        self.assertGreater(float(influence[30]), 0.)

    def test_fixed_architecture_config_and_selection_validation(self):
        for bad in ({"hidden_size": 32}, {"hidden_channels": 16}, {"kernel_size": 5}, {"dilations": [1, 4]},
                    {"dilations": [True, 2]}, {"device": "mps"}, {"num_workers": False}, {"dropout": True}):
            with self.assertRaises(ValueError):
                _get_config(bad)
        def key(auc, ap, bce):
            return selection_key({"macro_roc_auc": auc, "macro_average_precision": ap, "binary_cross_entropy": bce})
        self.assertGreater(key(.8, .1, 3.), key(.7, .9, .1))
        self.assertGreater(key(.8, .7, 3.), key(.8, .6, .1))
        self.assertGreater(key(.8, .7, .2), key(.8, .7, .3))
        self.assertFalse(key(.8, .7, .2) > key(.8, .7, .2))
        with self.assertRaises(ValueError):
            key(None, .7, .2)


class TrainingTests(unittest.TestCase):
    cfg = {"max_epochs": 1, "patience": 1, "batch_size": 2, "num_threads": 2}

    def test_determinism_checkpoint_identity_progress_and_no_overwrite(self):
        train, val, norm = datasets()
        log = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(log):
            first_dir, second_dir = Path(directory) / "first", Path(directory) / "second"
            first, metadata = train_model(train, val, self.cfg, first_dir, norm, [2] * 5)
            self.assertEqual(train.accesses, len(train))
            self.assertEqual(val.accesses, len(val))
            second, _ = train_model(train, val, self.cfg, second_dir, norm, [2] * 5)
            probabilities = first.predict_proba(val)
            np.testing.assert_array_equal(probabilities, second.predict_proba(val))
            restored = load_checkpoint(first_dir / "best.pt")
            np.testing.assert_array_equal(probabilities, restored.predict_proba(val, progress_label="synthetic_replay"))
            self.assertFalse(np.allclose(probabilities.sum(axis=1), 1.))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertEqual(metadata["model"], "LeadImageSpatiotemporal")
            self.assertEqual(metadata["normalization"], norm)
            self.assertEqual(metadata["class_counts_train"], [2] * 5)
            self.assertEqual(metadata["feature_receptive_field"], 13)
            self.assertNotIn("lstm_layers", metadata)
            progress = json.loads((first_dir / "progress.json").read_text())
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["validation_records_seen"], len(val))
            self.assertIn("+00:00", progress["updated_utc"])
            with self.assertRaises(FileExistsError):
                train_model(train, val, self.cfg, first_dir, norm, [2] * 5)
            checkpoint = torch.load(first_dir / "best.pt", weights_only=True)
            checkpoint["model_name"] = "LeadImageCNNLSTM"
            torch.save(checkpoint, Path(directory) / "wrong_model.pt")
            with self.assertRaises(ValueError):
                load_checkpoint(Path(directory) / "wrong_model.pt")
            checkpoint["model_name"] = "LeadImageSpatiotemporal"
            checkpoint["config"]["dilations"] = [1, 4]
            torch.save(checkpoint, Path(directory) / "wrong_config.pt")
            with self.assertRaises(ValueError):
                load_checkpoint(Path(directory) / "wrong_config.pt")
        text = log.getvalue()
        self.assertIn("validation batch 0/2", text)
        self.assertIn("validation batch 2/2 records=4/4", text)
        self.assertIn("prediction:validation batch 2/2 records=4/4", text)
        self.assertIn("prediction:synthetic_replay", text)
        self.assertRegex(text, r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00")

    def test_checkpoint_prefers_ap_over_bce_and_stops_on_exact_tie(self):
        train, val, norm = datasets()
        first = {"macro_roc_auc": .7, "macro_average_precision": .5, "binary_cross_entropy": .3}
        second = {"macro_roc_auc": .7, "macro_average_precision": .6, "binary_cross_entropy": .8}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch(
                "src.models.ptbxl_spatiotemporal._evaluate_validation", side_effect=[first, second, second]):
            _, metadata = train_model(train, val, {**self.cfg, "max_epochs": 5}, directory, norm, [2] * 5)
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


if __name__ == "__main__":
    unittest.main()
