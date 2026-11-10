"""Isolated PTB-XL orchestration tests: no real images, training, or logs."""
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from torch.utils.data import Subset

from src import baseline_preservation as preservation
from src import run_ptbxl_spatiotemporal as runner

FIELDS = ["experiment_id", "date", "model", "status", "dataset_version", "split_version",
          "preprocessing", "random_seed", "hyperparameters", "accuracy", "balanced_accuracy",
          "macro_f1", "weighted_f1", "notes", "output_directory"]


class PTBXLSpatiotemporalRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "experiments/experiment_log.csv"
        self.log.parent.mkdir()
        with self.log.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([{key: name if key == "experiment_id" else "old" for key in FIELDS}
                              for name in ("lda", "cnn")])
        snapshot = self.root / "baseline_log.csv"
        snapshot.write_bytes(self.log.read_bytes())
        self.protected = self.root / "old_model.bin"
        self.protected.write_bytes(b"synthetic prior model")
        lock = self.root / "lock.json"
        self.write_json(lock, {"files": {"experiments/experiment_log.csv": self.sha(snapshot),
                                         "old_model.bin": self.sha(self.protected)}})
        with self.log.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(
                {key: "old_cnn_lstm" if key == "experiment_id" else "old" for key in FIELDS})
        with self.log.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=FIELDS).writerow(
                {key: "old_ptbxl_cnn_lstm" if key == "experiment_id" else "old" for key in FIELDS})
        self.original_log = self.log.read_bytes()
        prior = self.root / "experiments/protocols/ptbxl_spatiotemporal_image_v1/prior_artifacts.json"
        prior.parent.mkdir(parents=True)
        self.write_json(prior, {"sha256": {"old_model.bin": self.sha(self.protected),
                                           "experiments/experiment_log.csv": self.sha(self.log)}})
        self.source = self.root / "data/derived/ptbxl_images_v1"
        for name in ("protocol.json", "generation_state.json", "generation_summary.json", "verification.json",
                     "all_images.csv", "classification_manifest.csv", "classification_exclusions.csv",
                     "splits/train.csv", "splits/validation.csv", "splits/test.csv"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic source " + name)
        self.cache = self.root / "synthetic_cache"
        self.cache.mkdir()
        for name in ("protocol.json", "metadata.json", "cache_manifest.csv"):
            (self.cache / name).write_text("synthetic cache " + name)
        for name in runner.CODE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic frozen source " + name)
        self.config = {
            "model": "ptbxl_image_spatiotemporal", "class_names": list(runner.CLASS_NAMES),
            "input_shape": list(runner.INPUT_SHAPE), "seed": 42, "training": {"seed": 42},
            "prior_artifacts_sha256": self.sha(prior),
            "threshold": 0.5, "source_dataset_root": str(self.source.relative_to(self.root)),
            "cache_root": "synthetic_cache", "dataset_id": "synthetic_ptbxl", "split_version": "official",
            "preprocessing": {"image_only": True}, "test_policy": "once after freeze",
            "selection_rule": "validation macro AUROC", "sequence_interpretation": "image columns",
            "comparison_limit": "different task", "uncertainty": {"resamples": 5, "seed": 1},
            "source_manifest_sha256": self.sha(self.source / "classification_manifest.csv"),
            "source_verification_sha256": self.sha(self.source / "verification.json"),
            "source_protocol_sha256": self.sha(self.source / "protocol.json"),
        }
        self.config_path = self.root / "config.json"
        self.write_json(self.config_path, self.config)
        self.out = self.root / "results/ptbxl_spatiotemporal/attempt"
        self.calls = []
        self.replay_error = False
        self.validation_error = False
        self.fail_test = False
        self.normalization = {"fit_partition": "train", "mean": 0.1, "std": 0.2}
        for attribute, value in (("ROOT", self.root), ("LOG", self.log), ("LOCK", lock), ("SNAPSHOT", snapshot)):
            self.start_patch(patch.object(preservation, attribute, value))
        self.start_patch(patch.object(runner, "ROOT", self.root))
        self.start_patch(patch.object(runner, "environment", return_value={"synthetic": True}))
        self.start_patch(patch.object(runner, "CachedECGDataset", side_effect=self.dataset))
        self.train_mock = self.start_patch(patch.object(runner, "train_model", side_effect=self.train))
        self.start_patch(patch.object(runner, "load_checkpoint", return_value=types.SimpleNamespace(predict_proba=self.predict)))
        self.start_patch(patch.object(runner, "save_multilabel_evaluation", side_effect=self.evaluate))
        self.start_patch(patch.object(runner, "patient_cluster_bootstrap", return_value={"synthetic": True}))

    def start_patch(self, value):
        result = value.start()
        self.addCleanup(value.stop)
        return result

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value))

    def dataset(self, cache, split):
        self.assertEqual(cache, self.cache)
        self.calls.append("open_" + split)
        if split == "test":
            self.assertTrue((self.out / "checkpoint_replay.json").is_file())
            self.assertIn("predict_replay", self.calls)
        rows = [{"ecg_id": f"{split}-{i}", "patient_id": f"patient-{split}-{i}", "split": split,
                 **{f"label_{name}": bool((i + j) % 2) for j, name in enumerate(runner.CLASS_NAMES)}}
                for i in range(4)]
        return FakeDataset(rows, split, self.normalization)

    def train(self, train, validation, config, directory, normalization, counts):
        self.assertEqual((train.split, validation.split), ("train", "validation"))
        self.assertNotIn("open_test", self.calls)
        self.assertEqual(normalization, self.normalization)
        self.assertEqual(counts, [2] * 5)
        self.calls.append("train")
        (directory / "best.pt").write_bytes(b"selected checkpoint")
        p = np.tile(np.array([0.8, 0.7, 0.2, 0.1, 0.6], np.float32), (len(validation), 1))
        selected_metrics = runner.evaluate_multilabel(runner._labels(validation), p, runner.CLASS_NAMES)
        if self.validation_error:
            selected_metrics["macro_roc_auc"] += 0.1
        return types.SimpleNamespace(predict_proba=self.predict), {
            "best_epoch": 2, "test_data_accessed": False, "best_validation_metrics": selected_metrics}

    def predict(self, dataset):
        kind = "replay" if isinstance(dataset, Subset) else dataset.split
        self.assertTrue((self.out / "selection_frozen.json").is_file())
        self.calls.append("predict_" + kind)
        if kind == "test":
            state = json.loads((self.out / "test_inference_state.json").read_text())
            self.assertEqual(state["status"], "started")
            if self.fail_test:
                raise RuntimeError("synthetic test inference failure")
        p = np.tile(np.array([0.8, 0.7, 0.2, 0.1, 0.6], np.float32), (len(dataset), 1))
        if kind == "replay" and self.replay_error:
            p += 0.01
        return p

    def evaluate(self, out, split, rows, truth, probability, class_names):
        self.calls.append("evaluate_" + split)
        self.assertTrue(np.allclose(probability.sum(axis=1), 2.4))  # No softmax/row normalization.
        return runner.evaluate_multilabel(truth, probability, class_names)

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.config_path, "attempt")

    def log_rows(self):
        with self.log.open() as handle:
            return list(csv.DictReader(handle))

    def test_success_replays_training_before_one_test_and_preserves_log_prefix(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed")
        self.assertLess(self.calls.index("predict_replay"), self.calls.index("open_test"))
        self.assertEqual(self.calls.count("predict_test"), 1)
        self.assertTrue(self.log.read_bytes().startswith(self.original_log))
        record = self.log_rows()[-1]
        self.assertEqual(record["status"], "completed")
        self.assertEqual((record["accuracy"], record["balanced_accuracy"]), ("", ""))
        state = json.loads((self.out / "test_inference_state.json").read_text())
        self.assertEqual((state["status"], state["inference_calls"]), ("completed", 1))
        self.assertEqual(self.protected.read_bytes(), b"synthetic prior model")

    def test_checkpoint_replay_mismatch_prevents_opening_test(self):
        self.replay_error = True
        with self.assertRaisesRegex(ValueError, "Checkpoint reconstruction differs"):
            self.execute()
        self.assertNotIn("open_test", self.calls)
        self.assertFalse((self.out / "test_inference_state.json").exists())
        self.assertEqual(self.log_rows()[-1]["status"], "failed")
        self.assertEqual(json.loads((self.out / "run.json").read_text())["status"], "failed")

    def test_selected_validation_metrics_mismatch_prevents_test_and_replay(self):
        self.validation_error = True
        with self.assertRaisesRegex(ValueError, "Restored validation result differs"):
            self.execute()
        self.assertNotIn("open_test", self.calls)
        self.assertNotIn("predict_replay", self.calls)
        self.assertEqual(self.log_rows()[-1]["status"], "failed")

    def test_failed_test_keeps_attempt_marker_and_cannot_be_retried_under_same_id(self):
        self.fail_test = True
        with self.assertRaisesRegex(RuntimeError, "synthetic test inference failure"):
            self.execute()
        self.assertEqual(json.loads((self.out / "test_inference_state.json").read_text())["status"], "started")
        original_files = {str(path): path.read_bytes() for path in self.out.rglob("*") if path.is_file()}
        with self.assertRaisesRegex(ValueError, "already logged"):
            self.execute()
        self.assertEqual(self.calls.count("predict_test"), 1)
        self.assertEqual(original_files, {str(path): path.read_bytes() for path in self.out.rglob("*") if path.is_file()})

    def test_original_training_error_survives_failure_logging_error(self):
        self.train_mock.side_effect = RuntimeError("synthetic training failure")
        with patch.object(runner, "append_experiment_log", side_effect=OSError("synthetic log failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic training failure"):
                self.execute()
        self.assertEqual(self.log.read_bytes(), self.original_log)
        self.assertTrue((self.out / "failure.json").is_file())
        self.assertTrue((self.out / "logging_failure.json").is_file())
        self.assertNotIn("open_test", self.calls)

    def test_freeze_guard_detects_each_mutable_input(self):
        checkpoint = self.root / "selected.pt"
        checkpoint.write_bytes(b"selected")
        config_hash = self.sha(self.config_path)
        sources = runner._source_hashes()
        cache_hashes = {name: self.sha(self.cache / name) for name in ("protocol.json", "metadata.json", "cache_manifest.csv")}
        checkpoint_hash = self.sha(checkpoint)
        paths = [self.config_path, self.root / runner.CODE_FILES[0], self.cache / "metadata.json", checkpoint]
        for path in paths:
            old = path.read_bytes()
            path.write_bytes(old + b"changed")
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner._check_frozen(self.config_path, config_hash, sources, self.cache, cache_hashes,
                                     checkpoint, checkpoint_hash)
            path.write_bytes(old)

    def test_completed_log_write_then_check_failure_is_reported_as_already_appended(self):
        original_append = preservation.append_experiment_log
        first = True
        def append_then_fail(record):
            nonlocal first
            original_append(record)
            if first:
                first = False
                raise RuntimeError("post-append preservation failure")
        with patch.object(runner, "append_experiment_log", side_effect=append_then_fail):
            with self.assertRaisesRegex(RuntimeError, "post-append preservation failure"):
                self.execute()
        self.assertEqual(len(self.log_rows()), 5)
        self.assertEqual(self.log_rows()[-1]["status"], "completed")
        failure = json.loads((self.out / "failure.json").read_text())
        self.assertTrue(failure["central_log_already_appended"])


class FakeDataset:
    def __init__(self, rows, split, normalization):
        self.rows, self.split, self.normalization = rows, split, normalization
        self.normalize = True

    def __len__(self):
        return len(self.rows)


if __name__ == "__main__":
    unittest.main()
