"""Synthetic runner and append-only preservation checks; never touch project evidence."""
import contextlib
import csv
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import src.baseline_preservation as preservation
import src.run_cnn_lstm as runner


FIELDS = ["experiment_id", "date", "model", "status", "dataset_version", "split_version",
          "preprocessing", "random_seed", "hyperparameters", "accuracy", "balanced_accuracy",
          "macro_f1", "weighted_f1", "notes", "output_directory"]


def record(run_id):
    return {key: run_id if key == "experiment_id" else "synthetic" for key in FIELDS}


class IsolatedEvidenceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "experiments/experiment_log.csv"
        self.log.parent.mkdir()
        with self.log.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([record("lda_original"), record("cnn_original")])
        self.original = self.log.read_bytes()
        self.snapshot = self.root / "snapshot.csv"
        self.snapshot.write_bytes(self.original)
        self.protected = self.root / "original_model.bin"
        self.protected.write_bytes(b"unchanged synthetic model")
        self.lock = self.root / "lock.json"
        self.lock.write_text(json.dumps({"files": {
            "experiments/experiment_log.csv": hashlib.sha256(self.original).hexdigest(),
            "original_model.bin": hashlib.sha256(self.protected.read_bytes()).hexdigest(),
        }}))
        for attribute, value in [("ROOT", self.root), ("LOG", self.log),
                                 ("LOCK", self.lock), ("SNAPSHOT", self.snapshot)]:
            p = patch.object(preservation, attribute, value)
            p.start()
            self.addCleanup(p.stop)

    def read_rows(self):
        with self.log.open() as handle:
            return list(csv.DictReader(handle))


class PreservationTest(IsolatedEvidenceFixture):
    def test_append_preserves_exact_original_bytes_and_rows(self):
        preservation.append_experiment_log(record("new_run"))
        self.assertTrue(self.log.read_bytes().startswith(self.original))
        self.assertEqual([r["experiment_id"] for r in self.read_rows()],
                         ["lda_original", "cnn_original", "new_run"])
        check = preservation.verify_baseline_preservation()
        self.assertEqual(check["original_log_rows_preserved"], 2)
        self.assertEqual(check["appended_log_rows"], 1)

    def test_duplicate_id_and_wrong_schema_rejected_before_write(self):
        for new in [record("cnn_original"), {**record("new_run"), "unexpected": "field"}]:
            with self.subTest(record=new), self.assertRaises(ValueError):
                preservation.append_experiment_log(new)
            self.assertEqual(self.log.read_bytes(), self.original)

    def test_modified_model_snapshot_or_original_prefix_is_rejected(self):
        for path in [self.protected, self.snapshot, self.log]:
            old = path.read_bytes()
            path.write_bytes(b"changed" + old)
            with self.subTest(path=path), self.assertRaises(ValueError):
                preservation.verify_baseline_preservation()
            path.write_bytes(old)

    def test_malformed_or_duplicate_appended_rows_are_rejected(self):
        for extra in [b"only-one-field\r\n", self.original.splitlines(keepends=True)[1]]:
            self.log.write_bytes(self.original + extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                preservation.verify_baseline_preservation()
        self.log.write_bytes(self.original)


class RunnerTest(IsolatedEvidenceFixture):
    def setUp(self):
        super().setUp()
        self.config = {
            "model": "cnn_lstm", "seed": 42, "cnn_lstm": {"seed": 42},
            "dataset_version": "synthetic_data", "split_version": "synthetic_split",
            "split_sha256": "split_hash", "cache_sha256": "cache_hash",
            "class_names": ["Normal", "MI", "Abnormal", "History_MI"],
            "input_shape": [1, 128, 256], "selection_rule": "predeclared rule",
            "test_policy": "exploratory once after selection", "representation": "spatial image",
            "comparison_limit": "not isolated architecture ablation",
        }
        self.config_path = self.root / "synthetic_config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.x = np.stack([np.full((1, 128, 256), i / 20, np.float32) for i in range(12)])
        self.y = np.tile(np.arange(4), 3)
        self.parts = {"train": np.arange(4), "validation": np.arange(4, 8), "test": np.arange(8, 12)}
        self.rows = [{"sample_id": f"synthetic-{i}", "label": int(self.y[i]), "path": f"image-{i}"}
                     for i in range(12)]
        self.frozen = {"audit": {key: self.config[key] for key in
                                ["dataset_version", "split_version", "split_sha256", "cache_sha256"]},
                       "config": {"class_names": self.config["class_names"], "preprocessing": {}}}
        self.calls = []
        self.module = types.ModuleType("src.models.cnn_lstm")
        self.module.train_cnn_lstm = self.fake_train
        patches = [patch.object(runner, "ROOT", self.root),
                   patch.object(runner, "load_verified_data", return_value=(self.x, self.y, self.rows, self.parts, self.frozen)),
                   patch.object(runner, "source_hashes", return_value={"synthetic.py": "fixed"}),
                   patch.object(runner, "environment", return_value={"runtime": "synthetic"}),
                   patch.object(runner, "save_evaluation", side_effect=self.fake_evaluate),
                   patch.dict("sys.modules", {"src.models.cnn_lstm": self.module})]
        self.patches = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def fake_train(self, train_x, train_y, val_x, val_y, cfg, model_dir):
        np.testing.assert_array_equal(train_x, self.x[self.parts["train"]])
        np.testing.assert_array_equal(val_x, self.x[self.parts["validation"]])
        np.testing.assert_array_equal(train_y, self.y[self.parts["train"]])
        np.testing.assert_array_equal(val_y, self.y[self.parts["validation"]])
        self.calls.append("train")
        (model_dir / "best.pt").write_bytes(b"synthetic selected checkpoint")
        return types.SimpleNamespace(predict_proba=self.predict), {"best_epoch": 2}

    def predict(self, array):
        self.assertTrue((self.root / "results/cnn_lstm/attempt/selection_frozen.json").is_file())
        self.calls.append(("predict", tuple(np.round(array[:, 0, 0, 0], 3))))
        return np.full((len(array), 4), .25, dtype=np.float32)

    def fake_evaluate(self, out, subset, rows, labels, probabilities, class_names):
        self.calls.append(("evaluate", subset))
        np.testing.assert_array_equal(labels, self.y[self.parts[subset]])
        return {key: .25 for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(self.config_path, "attempt")

    def test_training_excludes_test_and_success_is_appended_once(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed")
        self.assertEqual([item for item in self.calls if isinstance(item, tuple) and item[0] == "evaluate"],
                         [("evaluate", "train"), ("evaluate", "validation"), ("evaluate", "test")])
        self.assertEqual(sum(isinstance(item, tuple) and item[0] == "predict" for item in self.calls), 3)
        self.assertEqual(self.read_rows()[-1]["status"], "completed")
        self.assertTrue(self.log.read_bytes().startswith(self.original))
        state = json.loads((self.root / "results/cnn_lstm/attempt/test_inference_state.json").read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["inference_calls"], 1)
        self.assertEqual(state["records"], len(self.parts["test"]))
        with self.assertRaises(ValueError):
            self.execute()
        self.assertEqual(len(self.read_rows()), 3)

    def test_training_failure_logged_without_any_inference(self):
        self.module.train_cnn_lstm = lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic training failure"))
        with self.assertRaisesRegex(RuntimeError, "synthetic training failure"):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.read_rows()[-1]["status"], "failed")
        self.assertTrue((self.root / "results/cnn_lstm/attempt/failure.json").is_file())
        self.assertEqual(json.loads((self.root / "results/cnn_lstm/attempt/run.json").read_text())["status"], "failed")
        self.assertFalse((self.root / "results/cnn_lstm/attempt/test_inference_state.json").exists())

    def test_source_mutation_refuses_all_inference(self):
        with patch.object(runner, "source_hashes", side_effect=[{"source": "old"}, {"source": "changed"}]):
            with self.assertRaisesRegex(ValueError, "changed during training"):
                self.execute()
        self.assertEqual(self.calls, ["train"])
        self.assertEqual(self.read_rows()[-1]["status"], "failed")

    def test_logging_error_does_not_mask_original_training_failure(self):
        self.module.train_cnn_lstm = lambda *args: (_ for _ in ()).throw(RuntimeError("original training failure"))
        with patch.object(runner, "append_experiment_log", side_effect=ValueError("logging unavailable")):
            with self.assertRaisesRegex(RuntimeError, "original training failure"):
                self.execute()
        self.assertTrue((self.root / "results/cnn_lstm/attempt/failure.json").is_file())
        logging_failure = json.loads((self.root / "results/cnn_lstm/attempt/logging_failure.json").read_text())
        self.assertIn("original training failure", logging_failure["original_error"])
        self.assertIn("logging unavailable", logging_failure["logging_error"])

    def test_failed_completion_append_does_not_leave_completed_run_claim(self):
        with patch.object(runner, "append_experiment_log", side_effect=ValueError("logging unavailable")):
            with self.assertRaises(ValueError):
                self.execute()
        completed = self.root / "results/cnn_lstm/attempt/run.json"
        self.assertTrue(not completed.exists() or json.loads(completed.read_text())["status"] != "completed")

    def test_frozen_split_mismatch_fails_before_training_and_test(self):
        self.config["split_sha256"] = "not_the_frozen_split"
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "Frozen comparison mismatch"):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.read_rows()[-1]["status"], "failed")
        self.assertFalse((self.root / "results/cnn_lstm/attempt/test_inference_state.json").exists())

    def test_invalid_development_probabilities_fail_before_test(self):
        self.predict = lambda array: np.full((len(array), 4), np.nan)
        with self.assertRaisesRegex(ValueError, "Invalid prediction probabilities"):
            self.execute()
        self.assertEqual(self.calls, ["train"])
        self.assertEqual(self.read_rows()[-1]["status"], "failed")
        self.assertFalse((self.root / "results/cnn_lstm/attempt/test_inference_state.json").exists())


if __name__ == "__main__":
    unittest.main()
