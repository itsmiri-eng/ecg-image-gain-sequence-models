"""Isolated saved-artifact report tests; no ECG files or model inference."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.evaluation.multilabel import evaluate_multilabel
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.report_ptbxl_spatiotemporal import CLASS_NAMES, PARTITIONS, _runtime_summary, _runtime_text, generate_report


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SpatiotemporalReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic_spatiotemporal_report_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run_id = "synthetic_fixture_only"
        self.out = self.root / "results/ptbxl_spatiotemporal" / self.run_id
        self.model = self.root / "models/ptbxl_spatiotemporal" / self.run_id
        self.out.mkdir(parents=True)
        self.model.mkdir(parents=True)
        (self.root / "docs").mkdir()
        (self.root / "docs/ptbxl_image_dataset.md").write_text(
            "SYNTHETIC TEST FIXTURE; source citation links only\n"
            "https://physionet.org/content/ptb-xl/1.0.3/\nhttps://doi.org/10.13026/kfzx-aw45\n"
            "https://doi.org/10.1038/s41597-020-0495-6\n")
        training_config = {"seed": 42, "num_threads": 4, "batch_size": 16, "num_workers": 0,
                           "max_epochs": 20, "patience": 4, "learning_rate": .001, "weight_decay": .0001,
                           "dropout": .2, "hidden_channels": 32, "kernel_size": 3, "dilations": [1, 2],
                           "gradient_clip_norm": 1., "device": "cpu", "scheduler": "cosine", "eta_min": .00001}
        self.config = {"model": "ptbxl_image_spatiotemporal", "class_names": list(CLASS_NAMES),
                       "input_shape": [12, 120, 1120], "threshold": .5, "training": training_config,
                       "uncertainty": {"resamples": 10, "seed": 2024, "confidence_level": .95}}
        rng = np.random.default_rng(23)
        self.metrics = {}
        truth = np.tile([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 0], [1, 0, 1, 1, 0]], (3, 1))
        for partition_index, split in enumerate(PARTITIONS, start=1):
            p = np.clip(.25 + .4 * truth + rng.normal(0, .25, truth.shape), .01, .99)
            self.metrics[split] = evaluate_multilabel(truth, p, CLASS_NAMES)
            rows = []
            for index, (labels, probabilities) in enumerate(zip(truth, p)):
                row = {"ecg_id": str(1000 * partition_index + index + 1),
                       "patient_id": str(1000 * partition_index + index // 4 + 100), "image_path": "not_loaded.png"}
                for name, label, probability in zip(CLASS_NAMES, labels, probabilities):
                    row.update({f"truth_{name}": int(label), f"probability_{name}": float(probability),
                                f"prediction_{name}": int(probability >= .5)})
                rows.append(row)
            with (self.out / f"{split}_predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(self.out / f"{split}_metrics.json", self.metrics[split])
            if split == "test":
                ci = patient_cluster_bootstrap(truth, p, [row["patient_id"] for row in rows], CLASS_NAMES, n_resamples=10, seed=2024)
        self.training = {"model": "LeadImageSpatiotemporal", "num_parameters": 17161, "config": training_config,
                         "normalization": {"fit_partition": "train", "mean": .02, "std": .08},
                         "best_epoch": 1, "epochs_run": 5, "runtime_seconds": 12.3, "stopped_early": True,
                         "sequence_shape": [70, 96], "temporal_blocks": 2, "convolutions_per_block": 2,
                         "kernel_size": 3, "dilations": [1, 2], "hidden_channels": 32, "feature_receptive_field": 13}
        digest = "a" * 64
        run = {"status": "completed", "experiment_id": self.run_id, "model_directory": str(self.model.relative_to(self.root)),
               "threshold": .5, "test_inference_calls": 1, "checkpoint_sha256": digest, "metrics": self.metrics,
               "training": self.training, "total_runtime_seconds": 15.2,
               "started_utc": "2026-09-25T00:00:00+00:00", "completed_utc": "2026-09-25T00:16:02+00:00"}
        files = {"run.json": run, "config.json": self.config, "training_metadata.json": self.training,
                 "provenance.json": {"configuration_sha256": digest, "synthetic_fixture_only": True},
                 "test_patient_bootstrap.json": ci,
                 "selection_frozen.json": {"checkpoint_sha256": digest, "configuration_sha256": digest},
                 "checkpoint_replay.json": {"passed": True, "split": "train", "maximum_absolute_probability_difference": 0,
                                             "checkpoint_sha256": digest},
                 "test_inference_state.json": {"status": "completed", "inference_calls": 1, "checkpoint_sha256": digest},
                 "preservation.json": {"changed_artifacts": []}}
        for name, value in files.items():
            write_json(self.out / name, value)
        write_json(self.model / "config.json", training_config)
        (self.model / "architecture.txt").write_text("Synthetic metadata fixture; no model constructed\n")
        validation = self.metrics["validation"]
        history = [{"epoch": epoch, "val_macro_roc_auc": validation["macro_roc_auc"] - .001 * (epoch - 1),
                    "val_macro_average_precision": validation["macro_average_precision"],
                    "val_binary_cross_entropy": validation["binary_cross_entropy"]} for epoch in range(1, 6)]
        with (self.model / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        write_json(self.out / "verification.json", {"status": "passed", "run_id": self.run_id,
                   "checkpoint_sha256": digest, "configuration_sha256": digest, "synthetic_fixture_only": True,
                   "run_artifact_sha256": {path.name: sha(path) for path in self.out.iterdir() if path.is_file()}})

    def test_complete_report_extracts_scores_and_preserves_causality_limits(self):
        source_hashes = {path: sha(path) for path in self.out.iterdir() if path.is_file()}
        result = generate_report(self.run_id, root=self.root)
        report = Path(result["report"])
        text = report.read_text()
        self.assertIn(f"**{self.metrics['test']['macro_roc_auc']:.4f}**", text)
        expected = "| Weighted F1 (positive-support weights) | " + " | ".join(
            f"{self.metrics[split]['weighted_f1']:.4f}" for split in PARTITIONS) + " | Not bootstrapped |"
        self.assertIn(expected, text)
        self.assertIn("Test specificity", text)
        self.assertIn("**17,161-parameter**", text)
        self.assertIn("13 supplied feature positions", text)
        self.assertIn("offline classifier, not streaming ECG processing", text)
        self.assertIn("exploratory rather than a fresh confirmatory test", text)
        self.assertIn("online, pre-update batch BCE", text)
        self.assertIn("not recovered 500 Hz waveform samples", text)
        self.assertIn("Recorded runner elapsed-timer duration: **0.25 minutes**", text)
        self.assertIn("UTC wall span was **16.03 minutes**", text)
        self.assertIn("not pure CPU execution time", text)
        self.assertIn("includes sleep/suspend gaps", text)
        self.assertIn(str(self.model / "training_curve.png"), text)
        self.assertIn(str(self.out / "test_confusion_matrices.png"), text)
        self.assertIn(str(self.root / "results/comparisons" / self.run_id / "comparison.md"), text)
        figures = Path(result["figure_directory"])
        self.assertEqual(len(list(figures.glob("*.png"))), 4)
        self.assertEqual(len(list(figures.glob("*.svg"))), 4)
        manifest = json.loads((figures / "report_manifest.json").read_text())
        self.assertFalse(manifest["new_model_inference_performed"])
        self.assertEqual(manifest["runtime_summary"]["runner_elapsed_timer_seconds"], 15.2)
        self.assertEqual(manifest["runtime_summary"]["utc_wall_elapsed_seconds"], 962.)
        self.assertEqual(manifest["output_sha256"][str(report)], sha(report))
        self.assertEqual(source_hashes, {path: sha(path) for path in source_hashes})
        with self.assertRaises(FileExistsError):
            generate_report(self.run_id, root=self.root)

    def test_wrong_metric_or_stale_verification_cannot_generate_report(self):
        path = self.out / "test_metrics.json"
        original = path.read_bytes()
        value = json.loads(original)
        value["weighted_f1"] += .01
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "test/weighted_f1"):
            generate_report(self.run_id, root=self.root)
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Verification is stale"):
            generate_report(self.run_id, root=self.root)
        self.assertFalse((self.root / "docs/ptbxl_spatiotemporal_report.md").exists())

    def test_prior_model_identity_is_rejected(self):
        self.training["model"] = "LeadImageCNNLSTM"
        write_json(self.out / "training_metadata.json", self.training)
        with self.assertRaisesRegex(ValueError, "Unsupported model identity"):
            generate_report(self.run_id, root=self.root)


class RuntimeReportingTest(unittest.TestCase):
    def test_wall_time_does_not_replace_recorded_timer_or_assume_cpu_time(self):
        run = {"total_runtime_seconds": 504., "started_utc": "2026-09-25T19:13:16Z",
               "completed_utc": "2026-09-25T19:29:18+00:00"}
        timing = _runtime_summary(run, {"runtime_seconds": 490.})
        self.assertEqual(timing["runner_elapsed_timer_seconds"], 504.)
        self.assertEqual(timing["training_elapsed_timer_seconds"], 490.)
        self.assertEqual(timing["utc_wall_elapsed_seconds"], 962.)
        self.assertIn("cannot quantify time asleep", _runtime_text(timing))
        run.update(started_utc="2026-09-26T01:00:00+04:00", completed_utc="2026-09-25T21:01:00Z")
        self.assertEqual(_runtime_summary(run, {"runtime_seconds": 30.})["utc_wall_elapsed_seconds"], 60.)

    def test_missing_timestamps_are_explicit_and_invalid_spans_rejected(self):
        timing = _runtime_summary({"total_runtime_seconds": 10.}, {"runtime_seconds": 8.})
        self.assertIsNone(timing["utc_wall_elapsed_seconds"])
        self.assertIn("UTC wall span is unavailable", _runtime_text(timing))
        for start, end in (("2026-09-25T00:00:00", "2026-09-25T00:00:10Z"),
                           ("2026-09-25T00:00:10Z", "2026-09-25T00:00:00Z"),
                           ("not a timestamp", "2026-09-25T00:00:00Z")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                _runtime_summary({"total_runtime_seconds": 10., "started_utc": start, "completed_utc": end}, {"runtime_seconds": 8.})


if __name__ == "__main__":
    unittest.main()
