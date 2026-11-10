"""Synthetic paired comparisons; no real ECG images or predictions are read."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.compare_ptbxl_models import (
    ARTIFACTS, CLASS_NAMES, METRICS, ROOT, _Inputs, _align, _load_run,
    _prediction_arrays, _run_timing, _summary, _write_csv, compare_runs, paired_patient_bootstrap,
)
from src.evaluation.multilabel import evaluate_multilabel


Y = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]] * 2, dtype=np.int8)
P = np.where(Y, .8, .2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixture(root, family, run_id, probabilities=P):
    """Small consistent saved-run fixture, including verifier-bound file hashes."""
    out = root / "results" / family / run_id
    out.mkdir(parents=True)
    model = root / "models" / family / run_id
    model.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"synthetic checkpoint; never loaded")
    checkpoint_hash = sha(model / "best.pt")
    cache = root / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "metadata.json", "cache_manifest.csv"):
        if not (cache / name).exists():
            (cache / name).write_text("synthetic identical cache binding")
    sources = {}
    for name in ("src/evaluation/multilabel.py", "src/evaluation/ptbxl_uncertainty.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        sources[name] = sha(path)
    cfg = {"model": family, "dataset_id": "synthetic", "split_version": "official-folds",
           "class_names": CLASS_NAMES, "input_shape": [12, 120, 1120], "threshold": .5,
           "seed": 42, "training": {"seed": 42}, "preprocessing": {"method": "same"},
           "selection_rule": "validation AUROC/AP/-BCE", "source_dataset_root": "data/source",
           "cache_root": "data/cache", "source_manifest_sha256": "1" * 64,
           "source_verification_sha256": "2" * 64, "source_protocol_sha256": "3" * 64}
    write_json(out / "config.json", cfg)
    config_hash = sha(out / "config.json")
    normalization = {"fit_partition": "train", "mean": .1, "std": .2}
    provenance = {"configuration_sha256": config_hash, "normalization": normalization,
                  "selection_rule": cfg["selection_rule"], "source_sha256": sources,
                  "cache_sha256": {name: sha(cache / name) for name in ("protocol.json", "metadata.json", "cache_manifest.csv")}}
    metrics = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        rows = []
        for index, (truth, p) in enumerate(zip(Y, probabilities)):
            row = {"ecg_id": str(split_index * 10 + index + 1),
                   "patient_id": str(split_index * 10 + index // 2 + 1),
                   "image_path": f"images/{split}/{index}.png"}
            for label, value, probability in zip(CLASS_NAMES, truth, p):
                row.update({f"truth_{label}": int(value), f"probability_{label}": float(probability),
                            f"prediction_{label}": int(probability >= .5)})
            rows.append(row)
        _write_csv(out / f"{split}_predictions.csv", rows)
        metrics[split] = evaluate_multilabel(Y, probabilities, CLASS_NAMES)
        write_json(out / f"{split}_metrics.json", metrics[split])
    training = {"config": cfg["training"], "normalization": normalization, "class_names": CLASS_NAMES,
                "num_parameters": 100, "runtime_seconds": 10., "best_epoch": 1, "epochs_run": 1,
                "best_validation_metrics": metrics["validation"]}
    run = {"status": "completed", "experiment_id": run_id, "model": cfg["model"], "dataset_id": cfg["dataset_id"],
           "split_version": cfg["split_version"], "threshold": .5, "training": training, "metrics": metrics,
           "checkpoint_sha256": checkpoint_hash, "test_inference_calls": 1, "total_runtime_seconds": 12.,
           "started_utc": "2026-09-26T00:00:00+00:00", "completed_utc": "2026-09-26T00:00:24+00:00",
           "model_directory": model.relative_to(root).as_posix()}
    frozen = {"training": training, "threshold": .5, "test_inference_started": False,
              "selection_rule": cfg["selection_rule"], "checkpoint_sha256": checkpoint_hash,
              "configuration_sha256": config_hash}
    state = {"status": "completed", "inference_calls": 1, "threshold": .5,
             "checkpoint_sha256": checkpoint_hash, "records": len(Y)}
    for name, value in (("run.json", run), ("provenance.json", provenance), ("training_metadata.json", training),
                         ("selection_frozen.json", frozen), ("test_inference_state.json", state)):
        write_json(out / name, value)
    refresh_verification(out, run_id, checkpoint_hash, config_hash)
    return out


def refresh_verification(out, run_id, checkpoint_hash=None, config_hash=None):
    old = json.loads((out / "verification.json").read_text()) if (out / "verification.json").exists() else {}
    write_json(out / "verification.json", {
        "status": "passed", "run_id": run_id, "failures": [],
        "checkpoint_sha256": checkpoint_hash or old["checkpoint_sha256"],
        "configuration_sha256": config_hash or old["configuration_sha256"],
        "run_artifact_sha256": {p.name: sha(p) for p in out.iterdir() if p.name != "verification.json"},
    })


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_predictions_give_exact_zero_differences_and_intervals(self):
        result = paired_patient_bootstrap(Y, P, P, ["a", "a", "b", "b"], CLASS_NAMES, n_resamples=8)
        for name in METRICS:
            interval = result["metrics"][name]
            self.assertEqual((interval["difference"], interval["ci_lower"], interval["ci_upper"]), (0., 0., 0.))
            self.assertEqual(interval["defined_resamples"], 8)
            self.assertEqual(interval["interpretation"], "includes_zero_no_resolved_difference")

    def test_one_patient_draw_shared_between_models_with_full_multiplicity(self):
        y = np.vstack((Y, Y[:2]))
        old, new = np.arange(30).reshape(6, 5) / 50, np.arange(30).reshape(6, 5) / 50 + .1
        fixed = np.array([3, 4, 5, 0, 2, 3, 4, 5])  # C, A, C: clusters of 3, 2, 3 records.
        with patch("src.compare_ptbxl_models._sample_record_indices", return_value=(fixed, np.array([2, 0, 2]))) as sampler, \
             patch("src.compare_ptbxl_models.evaluate_multilabel", wraps=evaluate_multilabel) as metric:
            result = paired_patient_bootstrap(y, old, new, ["A", "B", "A", "C", "C", "C"], CLASS_NAMES, n_resamples=2)
        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(metric.call_count, 6)  # two points + two calls per paired draw
        for old_call, new_call in ((2, 3), (4, 5)):
            np.testing.assert_array_equal(metric.call_args_list[old_call].args[0], y[fixed])
            np.testing.assert_array_equal(metric.call_args_list[new_call].args[0], y[fixed])
            np.testing.assert_array_equal(metric.call_args_list[old_call].args[1], old[fixed])
            np.testing.assert_array_equal(metric.call_args_list[new_call].args[1], new[fixed])
        self.assertEqual(result["resampled_record_counts"], {"minimum": 8, "maximum": 8, "mean": 8.})

    def test_accuracy_definitions_and_difference_direction(self):
        imperfect = P.copy()
        imperfect[:, 0] = 1 - imperfect[:, 0]
        result = paired_patient_bootstrap(Y, imperfect, P, ["a", "a", "b", "b"], CLASS_NAMES, n_resamples=3)
        self.assertEqual(result["metrics"]["subset_accuracy"]["old_point_estimate"], 0.)
        self.assertEqual(result["metrics"]["subset_accuracy"]["difference"], 1.)
        self.assertEqual(result["metrics"]["label_accuracy"]["old_point_estimate"], .8)
        self.assertAlmostEqual(result["metrics"]["label_accuracy"]["difference"], .2)
        self.assertEqual(result["metrics"]["subset_accuracy"]["interpretation"], "above_zero_new_higher_conditional")

    def test_undefined_paired_ranking_stays_null(self):
        result = paired_patient_bootstrap(np.zeros_like(Y), P, P, [1, 1, 2, 2], CLASS_NAMES, n_resamples=3)
        for name in ("macro_roc_auc", "macro_average_precision"):
            self.assertIsNone(result["metrics"][name]["difference"])
            self.assertIsNone(result["metrics"][name]["ci_lower"])
            self.assertEqual(result["metrics"][name]["undefined_resamples"], 3)

    def test_seed_determinism_and_global_rng_unchanged(self):
        before = np.random.get_state()
        one = paired_patient_bootstrap(Y, P, 1 - P, [1, 2, 3, 4], CLASS_NAMES, n_resamples=5, seed=17)
        two = paired_patient_bootstrap(Y, P, 1 - P, [1, 2, 3, 4], CLASS_NAMES, n_resamples=5, seed=17)
        after = np.random.get_state()
        self.assertEqual(one, two)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_resampling_and_shape_arguments(self):
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                paired_patient_bootstrap(Y, P, P, [1, 1, 2, 2], CLASS_NAMES, n_resamples=n)
        with self.assertRaises(ValueError):
            paired_patient_bootstrap(Y, P, P[:-1], [1, 1, 2, 2], CLASS_NAMES, n_resamples=2)


class SavedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.old_out = fixture(self.root, "ptbxl_cnn_lstm", "old")
        self.new_out = fixture(self.root, "ptbxl_spatiotemporal", "new")

    def tearDown(self):
        self.temporary.cleanup()

    def load_pair(self):
        inputs = _Inputs()
        old = _load_run(self.root, "ptbxl_cnn_lstm", "old", inputs)
        new = _load_run(self.root, "ptbxl_spatiotemporal", "new", inputs)
        return old, new, inputs

    def test_valid_artifact_load_alignment_and_inputs_unchanged(self):
        old, new, inputs = self.load_pair()
        _align(old, new)
        inputs.unchanged()
        self.assertEqual(old["partitions"]["test"]["metrics"]["n_samples"], 4)

    def test_recorded_timer_and_utc_wall_span_remain_distinct(self):
        run = {"total_runtime_seconds": 504., "started_utc": "2026-09-26T00:00:00Z",
               "completed_utc": "2026-09-26T04:16:02+04:00"}
        timing = _run_timing(run, {"runtime_seconds": 500.})
        self.assertEqual(timing["training_recorded_timer_seconds"], 500.)
        self.assertEqual(timing["run_recorded_timer_seconds"], 504.)
        self.assertEqual(timing["run_utc_wall_span_seconds"], 962.)
        self.assertEqual(timing["run_completed_utc"], "2026-09-26T00:16:02+00:00")
        for bad in ("2026-09-26T00:16:02", "invalid", "2026-09-25T23:59:59Z"):
            with self.subTest(timestamp=bad), self.assertRaises(ValueError):
                _run_timing(dict(run, completed_utc=bad), {"runtime_seconds": 500.})

    def test_alignment_refuses_reordering_ids_truth_and_preprocessing(self):
        old, new, _ = self.load_pair()
        for field in ("ecg_id", "patient_id", "truth_NORM"):
            bad = copy.deepcopy(new)
            bad["partitions"]["test"]["rows"][0][field] = "999"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "paired alignment"):
                _align(old, bad)
        bad = copy.deepcopy(new)
        bad["partitions"]["test"]["rows"].reverse()
        with self.assertRaises(ValueError):
            _align(old, bad)
        for section, field, replacement in (("config", "preprocessing", {"method": "different"}),
                                             ("config", "threshold", .6),
                                             ("provenance", "cache_sha256", {})):
            bad = copy.deepcopy(new)
            bad[section][field] = replacement
            with self.assertRaises(ValueError):
                _align(old, bad)

    def test_stale_hash_failed_run_and_wrong_verification_are_rejected(self):
        path = self.new_out / "test_predictions.csv"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "artifact hash changed"):
            _load_run(self.root, "ptbxl_spatiotemporal", "new", _Inputs())
        path.write_bytes(original)
        run_path = self.new_out / "run.json"
        run = json.loads(run_path.read_text())
        run["status"] = "failed"
        write_json(run_path, run)
        refresh_verification(self.new_out, "new")
        with self.assertRaisesRegex(ValueError, "completed matching"):
            _load_run(self.root, "ptbxl_spatiotemporal", "new", _Inputs())
        verification_path = self.new_out / "verification.json"
        verified = json.loads(verification_path.read_text())
        verified["run_id"] = "wrong"
        write_json(verification_path, verified)
        with self.assertRaisesRegex(ValueError, "matching passed"):
            _load_run(self.root, "ptbxl_spatiotemporal", "new", _Inputs())

    def test_threshold_decision_and_column_order_rejected(self):
        fields, rows = _Inputs().csv(self.new_out / "test_predictions.csv")
        rows[0]["prediction_NORM"] = "0"
        with self.assertRaisesRegex(ValueError, "fixed-threshold"):
            _prediction_arrays(fields, rows, "test")
        fields[3], fields[6] = fields[6], fields[3]
        with self.assertRaisesRegex(ValueError, "class/column order"):
            _prediction_arrays(fields, rows, "test")

    def test_validation_ranking_does_not_use_test_outcome(self):
        old, new, _ = self.load_pair()
        new["partitions"]["test"]["metrics"]["macro_roc_auc"] = .1
        ci = paired_patient_bootstrap(Y, P, P, [1, 1, 2, 2], CLASS_NAMES, n_resamples=2)
        result = _summary(old, new, ci)
        self.assertEqual(result["validation_ranking"]["ordering"], "exact_tie")
        self.assertTrue(result["test_comparison_exploratory"])
        self.assertIn("training-seed", result["limitations"])

    def test_comparison_writes_all_artifacts_and_refuses_overwrite(self):
        # Exercise real 500 paired draws on four synthetic rows, including real plots.
        result = compare_runs("old", "new", root=self.root)
        output = self.root / "results/comparisons/new"
        self.assertEqual(set(p.name for p in output.iterdir()), set(ARTIFACTS))
        self.assertEqual(result["paired_test_bootstrap"]["n_resamples"], 500)
        self.assertEqual(result["paired_test_bootstrap"]["seed"], 2024)
        for name, count in (("partition_metrics.csv", 6), ("per_label_metrics.csv", 30), ("metric_differences.csv", 18)):
            with (output / name).open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), count)
        for name, digest in result["output_artifact_sha256"].items():
            self.assertEqual(sha(output / name), digest)
        report = (output / "comparison.md").read_text()
        self.assertIn("includes zero", report)
        self.assertIn("Validation ordering (separate from test)", report)
        self.assertEqual(result["models"]["old"]["run_recorded_timer_seconds"], 12.)
        self.assertEqual(result["models"]["old"]["run_utc_wall_span_seconds"], 24.)
        self.assertIn("Training recorded timer (s)", report)
        self.assertIn("Run UTC wall span (s)", report)
        self.assertIn("no speed or efficiency conclusion", report)
        self.assertNotIn("Measured wall times", report)
        with (output / "partition_metrics.csv").open(newline="") as handle:
            first = next(csv.DictReader(handle))
        self.assertEqual(float(first["training_recorded_timer_seconds"]), 10.)
        self.assertEqual(float(first["run_recorded_timer_seconds"]), 12.)
        self.assertEqual(float(first["run_utc_wall_span_seconds"]), 24.)
        with self.assertRaises(FileExistsError):
            compare_runs("old", "new", root=self.root)

    def test_output_cannot_modify_old_run(self):
        with self.assertRaisesRegex(ValueError, "cannot modify either run"):
            compare_runs("old", "new", root=self.root, output_dir=self.old_out, overwrite=True)


if __name__ == "__main__":
    unittest.main()
