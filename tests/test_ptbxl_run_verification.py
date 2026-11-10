"""Synthetic saved-artifact checks; never load or infer on the real cohort."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.evaluation.multilabel import evaluate_multilabel
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.verify_ptbxl_run import (
    CLASS_NAMES, _compare, _hash, _json, _safe_path, _verify_bootstrap,
    _verify_history, _verify_log, _verify_per_label, _verify_predictions, verify,
)


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.y = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0],
                           [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]], dtype=np.int8)
        self.p = np.where(self.y, .8, .2)
        self.p[0, 0] = .5
        self.manifest = []
        self.predictions = []
        for index, (truth, probability) in enumerate(zip(self.y, self.p), start=1):
            source = {"ecg_id": str(index), "patient_id": str((index + 1) // 2),
                      "image_path": f"images/train/ecg_{index:05d}.png"}
            row = dict(source)
            for label, value, score in zip(CLASS_NAMES, truth, probability):
                source[f"label_{label}"] = str(bool(value))
                row.update({f"truth_{label}": int(value), f"probability_{label}": float(score),
                            f"prediction_{label}": int(score >= .5)})
            self.manifest.append(source)
            self.predictions.append(row)
        self.metrics = evaluate_multilabel(self.y, self.p, CLASS_NAMES)
        self.path = self.root / "predictions.csv"
        write_csv(self.path, self.predictions)

    def tearDown(self):
        self.temporary.cleanup()

    def test_recomputes_predictions_with_overlap_and_fixed_threshold_tie(self):
        y, p = _verify_predictions(self.path, self.manifest)
        np.testing.assert_array_equal(self.y, y)
        np.testing.assert_array_equal(self.p, p)
        self.assertGreater(p[0].sum(), 1)
        self.assertEqual(p[0, 0], .5)
        _compare(self.metrics, evaluate_multilabel(y, p, CLASS_NAMES))

    def test_identity_truth_and_prediction_tampering_rejected(self):
        for field, replacement in (("ecg_id", "99"), ("patient_id", "99"),
                                   ("truth_NORM", 0), ("prediction_NORM", 0),
                                   ("probability_NORM", "nan"), ("probability_NORM", "1.1")):
            with self.subTest(field=field, value=replacement):
                rows = copy.deepcopy(self.predictions)
                rows[0][field] = replacement
                write_csv(self.path, rows)
                with self.assertRaises(ValueError):
                    _verify_predictions(self.path, self.manifest)

    def test_reordered_rows_or_label_columns_are_rejected(self):
        write_csv(self.path, list(reversed(self.predictions)))
        with self.assertRaises(ValueError):
            _verify_predictions(self.path, self.manifest)
        fields = list(self.predictions[0])
        fields[3], fields[6] = fields[6], fields[3]
        write_csv(self.path, self.predictions, fields)
        with self.assertRaisesRegex(ValueError, "class order"):
            _verify_predictions(self.path, self.manifest)

    def test_recursive_metric_comparison_checks_confusions_and_undefined_values(self):
        actual = copy.deepcopy(self.metrics)
        actual["macro_f1"] += 5e-13
        _compare(self.metrics, actual)
        actual["confusion_matrices"]["MI"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "confusion_matrices"):
            _compare(self.metrics, actual)
        with self.assertRaises(ValueError):
            _compare({"auc": None}, {"auc": 0})
        with self.assertRaises(ValueError):
            _compare({"count": 1}, {"count": True})
        with self.assertRaises(ValueError):
            _compare({"score": 0.5}, {"score": float("nan")})

    def test_per_label_rows_verify_null_and_numeric_values(self):
        metrics = evaluate_multilabel(np.zeros_like(self.y), self.p, CLASS_NAMES)
        path = self.root / "per_label.csv"
        rows = [{"label": name, **metrics["per_label"][name]} for name in CLASS_NAMES]
        write_csv(path, rows)
        _verify_per_label(path, metrics)
        rows[0]["roc_auc"] = 0
        write_csv(path, rows)
        with self.assertRaises(ValueError):
            _verify_per_label(path, metrics)

    def history(self):
        # Epochs 2 and 3 tie in all three criteria; epoch 2 must be selected.
        keys = [(.7, .8, .4), (.8, .81, .35), (.8, .81, .35), (.79, .9, .2)]
        final = dict(self.metrics, macro_roc_auc=.8, macro_average_precision=.81,
                     binary_cross_entropy=.35)
        rows = []
        for index, (auc, ap, bce) in enumerate(keys, start=1):
            values = dict(final, macro_roc_auc=auc, macro_average_precision=ap,
                          binary_cross_entropy=bce)
            row = {"epoch": str(index), "learning_rate": ".001", "train_online_bce": ".4",
                   **{f"val_{k}": str(v) for k, v in values.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool)},
                   "checkpoint_improved": str(index <= 2),
                   "epochs_without_selection_improvement": str(max(0, index - 2)),
                   "max_gradient_norm_before_clip": "1.5", "epoch_seconds": "1"}
            rows.append(row)
        training = {"best_epoch": 2, "epochs_run": 4, "best_selection_key": [.8, .81, -.35],
                    "stopped_early": False, "best_validation_metrics": final,
                    "selected_epoch_online_train_bce": .4}
        return rows, training, final, {"max_epochs": 4, "patience": 2}

    def test_history_lexicographic_selection_keeps_earliest_exact_tie(self):
        args = self.history()
        result = _verify_history(*args)
        self.assertEqual(result["best_epoch"], 2)
        args[1]["best_epoch"] = 3
        with self.assertRaisesRegex(ValueError, "best_epoch"):
            _verify_history(*args)

    def test_history_checks_ap_then_bce_and_early_stopping(self):
        rows, training, final, config = self.history()
        rows[2]["val_macro_average_precision"] = ".82"
        with self.assertRaisesRegex(ValueError, "improvement"):
            _verify_history(rows, training, final, config)
        rows, training, final, config = self.history()
        rows[2]["val_binary_cross_entropy"] = ".34"
        with self.assertRaisesRegex(ValueError, "improvement"):
            _verify_history(rows, training, final, config)
        rows, training, final, config = self.history()
        config["patience"] = 1
        with self.assertRaisesRegex(ValueError, "continues after early stopping"):
            _verify_history(rows, training, final, config)

    def test_bootstrap_points_cluster_accounting_and_bounds(self):
        patients = ["a", "a", "b", "c"]
        config = {"resamples": 20, "seed": 7, "confidence_level": .95}
        ci = patient_cluster_bootstrap(self.y, self.p, patients, CLASS_NAMES, n_resamples=20, seed=7)
        result = _verify_bootstrap(ci, self.metrics, patients, config)
        self.assertFalse(result["intervals_recomputed"])
        self.assertEqual(result["n_unique_patients"], 3)
        for field, replacement in (("point_estimate", .123), ("ci_upper", 1.1),
                                   ("undefined_resamples", 100)):
            bad = copy.deepcopy(ci)
            bad["metrics"]["macro_f1"][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                _verify_bootstrap(bad, self.metrics, patients, config)
        bad = copy.deepcopy(ci)
        bad["patient_record_count_distribution"] = {"1": 4}
        with self.assertRaises(ValueError):
            _verify_bootstrap(bad, self.metrics, patients, config)

    def test_bootstrap_all_undefined_label_ranking_is_not_invented(self):
        y = np.zeros_like(self.y)
        metrics = evaluate_multilabel(y, self.p, CLASS_NAMES)
        ci = patient_cluster_bootstrap(y, self.p, ["a", "a", "b", "b"], CLASS_NAMES, n_resamples=3, seed=7)
        _verify_bootstrap(ci, metrics, ["a", "a", "b", "b"],
                          {"resamples": 3, "seed": 7, "confidence_level": .95})
        self.assertIsNone(ci["metrics"]["macro_roc_auc"]["ci_lower"])
        ci["metrics"]["macro_roc_auc"]["ci_lower"] = 0
        with self.assertRaises(ValueError):
            _verify_bootstrap(ci, metrics, ["a", "a", "b", "b"],
                              {"resamples": 3, "seed": 7, "confidence_level": .95})

    def log_fixture(self):
        cfg = {"dataset_id": "synthetic", "split_version": "synthetic-folds", "seed": 42,
               "training": {"seed": 42}, "preprocessing": {"fit": "train"}}
        run = {"started_utc": "2026-09-24T00:00:00+00:00"}
        row = {"experiment_id": "new", "status": "completed", "model": "PTBXL_Image_CNN_LSTM",
               "date": run["started_utc"], "dataset_version": cfg["dataset_id"],
               "split_version": cfg["split_version"], "random_seed": "42", "accuracy": "",
               "balanced_accuracy": "", "output_directory": "results/ptbxl_cnn_lstm/new",
               "hyperparameters": json.dumps(cfg["training"]), "preprocessing": json.dumps(cfg["preprocessing"]),
               "macro_f1": str(self.metrics["macro_f1"]), "weighted_f1": str(self.metrics["weighted_f1"])}
        prior, current = self.root / "prior.csv", self.root / "current.csv"
        old = dict(row, experiment_id="old")
        write_csv(prior, [old])
        write_csv(current, [old, row])
        return prior, current, "new", cfg, run, self.metrics

    def test_log_prefix_current_row_and_future_appends(self):
        args = self.log_fixture()
        result = _verify_log(*args)
        self.assertEqual(result["prior_log_rows"], 1)
        self.assertEqual(result["completed_run_log_sha256"], hashlib.sha256(args[1].read_bytes()).hexdigest())
        with args[1].open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        rows.append(dict(rows[-1], experiment_id="later"))
        write_csv(args[1], rows)
        later = _verify_log(*args)
        self.assertEqual(later["completed_run_log_sha256"], result["completed_run_log_sha256"])
        self.assertEqual(later["appended_log_rows"], 2)

    def test_log_duplicate_nonblank_accuracy_or_changed_prefix_rejected(self):
        for mode in ("duplicate", "accuracy", "prefix"):
            args = self.log_fixture()
            with args[1].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            if mode == "duplicate":
                rows.append(dict(rows[-1]))
            elif mode == "accuracy":
                rows[-1]["accuracy"] = ".95"
            else:
                rows[0]["model"] = "altered"
            write_csv(args[1], rows)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                _verify_log(*args)

    def test_hash_json_and_path_validation(self):
        path = self.root / "artifact.json"
        path.write_text('{"value": 1}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _hash(path, digest)
        path.write_text('{"value": 2}')
        with self.assertRaises(ValueError):
            _hash(path, digest)
        for content in ('{"x": 1, "x": 2}', '{"x": NaN}'):
            path.write_text(content)
            with self.assertRaises(ValueError):
                _json(path)
        with self.assertRaises(ValueError):
            _safe_path(self.root, "../escape")

    def test_read_only_verifier_refuses_incomplete_run(self):
        out = self.root / "results/ptbxl_cnn_lstm/test"
        out.mkdir(parents=True)
        for filename, value in (("run.json", {"status": "running"}), ("config.json", {}), ("provenance.json", {})):
            (out / filename).write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "run status"):
            verify("test", root=self.root)
        self.assertFalse((out / "verification.json").exists())
        with self.assertRaisesRegex(ValueError, "Invalid run ID"):
            verify("../escape", root=self.root)


if __name__ == "__main__":
    unittest.main()
