"""Hand-calculated multilabel metrics and undefined-class/output integrity checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.evaluation.multilabel import evaluate_multilabel, save_multilabel_evaluation

NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
TRUTH = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0], [0, 0, 0, 1, 1]])
PROB = np.array([[.9, .8, .8, .1, .2], [.7, .9, .4, .2, .3], [.6, .4, .3, .9, .7], [.1, .2, .2, .6, .8]])


def source_rows():
    return [{"ecg_id": i + 1, "patient_id": str(i // 2 + 10), "split": "validation",
             "image_path": f"images/validation/ecg_{i+1:05d}.png",
             **{f"label_{name}": bool(y[j]) for j, name in enumerate(NAMES)}} for i, y in enumerate(TRUTH)]


class MultilabelMetricTests(unittest.TestCase):
    def test_metrics_match_hand_calculated_binary_decisions_and_ranks(self):
        result = evaluate_multilabel(TRUTH, PROB, NAMES)
        self.assertEqual(result["subset_accuracy"], .25)
        self.assertEqual(result["label_accuracy"], .7)
        self.assertEqual(result["hamming_loss"], .3)
        self.assertAlmostEqual(result["macro_precision"], 11 / 15)
        self.assertAlmostEqual(result["macro_recall"], .7)
        self.assertAlmostEqual(result["macro_f1"], 52 / 75)
        self.assertAlmostEqual(result["micro_f1"], .7)
        self.assertAlmostEqual(result["weighted_f1"], 52 / 75)
        self.assertAlmostEqual(result["macro_roc_auc"], .8)
        self.assertAlmostEqual(result["micro_roc_auc"], 161 / 200)
        self.assertAlmostEqual(result["macro_average_precision"], 53 / 60)
        self.assertAlmostEqual(result["micro_average_precision"], 731 / 900)
        expected = {
            "NORM": (2, 1, 1, 0, .5, .75, .75, 5 / 6),
            "MI": (1, 1, 1, 1, .5, .5, .75, 5 / 6),
            "STTC": (1, 0, 2, 1, 1., .75, 1., 1.),
            "CD": (2, 0, 2, 0, 1., 1., 1., 1.),
            "HYP": (1, 1, 1, 1, .5, .5, .5, .75),
        }
        for name, values in expected.items():
            label = result["per_label"][name]
            tp, fp, tn, fn, specificity, balanced, auc, ap = values
            self.assertEqual([label[k] for k in ("tp", "fp", "tn", "fn")], [tp, fp, tn, fn])
            self.assertEqual(result["confusion_matrices"][name], [[tn, fp], [fn, tp]])
            self.assertEqual(label["support"], 2)
            self.assertAlmostEqual(label["specificity"], specificity)
            self.assertAlmostEqual(label["balanced_accuracy"], balanced)
            self.assertAlmostEqual(label["roc_auc"], auc)
            self.assertAlmostEqual(label["average_precision"], ap)
        json.dumps(result, allow_nan=False)

    def test_undefined_labels_are_null_and_partial_macro_is_explicit(self):
        truth = np.zeros((4, 5), dtype=int)
        truth[:, 0] = 1
        truth[:2, 1] = 1
        result = evaluate_multilabel(truth, np.full((4, 5), .6), NAMES)
        self.assertIsNone(result["macro_roc_auc"])
        self.assertIsNone(result["macro_average_precision"])
        self.assertEqual(result["roc_auc_defined_labels"], ["MI"])
        self.assertEqual(result["average_precision_defined_labels"], ["NORM", "MI"])
        self.assertEqual(result["macro_roc_auc_defined_labels"], .5)
        self.assertEqual(result["macro_average_precision_defined_labels"], .75)
        self.assertIsNone(result["per_label"]["NORM"]["roc_auc"])
        self.assertEqual(result["per_label"]["NORM"]["average_precision"], 1.)
        self.assertIsNone(result["per_label"]["NORM"]["specificity"])
        self.assertIsNone(result["per_label"]["STTC"]["average_precision"])
        self.assertIsNone(result["per_label"]["STTC"]["balanced_accuracy"])
        self.assertAlmostEqual(result["micro_average_precision"], .3)
        json.dumps(result, allow_nan=False)

    def test_support_weighted_f1_differs_from_unweighted_macro(self):
        truth = np.zeros((5, 5), dtype=int)
        for column, support in enumerate((5, 1, 2, 3, 4)):
            truth[:support, column] = 1
        probabilities = np.full((5, 5), .1)
        probabilities[:, 0] = .9
        probabilities[0, 2] = .9
        probabilities[:, 3] = .9
        result = evaluate_multilabel(truth, probabilities, NAMES)
        # Label F1s are 1, 0, 2/3, 3/4, 0 with positive supports 5,1,2,3,4.
        self.assertAlmostEqual(result["macro_f1"], 29 / 60)
        self.assertAlmostEqual(result["weighted_f1"], 103 / 180)
        self.assertAlmostEqual(result["micro_f1"], 9 / 13)

    def test_all_negative_and_all_positive_truth_use_defined_metric_policies(self):
        for value in (0, 1):
            with self.subTest(value=value):
                truth = np.full((2, 5), value)
                result = evaluate_multilabel(truth, truth.astype(float), NAMES)
                self.assertEqual(result["subset_accuracy"], 1.)
                self.assertEqual(result["label_accuracy"], 1.)
                self.assertEqual(result["hamming_loss"], 0.)
                self.assertIsNone(result["macro_roc_auc"])
                self.assertIsNone(result["micro_roc_auc"])
                self.assertEqual(result["macro_average_precision"], 1. if value else None)
                self.assertEqual(result["micro_average_precision"], 1. if value else None)
                self.assertEqual(result["weighted_f1"], float(value))
                self.assertTrue(np.isfinite(result["binary_cross_entropy"]))
                self.assertLess(result["binary_cross_entropy"], 1e-12)
                json.dumps(result, allow_nan=False)

    def test_independent_probabilities_and_positive_threshold_tie(self):
        probabilities = np.full((3, 5), .5)
        self.assertTrue(np.all(probabilities.sum(axis=1) > 1))
        result = evaluate_multilabel(np.ones((3, 5)), probabilities, NAMES)
        self.assertEqual(result["subset_accuracy"], 1.)
        self.assertAlmostEqual(result["binary_cross_entropy"], np.log(2))
        self.assertEqual(result["threshold"], .5)
        with self.assertRaisesRegex(ValueError, "fixed threshold"):
            evaluate_multilabel(TRUTH, PROB, NAMES, threshold=.6)

    def test_invalid_shapes_values_and_names_rejected(self):
        bad_probabilities = [np.full((4, 5), np.nan), np.full((4, 5), np.inf), np.full((4, 5), -0.1),
                             np.full((4, 5), 1.1), PROB[:, :4], PROB[:3], PROB.astype(complex)]
        for p in bad_probabilities:
            with self.subTest(probability_shape=p.shape):
                with self.assertRaises(ValueError):
                    evaluate_multilabel(TRUTH, p, NAMES)
        for y in (TRUTH.ravel(), TRUTH[:, :4], np.full((4, 5), .5), np.full((4, 5), np.nan), TRUTH.astype(str)):
            with self.subTest(truth_shape=y.shape):
                with self.assertRaises(ValueError):
                    evaluate_multilabel(y, PROB, NAMES)
        with self.assertRaises(ValueError):
            evaluate_multilabel(np.empty((0, 5)), np.empty((0, 5)), NAMES)
        for names in (NAMES[:4], ["NORM"] * 5, ["", "MI", "STTC", "CD", "HYP"]):
            with self.assertRaises(ValueError):
                evaluate_multilabel(TRUTH, PROB, names)

    def test_metrics_are_deterministic(self):
        first = evaluate_multilabel(TRUTH, PROB, NAMES)
        second = evaluate_multilabel(TRUTH.tolist(), PROB.tolist(), tuple(NAMES))
        self.assertEqual(first, second)


class MultilabelOutputTests(unittest.TestCase):
    def test_saved_outputs_preserve_source_truth_probabilities_and_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            metrics = save_multilabel_evaluation(out, "validation", source_rows(), TRUTH, PROB, NAMES)
            self.assertEqual(json.loads((out / "validation_metrics.json").read_text()), metrics)
            with (out / "validation_predictions.csv").open() as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 4)
            self.assertEqual([row["ecg_id"] for row in predictions], ["1", "2", "3", "4"])
            self.assertEqual([row["patient_id"] for row in predictions], ["10", "10", "11", "11"])
            for i, row in enumerate(predictions):
                for j, name in enumerate(NAMES):
                    self.assertEqual(int(row[f"truth_{name}"]), int(TRUTH[i, j]))
                    self.assertAlmostEqual(float(row[f"probability_{name}"]), PROB[i, j])
                    self.assertEqual(int(row[f"prediction_{name}"]), int(PROB[i, j] >= .5))
            with (out / "validation_per_label.csv").open() as handle:
                per_label = list(csv.DictReader(handle))
            self.assertEqual([row["label"] for row in per_label], NAMES)
            with Image.open(out / "validation_confusion_matrices.png") as image:
                image.load()
                self.assertGreater(image.width, image.height)

    def test_bad_manifest_alignment_rejected_before_writing(self):
        bad_sets = []
        rows = source_rows()
        rows[0]["label_NORM"] = False
        bad_sets.append(rows)
        rows = source_rows()
        rows[1]["ecg_id"] = rows[0]["ecg_id"]
        bad_sets.append(rows)
        rows = source_rows()
        rows[0]["split"] = "test"
        bad_sets.append(rows)
        rows = source_rows()
        del rows[0]["patient_id"]
        bad_sets.append(rows)
        bad_sets.append(source_rows()[:-1])
        for rows in bad_sets:
            with self.subTest(rows=rows):
                with tempfile.TemporaryDirectory() as directory:
                    out = Path(directory) / "evaluation"
                    with self.assertRaises(ValueError):
                        save_multilabel_evaluation(out, "validation", rows, TRUTH, PROB, NAMES)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
