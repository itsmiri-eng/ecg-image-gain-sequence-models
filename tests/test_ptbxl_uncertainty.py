"""Synthetic patient multiplicity, percentile and deterministic bootstrap checks."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.evaluation.ptbxl_uncertainty import (
    METRIC_NAMES, _patient_groups, _sample_record_indices, patient_cluster_bootstrap,
)

NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
TRUTH = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]] * 2)
PROB = np.where(TRUTH == 1, .9, .1)
PATIENTS = ["A", "A", "B", "B"]


class FixedPatientDraws:
    def integers(self, low, high, size):
        if (low, high, size) != (0, 3, 3):
            raise AssertionError("Must sample exactly three patient clusters")
        return np.array([2, 0, 2])


class PatientBootstrapTests(unittest.TestCase):
    def test_cluster_draws_keep_every_record_and_repeated_patient_multiplicity(self):
        groups = _patient_groups(["A", "B", "A", "C", "C", "C"], 6)
        indices, draws = _sample_record_indices(groups, FixedPatientDraws())
        np.testing.assert_array_equal(draws, [2, 0, 2])
        np.testing.assert_array_equal(indices, [3, 4, 5, 0, 2, 3, 4, 5])
        self.assertEqual(len(indices), 8)  # Cluster sizes vary; original cohort has six ECGs.
        counts = np.bincount(indices, minlength=6)
        np.testing.assert_array_equal(counts, [1, 0, 1, 2, 2, 2])

    def test_perfect_predictions_have_exact_unit_intervals_for_each_metric(self):
        result = patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=12)
        self.assertEqual(result["n_unique_patients"], 2)
        self.assertEqual(result["patient_draws_per_resample"], 2)
        self.assertEqual(result["patient_record_count_distribution"], {"2": 2})
        self.assertEqual(result["resampled_record_counts"], {"minimum": 4, "maximum": 4, "mean": 4.})
        for name in METRIC_NAMES:
            value = result["metrics"][name]
            self.assertEqual(value["point_estimate"], 1.)
            self.assertEqual((value["ci_lower"], value["ci_upper"]), (1., 1.))
            self.assertEqual((value["defined_resamples"], value["undefined_resamples"]), (12, 0))
        json.dumps(result, allow_nan=False)

    def test_linear_percentile_boundaries_and_metric_specific_undefined_counts(self):
        point = {name: .5 for name in METRIC_NAMES}
        replicates = [{name: value for name in METRIC_NAMES} for value in (0., .25, .5, .75, 1.)]
        replicates[0]["macro_roc_auc"] = None
        replicates[4]["macro_roc_auc"] = None
        with patch("src.evaluation.ptbxl_uncertainty.evaluate_multilabel", side_effect=[point, *replicates]) as evaluator:
            result = patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=5)
        self.assertEqual(evaluator.call_count, 6)
        complete = result["metrics"]["macro_f1"]
        self.assertAlmostEqual(complete["ci_lower"], .025)
        self.assertAlmostEqual(complete["ci_upper"], .975)
        incomplete = result["metrics"]["macro_roc_auc"]
        self.assertAlmostEqual(incomplete["ci_lower"], .2625)
        self.assertAlmostEqual(incomplete["ci_upper"], .7375)
        self.assertEqual((incomplete["defined_resamples"], incomplete["undefined_resamples"]), (3, 2))

    def test_no_positive_truth_leaves_ranking_intervals_null(self):
        truth = np.zeros((3, 5), dtype=int)
        result = patient_cluster_bootstrap(truth, np.full((3, 5), .1), ["A", "B", "C"], NAMES, n_resamples=7)
        for name in ("macro_roc_auc", "macro_average_precision"):
            interval = result["metrics"][name]
            self.assertIsNone(interval["point_estimate"])
            self.assertIsNone(interval["ci_lower"])
            self.assertIsNone(interval["ci_upper"])
            self.assertEqual((interval["defined_resamples"], interval["undefined_resamples"]), (0, 7))
        self.assertEqual(result["metrics"]["macro_f1"]["ci_lower"], 0.)
        self.assertEqual(result["metrics"]["subset_accuracy"]["ci_lower"], 1.)
        json.dumps(result, allow_nan=False)

    def test_single_defined_replicate_is_reported_without_an_interval(self):
        result = patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=1)
        for interval in result["metrics"].values():
            self.assertEqual(interval["defined_resamples"], 1)
            self.assertIsNone(interval["ci_lower"])
            self.assertIsNone(interval["ci_upper"])
            self.assertEqual(interval["interval_status"], "insufficient_defined_resamples")

    def test_same_seed_is_deterministic_and_global_numpy_rng_is_unchanged(self):
        original_global_state = np.random.get_state()
        try:
            np.random.seed(83)
            before = np.random.get_state()
            one = patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=8, seed=2024)
            two = patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=8, seed=2024)
            after = np.random.get_state()
            self.assertEqual(one, two)
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
        finally:
            np.random.set_state(original_global_state)

    def test_invalid_patient_ids_and_sampling_arguments_are_rejected(self):
        for ids in (["A"], [["A"], ["A"], ["B"], ["B"]], ["A", None, "B", "B"],
                    ["A", "", "B", "B"], [1, np.nan, 2, 2], [1, 1.5, 2, 2], [True, True, False, False]):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    patient_cluster_bootstrap(TRUTH, PROB, ids, NAMES, n_resamples=2)
        for bad_n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=bad_n)
        for bad_seed in (-1, .5, True):
            with self.assertRaises(ValueError):
                patient_cluster_bootstrap(TRUTH, PROB, PATIENTS, NAMES, n_resamples=2, seed=bad_seed)

    def test_numeric_patient_ids_are_normalized_before_grouping(self):
        groups = _patient_groups([1, 1., "1", 2], 4)
        self.assertEqual(len(groups), 2)
        np.testing.assert_array_equal(groups[0], [0, 1, 2])
        np.testing.assert_array_equal(groups[1], [3])


if __name__ == "__main__":
    unittest.main()
