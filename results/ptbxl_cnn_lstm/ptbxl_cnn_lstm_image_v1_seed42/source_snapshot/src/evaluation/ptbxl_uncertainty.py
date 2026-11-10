"""Patient-cluster bootstrap uncertainty conditional on a frozen ECG model.

Every patient draw includes all of that patient's observed ECG records, with
full multiplicity when a patient is drawn repeatedly. No thresholds, calibration
parameters or models are fitted. The bootstrap seed does not change training RNGs.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from src.evaluation.multilabel import evaluate_multilabel

METRIC_NAMES = (
    "macro_roc_auc", "macro_average_precision", "macro_f1", "micro_f1",
    "subset_accuracy", "label_accuracy",
)


def _patient_groups(patient_ids, n_records):
    ids = np.asarray(patient_ids, dtype=object)
    if ids.ndim != 1 or len(ids) != n_records:
        raise ValueError("Patient IDs must be a one-dimensional identifier per ECG record")
    groups = defaultdict(list)
    for index, value in enumerate(ids):
        if isinstance(value, str):
            key = value.strip()
            if not key:
                raise ValueError("Patient identifiers cannot be empty")
        elif isinstance(value, (bool, np.bool_)):
            raise ValueError("Boolean values are not patient identifiers")
        elif isinstance(value, (int, np.integer)):
            key = str(int(value))
        elif isinstance(value, (float, np.floating)) and np.isfinite(value) and value == int(value):
            key = str(int(value))
        else:
            raise ValueError("Patient identifiers must be nonempty strings or finite integers")
        groups[key].append(index)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def _sample_record_indices(groups, rng):
    """Return full record clusters and sampled patient positions, with repeats."""
    patient_draws = rng.integers(0, len(groups), size=len(groups))
    indices = np.concatenate([groups[index] for index in patient_draws])
    return indices, patient_draws


def patient_cluster_bootstrap(
    y, p, patient_ids, class_names, n_resamples=500, seed=2024,
):
    """Return point estimates and 95% patient-bootstrap percentile intervals.

    `metrics[name]` contains point_estimate, ci_lower, ci_upper, defined_resamples,
    undefined_resamples and interval_status. Intervals use NumPy linear quantiles
    at 0.025 and 0.975 on defined replicates only, and are null when fewer than
    two replicates are defined. Full-five-label ranking metrics retain the strict
    undefined-label policy of evaluate_multilabel; no partial-class replacement.
    """
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    n_resamples, seed = int(n_resamples), int(seed)
    truth, probabilities, names = np.asarray(y), np.asarray(p), list(class_names)
    point = evaluate_multilabel(truth, probabilities, names, threshold=0.5)
    groups = _patient_groups(patient_ids, len(truth))
    rng = np.random.default_rng(seed)
    metric_samples = {name: [] for name in METRIC_NAMES}
    record_counts = []
    for _ in range(n_resamples):
        indices, _patient_draws = _sample_record_indices(groups, rng)
        record_counts.append(len(indices))
        result = evaluate_multilabel(truth[indices], probabilities[indices], names, threshold=0.5)
        for name in METRIC_NAMES:
            value = result[name]
            if value is not None:
                if not np.isfinite(value):
                    raise ValueError(f"Metric {name} unexpectedly returned a nonfinite number")
                metric_samples[name].append(float(value))
    intervals = {}
    for name in METRIC_NAMES:
        values = metric_samples[name]
        if len(values) >= 2:
            lower, upper = np.quantile(values, [0.025, 0.975], method="linear")
            lower, upper, status = float(lower), float(upper), "defined"
        else:
            lower, upper, status = None, None, "insufficient_defined_resamples"
        intervals[name] = {
            "point_estimate": point[name], "ci_lower": lower, "ci_upper": upper,
            "defined_resamples": len(values), "undefined_resamples": n_resamples - len(values),
            "interval_status": status,
        }
    return {
        "method": "Patient-cluster nonparametric percentile bootstrap",
        "confidence_level": 0.95, "percentiles": [2.5, 97.5], "quantile_method": "linear",
        "n_records": len(truth), "n_unique_patients": len(groups),
        "patient_draws_per_resample": len(groups), "n_resamples": n_resamples,
        "seed": seed, "seed_role": "Uncertainty resampling only; independent local NumPy Generator",
        "class_names": names, "decision_threshold": 0.5,
        "metrics": intervals,
        "patient_record_count_distribution": {str(size): count for size, count in sorted(Counter(len(group) for group in groups).items())},
        "resampled_record_counts": {"minimum": min(record_counts), "maximum": max(record_counts),
                                    "mean": float(np.mean(record_counts))},
        "cluster_policy": "Draw the observed number of distinct patients with replacement; include every observed ECG for each draw, repeating all records when that patient is redrawn. Record count may vary when cluster sizes differ.",
        "patient_id_policy": "String identifiers have surrounding whitespace removed; numeric integer-valued identifiers become integer strings. Source identifiers must consistently refer to patients.",
        "undefined_policy": "Undefined replicates are counted and omitted separately per metric. Ranking metrics retain the full-five-label policy; no defined-label-only substitution. Intervals require at least two defined replicates.",
        "interpretation": "Sampling uncertainty conditional on the fitted model, fixed threshold, observed patient cohort and empirical cluster distribution. These intervals do not quantify training-seed variability or external-domain generalization.",
        "limitations": "Intervals based on defined replicates only are additionally conditional on metric estimability when resampling omits requisite labels. Small patient counts, class rarity and degenerate predictions can make percentile intervals unstable or degenerate. No multiplicity adjustment or bias correction is applied.",
        "threshold_fitting_performed": False, "calibration_performed": False, "model_training_performed": False,
    }
