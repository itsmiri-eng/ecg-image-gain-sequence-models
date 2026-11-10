"""Calibration, threshold and integrated-gradients helpers behave as the frozen protocol says."""
import numpy as np
import torch

from src.analyze_calibration_thresholds import choose_threshold, ece, f1_sens_spec
from src.analyze_pulse_swap import apply, cluster_bootstrap
from src.analyze_saliency import STEPS, column_attributions


def test_ece_is_zero_when_bins_match_prevalence_and_measures_the_gap_otherwise():
    p = np.array([0.15] * 20 + [0.85] * 20)
    y = np.array([1] * 3 + [0] * 17 + [1] * 17 + [0] * 3)
    assert ece(y, p) < 1e-12
    assert abs(ece(np.zeros(10), np.full(10, 0.3)) - 0.3) < 1e-12


def test_f1_sensitivity_specificity():
    f1, sens, spec = f1_sens_spec(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (f1, sens, spec) == (0.5, 0.5, 0.5)


def test_threshold_maximizes_validation_f1_and_breaks_ties_toward_half():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])
    assert choose_threshold(y, p) == 0.3  # 0.21-0.30 all give F1 = 1; 0.30 is closest to 0.5
    assert choose_threshold(np.array([0, 1]), np.array([0.6, 0.6])) == 0.5  # every grid value up to 0.6 ties


def test_pulse_swap_changes_only_the_pulse_columns():
    x = torch.rand(2, 12, 120, 1120)
    before = x.clone()
    blocks ={g: torch.full((12, 120, 40), g) for g in (10.0, 5.0, 2.5)}
    sham = torch.full((2, 12, 120, 40), 7.0)
    assert apply(x, "original", blocks, sham) is x
    for condition, value in (("pulse_10", 10.0), ("pulse_2.5", 2.5), ("sham", 7.0), ("mask", 0.0)):
        y = apply(x, condition, blocks, sham)
        assert torch.all(y[..., 48:88] == value)
        assert torch.equal(y[..., :48], x[..., :48]) and torch.equal(y[..., 88:], x[..., 88:])
    assert torch.equal(x, before)  # the input page is never modified in place


def test_cluster_bootstrap_mean_and_interval():
    mean, low, high = cluster_bootstrap([1.0, 1.0, 3.0], ["a", "a", "b"])
    assert abs(mean - 5 / 3) < 1e-12 and 1.0 <= low <= mean <= high <= 3.0


def test_integrated_gradients_of_a_linear_model_are_exact():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12 * 120 * 1120, 5)).eval()
    x, baseline = torch.rand(1, 12, 120, 1120), torch.zeros(1, 12, 120, 1120)
    columns = column_attributions(model, x, baseline, 4, "cpu")
    exact = (model[1].weight[4].view(12, 120, 1120) * x[0]).abs().sum(dim=(0, 1)).detach().numpy()
    assert columns.shape == (1120,) and STEPS == 16
    assert np.allclose(columns, exact, rtol=1e-4, atol=1e-6)
