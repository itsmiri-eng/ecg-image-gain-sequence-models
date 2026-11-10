import hashlib

import cv2
import numpy as np
import pytest
from scipy.io import savemat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

from src import external_test as ext
from src.run_ptbxl_lda import _reduce

NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def header_text(names=NAMES, fs=500, samples=5000, gain="1000.0(0)/mv", dx="164873001,426783006"):
    lines = [f"E00001 {len(names)} {fs} {samples}"]
    lines += [f"E00001.mat 16+24 {gain} 16 0 12 345 0 {name}" for name in names]
    lines += ["# Age: 60", "# Sex: Male", f"# Dx: {dx}", "# Rx: Unknown"]
    return "\n".join(lines) + "\n"


def test_parse_header_gain_baseline_and_dx():
    h = ext.parse_header(header_text(gain="500(7)/mV"))
    assert h["fs"] == 500 and h["n_samples"] == 5000 and h["dx"] == ["164873001", "426783006"]
    assert h["leads"][0]["gain"] == 500 and h["leads"][0]["baseline"] == 7 and h["leads"][3]["name"] == "aVR"
    assert ext.parse_header(header_text(gain="1000/mV"))["leads"][0]["baseline"] == 0  # falls back to ADC zero


def test_load_signal_reorders_leads_and_scales(tmp_path):
    order = (6, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
    names = tuple(NAMES[i] for i in order)
    digital = np.stack([np.full(5000, 1000 * (i + 1), np.int16) for i in order])
    savemat(tmp_path / "E00001.mat", {"val": digital}, format="4")
    (tmp_path / "E00001.hea").write_text(header_text(names=names))
    h = ext.parse_header((tmp_path / "E00001.hea").read_text())
    signal = ext.load_signal(tmp_path / "E00001.hea", h)
    assert signal.shape == (5000, 12)
    np.testing.assert_allclose(signal[0], np.arange(1, 13))  # renderer order I..V6, in mV


def test_inclusion_steps():
    mapping = {"164873001": {"HYP"}, "164865005": {"MI"}}
    assert ext.inclusion(ext.parse_header(header_text()), mapping) == ("included", {"HYP"})
    assert ext.inclusion(ext.parse_header(header_text(names=NAMES[:11])), mapping)[0] == "not_12_standard_leads"
    assert ext.inclusion(ext.parse_header(header_text(fs=250)), mapping)[0] == "not_500_hz"
    assert ext.inclusion(ext.parse_header(header_text(samples=2500)), mapping)[0] == "shorter_than_10_s"
    assert ext.inclusion(ext.parse_header(header_text(dx="426783006")), mapping)[0] == "no_mapped_superclass"


def test_lda_rebuild_matches_sklearn(tmp_path):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 6)), rng.integers(0, 2, (200, 5))
    mean, components = x.mean(axis=0), np.linalg.qr(rng.normal(size=(6, 4)))[0].T
    z = (x - mean) @ components.T
    ldas = [LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(z, y[:, j]) for j in range(5)]
    np.savez(tmp_path / "pca_lda.npz", pca_mean=mean, pca_components=components,
             lda_coef=np.stack([l.coef_[0] for l in ldas]), lda_intercept=np.asarray([l.intercept_[0] for l in ldas]),
             lda_priors=np.stack([l.priors_ for l in ldas]))
    expected = np.column_stack([l.predict_proba(z)[:, 1] for l in ldas])
    np.testing.assert_allclose(ext.LDAModel(tmp_path / "pca_lda.npz").predict_proba(x), expected, rtol=1e-10, atol=1e-12)


def test_pooled_features_match_lda_extraction(tmp_path):
    stack = np.random.default_rng(1).integers(0, 256, (1440, 1120), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "page.png"), stack)
    expected = _reduce((tmp_path / "page.png", hashlib.sha256(stack.tobytes()).hexdigest())).astype(np.float32)
    np.testing.assert_array_equal(ext.pooled_features(stack[None])[0], expected)


def test_auroc_and_holm():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 2, 300), rng.random(300)
    assert ext.auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert ext.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_hierarchical_difference_of_identical_models_is_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, (400, 5))
    a = {s: rng.random((400, 5)) for s in (42, 43, 44)}
    result = ext.hierarchical_difference(y, a, a, lambda yy, pp: ext.macro_auroc(yy, pp, ext.CLASS_NAMES), [42, 43, 44],
                                         resamples=50)
    assert result["difference"] == 0 and result["ci_lower"] == 0 and result["p_bootstrap"] == 1.0


def test_hierarchical_difference_with_patient_clusters_detects_a_better_model():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, (300, 5))
    noise = {s: rng.random((300, 5)) for s in (42, 43)}
    better = {s: 0.5 * y + 0.5 * noise[s] for s in (42, 43)}
    clusters = [np.arange(i, i + 3) for i in range(0, 300, 3)]
    result = ext.hierarchical_difference(y, noise, better, lambda yy, pp: ext.macro_auroc(yy, pp, ext.CLASS_NAMES),
                                         [42, 43], resamples=50, clusters=clusters)
    assert result["difference"] > 0.3 and result["ci_lower"] > 0 and result["p_bootstrap"] < 0.05
