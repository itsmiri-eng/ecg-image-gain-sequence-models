"""PTB-XL multilabel PCA + linear SVM baseline (protocol ptbxl_v2_svm).

Usage:
    python -X utf8 -m src.run_ptbxl_svm --seed 42

Same features and train-only PCA as src/run_ptbxl_lda.py (8x8 area-pooled cached ink images,
25,200 features). PCA scores are standardized with training statistics, then one linear SVM per
label (five independent binary classifiers, unweighted). Probabilities come from Platt sigmoid
calibration fitted inside three patient-grouped training folds, so validation and test are never
used to fit anything. (PCA components, C) are chosen from the grid in the config by validation macro
AUROC; the test partition is predicted once afterwards.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.common import ROOT, environment, read_json, sha256_file, utc_now, write_csv, write_json
from src.evaluation.multilabel import evaluate_multilabel, save_multilabel_evaluation
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.ptbxl_learning_data import CLASS_NAMES
from src.run_ptbxl_experiment import append_log_locked, check_unchanged, completed_run_hashes, labels, make_dataset
from src.run_ptbxl_lda import features

CONFIG = ROOT / "experiments/configs/ptbxl_v2_svm.json"
LOG = ROOT / "experiments/experiment_log.csv"


class PCASVM:
    """Train-only PCA projection and scaling, then five calibrated binary linear SVMs."""
    def __init__(self, pca, n_components, scaler, svms):
        self.pca, self.n_components, self.scaler, self.svms = pca, n_components, scaler, svms

    def predict_proba(self, x):
        z = self.scaler.transform(self.pca.transform(x)[:, :self.n_components])
        return np.column_stack([svm.predict_proba(z)[:, 1] for svm in self.svms]).astype(np.float64)


def fit_svms(z, y, c, folds):
    return [CalibratedClassifierCV(LinearSVC(C=c, dual="auto", max_iter=20000), method="sigmoid", cv=folds).fit(z, y[:, j])
            for j in range(y.shape[1])]


def run(seed, run_id=None, config=CONFIG):
    cfg = read_json(config)
    run_id = run_id or f"{cfg['protocol']}_seed{seed}"
    with LOG.open() as handle:
        if any(row["experiment_id"] == run_id for row in csv.DictReader(handle)):
            raise ValueError(f"Run ID {run_id} already logged")
    family = cfg["result_family"]
    out, model_dir = ROOT / "results" / family / run_id, ROOT / "models" / family / run_id
    if out.exists() or model_dir.exists():
        raise FileExistsError("Run already exists; never repeat its test inference")
    cache = ROOT / cfg["cache_root"]
    cache_hashes = {name: sha256_file(cache / name) for name in ("protocol.json", "metadata.json", "cache_manifest.csv")}
    if cache_hashes != cfg["cache_sha256"]:
        raise ValueError("Cached input dataset differs from the frozen protocol binding")
    protected = completed_run_hashes()
    original_log = LOG.read_bytes()
    out.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    started, timer = utc_now(), time.perf_counter()
    write_json(out / "config.json", {**cfg, "seed": seed, "run_id": run_id})
    write_json(out / "provenance.json", {"started_utc": started, "environment": environment(),
               "configuration_sha256": sha256_file(config), "cache_sha256": cache_hashes,
               "source_sha256": {name: sha256_file(ROOT / name) for name in
                                 ("src/run_ptbxl_svm.py", "src/run_ptbxl_lda.py", "src/run_ptbxl_experiment.py",
                                  "src/evaluation/multilabel.py", "src/evaluation/ptbxl_uncertainty.py", "src/common.py")}})
    train, validation = (make_dataset(cfg, split, normalize=False) for split in ("train", "validation"))
    x_train, x_val = features(train), features(validation)
    y_train, y_val = labels(train), labels(validation)
    folds = list(GroupKFold(n_splits=3).split(x_train, groups=[row["patient_id"] for row in train.rows]))
    fit_started = time.perf_counter()
    pca = PCA(n_components=max(cfg["pca_components_grid"]), svd_solver="randomized", random_state=seed).fit(x_train)
    z_train_all, z_val_all = pca.transform(x_train), pca.transform(x_val)
    search = []
    for k in cfg["pca_components_grid"]:
        scaler = StandardScaler().fit(z_train_all[:, :k])
        z_train, z_val = scaler.transform(z_train_all[:, :k]), scaler.transform(z_val_all[:, :k])
        for c in cfg["svm_c_grid"]:
            svms = fit_svms(z_train, y_train, c, folds)
            m = evaluate_multilabel(y_val, np.column_stack([s.predict_proba(z_val)[:, 1] for s in svms]), CLASS_NAMES)
            search.append({"pca_components": k, "C": c, "validation_macro_roc_auc": m["macro_roc_auc"],
                           "validation_macro_average_precision": m["macro_average_precision"],
                           "validation_binary_cross_entropy": m["binary_cross_entropy"],
                           "explained_variance": float(pca.explained_variance_ratio_[:k].sum())})
            print(f"{utc_now()} PCA {k} C {c}: validation macro AUROC {m['macro_roc_auc']:.5f}", flush=True)
    best = max(search, key=lambda r: (r["validation_macro_roc_auc"], r["validation_macro_average_precision"],
                                      -r["validation_binary_cross_entropy"], -r["pca_components"], -r["C"]))
    k, c = best["pca_components"], best["C"]
    scaler = StandardScaler().fit(z_train_all[:, :k])
    model = PCASVM(pca, k, scaler, fit_svms(scaler.transform(z_train_all[:, :k]), y_train, c, folds))
    fit_seconds = time.perf_counter() - fit_started
    write_csv(out / "validation_hyperparameter_search.csv", search)
    joblib.dump(model, model_dir / "pca_svm.joblib")
    checkpoint_hash = sha256_file(model_dir / "pca_svm.joblib")
    write_json(out / "selection_frozen.json", {"frozen_utc": utc_now(), "pca_components": k, "C": c,
               "checkpoint_sha256": checkpoint_hash, "test_inference_started": False})
    model = joblib.load(model_dir / "pca_svm.joblib")  # test predictions come from the frozen file
    metrics = {"train": save_multilabel_evaluation(out, "train", train.rows, y_train, model.predict_proba(x_train), CLASS_NAMES),
               "validation": save_multilabel_evaluation(out, "validation", validation.rows, y_val,
                                                        model.predict_proba(x_val), CLASS_NAMES)}
    if abs(metrics["validation"]["macro_roc_auc"] - best["validation_macro_roc_auc"]) > 1e-9:
        raise ValueError("Frozen model does not reproduce the selected validation result")
    del x_train, z_train_all
    check_unchanged(protected)
    test = make_dataset(cfg, "test", normalize=False)
    write_json(out / "test_inference_state.json", {"status": "started", "started_utc": utc_now(),
               "checkpoint_sha256": checkpoint_hash, "inference_calls": 1})
    probability = model.predict_proba(features(test))
    truth = labels(test)
    metrics["test"] = save_multilabel_evaluation(out, "test", test.rows, truth, probability, CLASS_NAMES)
    write_json(out / "test_inference_state.json", {"status": "completed", "completed_utc": utc_now(),
               "checkpoint_sha256": checkpoint_hash, "inference_calls": 1, "records": len(test)})
    write_json(out / "test_patient_bootstrap.json", patient_cluster_bootstrap(
        truth, probability, [row["patient_id"] for row in test.rows], CLASS_NAMES,
        n_resamples=cfg["uncertainty"]["resamples"], seed=cfg["uncertainty"]["seed"]))
    check_unchanged(protected)
    t = metrics["test"]
    append_log_locked({
        "experiment_id": run_id, "date": started, "model": cfg["log_model_name"], "status": "completed",
        "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
        "preprocessing": json.dumps({**cfg["preprocessing"], "svm_reduction": cfg["svm_reduction"]}, sort_keys=True),
        "random_seed": seed, "hyperparameters": json.dumps({"pca_components": k, "C": c, "pca_grid": cfg["pca_components_grid"],
                                                            "c_grid": cfg["svm_c_grid"], "svm": cfg["svm"]}, sort_keys=True),
        "accuracy": "", "balanced_accuracy": "", "macro_f1": t["macro_f1"], "weighted_f1": t["weighted_f1"],
        "notes": (f"Five-label PTB-XL protocol v2; test macro AUROC={t['macro_roc_auc']:.6f}; macro AP="
                  f"{t['macro_average_precision']:.6f}; subset_accuracy={t['subset_accuracy']:.6f}. "
                  "8x8 area-pooled inputs; PCA components and C chosen on validation."),
        "output_directory": out.relative_to(ROOT).as_posix()}, original_log)
    result = {"status": "completed", "experiment_id": run_id, "model": "pca_svm", "seed": seed, "started_utc": started,
              "completed_utc": utc_now(), "total_runtime_seconds": time.perf_counter() - timer,
              "fit_seconds": fit_seconds, "pca_components": k, "C": c, "hyperparameter_search": search,
              "num_parameters": int(k * x_val.shape[1] + x_val.shape[1] + 2 * k + 5 * (k + 1)),
              "parameter_count_definition": "PCA projection (k x 25,200) + PCA mean + scaler (2k) + five SVM weight vectors "
                                            "and intercepts (per calibration fold ensemble member: x3, not counted)",
              "metrics": metrics, "checkpoint_sha256": checkpoint_hash, "test_inference_calls": 1, "threshold": 0.5,
              "environment": environment()}
    write_json(out / "run.json", result)
    print(json.dumps({"run_id": run_id, "pca_components": k, "C": c, "test_macro_auc": t["macro_roc_auc"],
                      "test_macro_f1": t["macro_f1"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=CONFIG)
    arguments = parser.parse_args()
    run(arguments.seed, arguments.run_id, arguments.config)
