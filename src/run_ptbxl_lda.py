"""PTB-XL multilabel PCA + LDA baseline (protocol ptbxl_v2_lda).

Usage:
    python -m src.run_ptbxl_lda --seed 42

Inputs are the same cached 12x120x1120 ink images used by every PTB-XL network.
Because PCA on 1.6 million pixels per ECG is infeasible on 8 GB of memory, each
image is reduced deterministically by 8x8 area averaging to 12x15x140 = 25,200
features (a stated representation difference from the networks). PCA is fitted
on training images only. One shrinkage LDA per label (five independent binary
classifiers, empirical priors, matching the unweighted networks) gives the five
probabilities. The number of PCA components is chosen from a prespecified grid
by validation macro AUROC; the test partition is predicted once afterwards.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.baseline_preservation import append_experiment_log
from src.common import ROOT, environment, read_json, sha256_file, utc_now, write_csv, write_json
from src.evaluation.multilabel import evaluate_multilabel, save_multilabel_evaluation
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap
from src.ptbxl_learning_data import CachedECGDataset, CLASS_NAMES, INPUT_SHAPE
from src.run_ptbxl_experiment import append_log_locked, check_unchanged, completed_run_hashes, labels, make_dataset

CONFIG = ROOT / "experiments/configs/ptbxl_v2_lda.json"
FEATURES = ROOT / "data/processed/ptbxl_lda_features_v1"
POOL = 8
LOG = ROOT / "experiments/experiment_log.csv"


def _reduce(task):
    path, pixel_sha = task
    stack = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if stack is None or stack.shape != (1440, 1120) or hashlib.sha256(stack.tobytes()).hexdigest() != pixel_sha:
        raise ValueError(f"Cache image failed decoding or checksum: {path}")
    image = stack.reshape(INPUT_SHAPE).astype(np.float32) / 255
    leads, height, width = INPUT_SHAPE
    return image.reshape(leads, height // POOL, POOL, width // POOL, POOL).mean(axis=(2, 4)).ravel().astype(np.float16)


def features(dataset, workers=4, directory=FEATURES):
    """Area-pooled ink features in dataset order, cached on disk after first extraction."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dataset.split}.npy"
    ids_path = directory / f"{dataset.split}_ecg_ids.npy"
    ids = np.asarray([row["ecg_id"] for row in dataset.rows])
    if path.exists() and ids_path.exists() and np.array_equal(np.load(ids_path), ids):
        return np.load(path).astype(np.float32)
    tasks = [(dataset.cache_root / row["cache_path"], row["cache_pixel_sha256"]) for row in dataset.rows]
    started = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        x = np.stack(list(pool.map(_reduce, tasks, chunksize=64)))
    np.save(path, x)
    np.save(ids_path, ids)
    print(f"{utc_now()} extracted {dataset.split} features {x.shape} in {time.perf_counter() - started:.1f}s", flush=True)
    return x.astype(np.float32)


class PCALDA:
    """Train-only PCA projection, then five independent binary shrinkage LDAs."""
    def __init__(self, pca, n_components, ldas):
        self.pca, self.n_components, self.ldas = pca, n_components, ldas

    def predict_proba(self, x):
        z = self.pca.transform(x)[:, :self.n_components]
        return np.column_stack([lda.predict_proba(z)[:, 1] for lda in self.ldas]).astype(np.float64)


def fit_ldas(z, y):
    return [LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(z, y[:, j]) for j in range(y.shape[1])]


def run(seed, run_id=None, config=CONFIG):
    cfg = read_json(config)
    run_id = run_id or f"{cfg['protocol']}_seed{seed}"
    with LOG.open() as handle:
        if any(row["experiment_id"] == run_id for row in csv.DictReader(handle)):
            raise ValueError(f"Run ID {run_id} already logged")
    family = cfg.get("result_family", "ptbxl_v2")
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
    feature_dir = ROOT / "data/processed" / cfg.get("lda_features_dir", "ptbxl_lda_features_v1")
    train, validation = (make_dataset(cfg, split, normalize=False) for split in ("train", "validation"))
    x_train, x_val = features(train, directory=feature_dir), features(validation, directory=feature_dir)
    y_train, y_val = labels(train), labels(validation)
    fit_started = time.perf_counter()
    grid = cfg["pca_components_grid"]
    pca = PCA(n_components=max(grid), svd_solver="randomized", random_state=seed).fit(x_train)
    z_train, z_val = pca.transform(x_train), pca.transform(x_val)
    search = []
    for k in grid:
        ldas = fit_ldas(z_train[:, :k], y_train)
        p_val = np.column_stack([lda.predict_proba(z_val[:, :k])[:, 1] for lda in ldas])
        m = evaluate_multilabel(y_val, p_val, CLASS_NAMES)
        search.append({"pca_components": k, "validation_macro_roc_auc": m["macro_roc_auc"],
                       "validation_macro_average_precision": m["macro_average_precision"],
                       "validation_binary_cross_entropy": m["binary_cross_entropy"],
                       "explained_variance": float(pca.explained_variance_ratio_[:k].sum())})
        print(f"{utc_now()} PCA {k}: validation macro AUROC {m['macro_roc_auc']:.5f}", flush=True)
    best = max(search, key=lambda r: (r["validation_macro_roc_auc"], r["validation_macro_average_precision"],
                                      -r["validation_binary_cross_entropy"], -r["pca_components"]))
    k = best["pca_components"]
    model = PCALDA(pca, k, fit_ldas(z_train[:, :k], y_train))
    fit_seconds = time.perf_counter() - fit_started
    write_csv(out / "validation_component_search.csv", search)
    np.savez(model_dir / "pca_lda.npz", pca_mean=pca.mean_, pca_components=pca.components_[:k],
             lda_coef=np.stack([lda.coef_[0] for lda in model.ldas]),
             lda_intercept=np.asarray([lda.intercept_[0] for lda in model.ldas]),
             lda_priors=np.stack([lda.priors_ for lda in model.ldas]))
    checkpoint_hash = sha256_file(model_dir / "pca_lda.npz")
    write_json(out / "selection_frozen.json", {"frozen_utc": utc_now(), "pca_components": k,
               "checkpoint_sha256": checkpoint_hash, "test_inference_started": False})
    metrics = {"train": save_multilabel_evaluation(out, "train", train.rows, y_train, model.predict_proba(x_train), CLASS_NAMES),
               "validation": save_multilabel_evaluation(out, "validation", validation.rows, y_val,
                                                        model.predict_proba(x_val), CLASS_NAMES)}
    del x_train, z_train
    check_unchanged(protected)
    test = make_dataset(cfg, "test", normalize=False)
    write_json(out / "test_inference_state.json", {"status": "started", "inference_calls": 1})
    probability = model.predict_proba(features(test, directory=feature_dir))
    truth = labels(test)
    metrics["test"] = save_multilabel_evaluation(out, "test", test.rows, truth, probability, CLASS_NAMES)
    write_json(out / "test_inference_state.json", {"status": "completed", "inference_calls": 1, "records": len(test)})
    write_json(out / "test_patient_bootstrap.json", patient_cluster_bootstrap(
        truth, probability, [row["patient_id"] for row in test.rows], CLASS_NAMES,
        n_resamples=cfg["uncertainty"]["resamples"], seed=cfg["uncertainty"]["seed"]))
    check_unchanged(protected)
    t = metrics["test"]
    append_log_locked({
        "experiment_id": run_id, "date": started, "model": cfg.get("log_model_name", "PTBXL_Image_PCA_LDA"), "status": "completed",
        "dataset_version": cfg["dataset_id"], "split_version": cfg["split_version"],
        "preprocessing": json.dumps({**cfg["preprocessing"], "lda_reduction": cfg["lda_reduction"]}, sort_keys=True),
        "random_seed": seed, "hyperparameters": json.dumps({"pca_components": k, "grid": grid, "lda": cfg["lda"]}, sort_keys=True),
        "accuracy": "", "balanced_accuracy": "", "macro_f1": t["macro_f1"], "weighted_f1": t["weighted_f1"],
        "notes": (f"Five-label PTB-XL protocol v2; test macro AUROC={t['macro_roc_auc']:.6f}; macro AP="
                  f"{t['macro_average_precision']:.6f}; subset_accuracy={t['subset_accuracy']:.6f}. "
                  "8x8 area-pooled inputs; PCA components chosen on validation."),
        "output_directory": str(out.relative_to(ROOT))}, original_log)
    result = {"status": "completed", "experiment_id": run_id, "model": "pca_lda", "seed": seed, "started_utc": started,
              "completed_utc": utc_now(), "total_runtime_seconds": time.perf_counter() - timer,
              "fit_seconds": fit_seconds, "pca_components": k, "component_search": search,
              "num_parameters": int(k * x_val.shape[1] + x_val.shape[1] + 5 * (k + 1)),
              "parameter_count_definition": "PCA projection (k x 25,200) + PCA mean + five LDA weight vectors and intercepts",
              "metrics": metrics, "checkpoint_sha256": checkpoint_hash, "test_inference_calls": 1, "threshold": 0.5,
              "environment": environment()}
    write_json(out / "run.json", result)
    print(json.dumps({"run_id": run_id, "pca_components": k, "test_macro_auc": t["macro_roc_auc"],
                      "test_macro_f1": t["macro_f1"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=CONFIG)
    arguments = parser.parse_args()
    run(arguments.seed, arguments.run_id, arguments.config)
