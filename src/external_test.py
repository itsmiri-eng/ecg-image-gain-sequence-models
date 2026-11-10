"""External confirmatory test (frozen plan: docs/external_test_plan.md, experiments/external/plan_freeze.json).

Usage (each step refuses to run before the previous one; run under the CPU cap):
    python -X utf8 -m src.external_test inventory --dataset georgia
    python -X utf8 -m src.external_test render --dataset georgia --workers 3
    python -X utf8 -m src.external_test infer --dataset georgia [--device cuda]
    python -X utf8 -m src.external_test analyze --dataset georgia
Datasets: georgia (primary; training/georgia) and china (secondary; training/chapman_shaoxing + training/ningbo).

- inventory: parses the Challenge headers, applies the frozen inclusion steps and label mapping, counts every
  step and decides which labels are eligible (>= 50 positives and >= 50 negatives).
- render: for each included record, draws the first 10 s with the unchanged v1 renderer (adaptive gain) and the
  unchanged fixed-gain renderer, applies the unchanged preprocess_page, and stores lossless PNG stacks plus the
  100 Hz signal (resample_poly 1/5) for the signal reference.
- infer: first checks that every model, as reloaded here, reproduces its saved PTB-XL validation probabilities;
  then decodes each external page once and runs every completed model on it. One inference per run and
  dataset, guarded by inference_state.json; never repeated.
- analyze: the frozen statistics (H1 head pairs, H2 fixed vs adaptive gain, H3 external minus PTB-XL test).
Writes data/external/cache/<dataset>/, experiments/external/<dataset>/ and results/external/<dataset>/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import re

import cv2
import numpy as np

from src.common import ROOT, read_json, sha256_file, utc_now, write_csv, write_json

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
CHALLENGE = ROOT / "data/external/challenge-2021/training"
DATASETS = {"georgia": ("georgia",), "china": ("chapman_shaoxing", "ningbo")}
SAMPLES, FS = 5000, 500
MIN_POSITIVES = MIN_NEGATIVES = 50
RESAMPLES, BOOT_SEED = 2000, 2024
HEADS = ("cnn_pool", "cnn_pool_wide", "cnn_lstm", "temporal_conv", "hybrid_vit")
V1_FAMILIES = {"ptbxl_cnn_lstm": "cnn_lstm", "ptbxl_spatiotemporal": "temporal_conv", "ptbxl_hybrid_vit": "hybrid_vit"}
RUN_FAMILIES = ("ptbxl_v2", "ptbxl_v2_fixedgain", "ptbxl_signal", *V1_FAMILIES)


def experiment_dir(dataset):
    return ROOT / "experiments/external" / dataset


def cache_dir(dataset):
    return ROOT / "data/external/cache" / dataset


def result_dir(dataset):
    return ROOT / "results/external" / dataset


# ---------------------------------------------------------------- reading Challenge records

def parse_header(text):
    """WFDB header of one Challenge record -> dict (fs, samples, per-lead gain/baseline/units, Dx codes, mat file)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = lines[0].split()
    n_signals, fs, n_samples = int(first[1]), float(first[2].split("/")[0]), int(first[3])
    leads = []
    for line in lines[1:1 + n_signals]:
        parts = line.split()
        match = re.fullmatch(r"([-+\d.eE]+)(?:\((-?\d+)\))?(?:/(\S+))?", parts[2])
        if match is None:
            raise ValueError(f"Unreadable gain field: {parts[2]}")
        gain = float(match.group(1)) or 200.0  # WFDB: gain 0 means the default 200
        baseline = int(match.group(2)) if match.group(2) is not None else int(parts[4])
        leads.append({"file": parts[0], "format": parts[1], "gain": gain, "baseline": baseline,
                      "units": (match.group(3) or "mV"), "name": parts[-1]})
    dx = []
    for line in lines[1 + n_signals:]:
        found = re.match(r"#\s*Dx:\s*(.*)$", line)
        if found:
            dx = [code.strip() for code in found.group(1).split(",") if code.strip()]
    return {"n_signals": n_signals, "fs": fs, "n_samples": n_samples, "leads": leads, "dx": dx}


def load_signal(hea_path, header):
    """(samples, 12) float64 millivolts in the renderer's lead order."""
    from scipy.io import loadmat
    files = {lead["file"] for lead in header["leads"]}
    if len(files) != 1:
        raise ValueError("Expected one signal file per record")
    digital = np.asarray(loadmat(str(Path(hea_path).with_name(files.pop())))["val"], dtype=np.float64)
    if digital.shape != (header["n_signals"], header["n_samples"]):
        raise ValueError(f"Signal shape {digital.shape} differs from header")
    names = [lead["name"].upper() for lead in header["leads"]]
    physical = np.stack([(digital[i] - lead["baseline"]) / lead["gain"] for i, lead in enumerate(header["leads"])], axis=1)
    return physical[:, [names.index(name) for name in LEADS]]


def inclusion(header, mapping):
    """(status, labels): status 'included' or the first failed frozen step."""
    names = [lead["name"].upper() for lead in header["leads"]]
    if header["n_signals"] != 12 or sorted(names) != sorted(LEADS):
        return "not_12_standard_leads", None
    if header["fs"] != FS:
        return "not_500_hz", None
    if any(lead["units"].lower() != "mv" for lead in header["leads"]):
        return "not_millivolt", None
    if header["n_samples"] < SAMPLES:
        return "shorter_than_10_s", None
    labels = {c for code in header["dx"] for c in mapping.get(code, ())}
    if not labels:
        return "no_mapped_superclass", None
    return "included", labels


def read_mapping():
    with (ROOT / "experiments/external/label_mapping.csv").open(newline="") as handle:
        return {r["snomed_code"]: set(filter(None, r["assigned_superclasses"].split("|"))) for r in csv.DictReader(handle)}


def check_freeze():
    """Frozen files must match the freeze, or the last of a chain of dated amendments (amendments.json)."""
    freeze = read_json(ROOT / "experiments/external/plan_freeze.json")
    path = ROOT / "experiments/external/amendments.json"
    amendments = read_json(path) if path.exists() else []
    for name, digest in freeze["sha256"].items():
        for amendment in amendments:
            if amendment["file"] == name and amendment["previous_sha256"] == digest:
                digest = amendment["new_sha256"]
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Frozen file changed since the plan freeze without a recorded amendment: {name}")
    return freeze


def inventory(dataset):
    check_freeze()
    out = experiment_dir(dataset)
    if (out / "inclusion.csv").exists():
        raise FileExistsError("Inventory already made; it is not repeated")
    mapping, rows = read_mapping(), []
    for source in DATASETS[dataset]:
        for path in sorted((CHALLENGE / source).rglob("*.hea")):
            header = parse_header(path.read_text(encoding="utf-8", errors="replace"))
            status, labels = inclusion(header, mapping)
            if not path.with_name(header["leads"][0]["file"]).is_file():
                # Amendment 6 Nov 2026: the signal file never passed PhysioNet's published SHA-256 (not kept).
                status, labels = "signal_file_failed_checksum", None
            rows.append({"record": f"{source}/{path.stem}", "source": source,
                         "header": path.relative_to(ROOT).as_posix(), "status": status,
                         "fs": header["fs"], "samples": header["n_samples"], "dx": "|".join(header["dx"]),
                         **{f"label_{c}": int(labels is not None and c in labels) for c in CLASS_NAMES}})
    included = [r for r in rows if r["status"] == "included"]
    counts = {"records": len(rows), **{s: sum(r["status"] == s for r in rows) for s in
              ("signal_file_failed_checksum", "not_12_standard_leads", "not_500_hz", "not_millivolt", "shorter_than_10_s",
               "no_mapped_superclass", "included")}}
    positives = {c: sum(r[f"label_{c}"] for r in included) for c in CLASS_NAMES}
    eligible = [c for c in CLASS_NAMES if positives[c] >= MIN_POSITIVES and len(included) - positives[c] >= MIN_NEGATIVES]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "inclusion.csv", rows)
    summary = {"created_utc": utc_now(), "dataset": dataset, "sources": list(DATASETS[dataset]), "counts": counts,
               "positives": positives, "eligible_labels": eligible,
               "excluded_labels": [c for c in CLASS_NAMES if c not in eligible], "plan_freeze_checked": True}
    write_json(out / "inventory.json", summary)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- rendering

def _render_job(task):
    from src import ptbxl_fixed_gain, ptbxl_rendering
    from src.ptbxl_learning_data import preprocess_page
    from scipy.signal import resample_poly
    row, config, out = task
    cv2.setNumThreads(1)
    hea = ROOT / row["header"]
    header = parse_header(hea.read_text(encoding="utf-8", errors="replace"))
    signal = load_signal(hea, header)[:SAMPLES]
    result = {"record": row["record"], "finite": bool(np.isfinite(signal).all())}
    if not result["finite"]:
        return result
    name = row["record"].replace("/", "__")
    for kind, render in (("adaptive", ptbxl_rendering.render_page), ("fixed", ptbxl_fixed_gain.render_fixed_gain)):
        page, meta = render(signal, config)
        stack = preprocess_page(page).reshape(1440, 1120)
        ok, encoded = cv2.imencode(".png", stack, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        destination = Path(out) / kind / f"{name}.png"
        with destination.open("xb") as handle:
            handle.write(encoded.tobytes())
        result[f"{kind}_pixel_sha256"] = hashlib.sha256(stack.tobytes()).hexdigest()
        result[f"{kind}_gain_mm_per_mv"] = meta["gain_mm_per_mv"]
        if kind == "fixed":
            result["fixed_clipped_points"] = meta["clipped_point_count"]
    result["signal100"] = resample_poly(signal, 1, 5, axis=0).T.astype(np.float32)  # (12, 1000)
    return result


def render(dataset, workers=3):
    check_freeze()
    out, exp = cache_dir(dataset), experiment_dir(dataset)
    if (out / "manifest.csv").exists():
        raise FileExistsError("Cache already built; never rebuilt in place")
    with (exp / "inclusion.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "included"]
    for kind in ("adaptive", "fixed"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / "experiments/configs/ptbxl_images_v1.json")
    results = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(_render_job, [(r, config, str(out)) for r in rows], chunksize=16):
            results.append(result)
            if len(results) % 2000 == 0:
                print(f"{utc_now()} rendered {len(results)}/{len(rows)}", flush=True)
    kept = [r for r in results if r["finite"]]
    np.save(out / "signals100.npy", np.stack([r.pop("signal100") for r in kept]))
    by_record = {r["record"]: r for r in kept}
    manifest = [{**{k: row[k] for k in ("record", "source", *[f"label_{c}" for c in CLASS_NAMES])},
                 **{k: v for k, v in by_record[row["record"]].items() if k not in ("record", "finite")}}
                for row in rows if row["record"] in by_record]
    write_csv(out / "manifest.csv", manifest)
    write_json(out / "metadata.json", {"created_utc": utc_now(), "dataset": dataset, "records": len(manifest),
               "excluded_nonfinite": len(results) - len(kept), "manifest_sha256": sha256_file(out / "manifest.csv"),
               "signals100_sha256": sha256_file(out / "signals100.npy"),
               "code_sha256": {n: sha256_file(ROOT / n) for n in ("src/external_test.py", "src/ptbxl_rendering.py",
                                                                 "src/ptbxl_fixed_gain.py", "src/ptbxl_learning_data.py")},
               "render_config_sha256": sha256_file(ROOT / "experiments/configs/ptbxl_images_v1.json")})
    print(f"{utc_now()} cache complete: {len(manifest)} records, {len(results) - len(kept)} excluded as non-finite")


# ---------------------------------------------------------------- models

class LDAModel:
    """PCA + five binary LDAs rebuilt from pca_lda.npz (binary LDA probability = sigmoid of the decision)."""
    def __init__(self, path):
        z = np.load(path)
        self.mean, self.components = z["pca_mean"], z["pca_components"]
        self.coef, self.intercept = z["lda_coef"], z["lda_intercept"]

    def predict_proba(self, features):
        scores = (features - self.mean) @ self.components.T @ self.coef.T + self.intercept
        return 1 / (1 + np.exp(-scores))


def pooled_features(stacks):
    """The PCA baselines' input: 8x8 area-pooled ink of (N, 1440, 1120) uint8 stacks, as in src/run_ptbxl_lda.py."""
    from src.run_ptbxl_lda import POOL
    image = stacks.reshape(len(stacks), 12, 120, 1120).astype(np.float32) / 255
    pooled = image.reshape(len(stacks), 12, 120 // POOL, POOL, 1120 // POOL, POOL).mean(axis=(3, 5))
    return pooled.reshape(len(stacks), -1).astype(np.float16).astype(np.float32)


def completed_runs():
    """[(run_id, family, kind, page, head, seed)] for every completed PTB-XL run."""
    runs = []
    for family in RUN_FAMILIES:
        for path in sorted((ROOT / "results" / family).glob("*/run.json")):
            run_id, run = path.parent.name, read_json(path)
            model_dir = ROOT / "models" / family / run_id
            page = "fixed" if family == "ptbxl_v2_fixedgain" else "adaptive"
            seed = int(run.get("seed") or 42)
            if (model_dir / "pca_lda.npz").exists():
                runs.append((run_id, family, "lda", page, "pca_lda", seed))
            elif (model_dir / "pca_svm.joblib").exists():
                runs.append((run_id, family, "svm", page, "pca_svm", seed))
            elif family == "ptbxl_signal":
                runs.append((run_id, family, "signal", "signal", "resnet1d_wang", seed))
            else:
                head = V1_FAMILIES.get(family) or read_json(model_dir / "metadata.json")["model"]
                runs.append((run_id, family, "network", page, head, seed))
    return runs


def load_models(runs, device):
    import joblib
    import torch
    from src.models.ptbxl_cnn_pool_wide import LeadImageCNNPoolWide
    from src.models.ptbxl_resnet1d import ResNet1dWang
    from src.training.ptbxl_trainer import MODELS
    table = {**MODELS, "cnn_pool_wide": LeadImageCNNPoolWide, "resnet1d_wang": ResNet1dWang}
    models = {}
    for run_id, family, kind, page, head, _ in runs:
        model_dir = ROOT / "models" / family / run_id
        if kind == "lda":
            models[run_id] = (LDAModel(model_dir / "pca_lda.npz"), None)
        elif kind == "svm":
            # Saved by `python -m src.run_ptbxl_svm`, so the pickle names the class __main__.PCASVM.
            import sys
            from src.run_ptbxl_svm import PCASVM
            main = sys.modules["__main__"]
            had = hasattr(main, "PCASVM")
            main.PCASVM = PCASVM
            try:
                models[run_id] = (joblib.load(model_dir / "pca_svm.joblib"), None)
            finally:
                if not had:
                    del main.PCASVM
        else:
            checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
            model = table[head]()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            models[run_id] = (model.to(device).eval(), checkpoint["normalization"])
    return models


def predict(models, runs, stacks, signals, device):
    """Probabilities of every run for one batch: stacks {page: (N,1440,1120) uint8}, signals (N,12,1000)."""
    import torch
    out, features = {}, {}
    with torch.inference_mode():
        for run_id, _, kind, page, _, _ in runs:
            model, normalization = models[run_id]
            if kind in ("lda", "svm"):
                if page not in features:
                    features[page] = pooled_features(stacks[page])
                out[run_id] = np.asarray(model.predict_proba(features[page]), dtype=np.float64)
            elif kind == "signal":
                mean = np.asarray(normalization["mean"], np.float32)[:, None]
                std = np.asarray(normalization["std"], np.float32)[:, None]
                x = torch.from_numpy((signals - mean) / std).to(device)
                out[run_id] = torch.sigmoid(model(x)).double().cpu().numpy()
            else:
                x = stacks[page].reshape(len(stacks[page]), 12, 120, 1120).astype(np.float32) / 255
                x = torch.from_numpy((x - normalization["mean"]) / normalization["std"]).to(device)
                out[run_id] = torch.sigmoid(model(x)).double().cpu().numpy()
    return out


def _decode(path, digest):
    stack = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if stack is None or stack.shape != (1440, 1120) or hashlib.sha256(stack.tobytes()).hexdigest() != digest:
        raise ValueError(f"Cached page failed decoding or checksum: {path}")
    return stack


def reproduction_check(runs, models, device):
    """Each reloaded model must reproduce its saved PTB-XL validation probabilities (no external data used)."""
    from src.ptbxl_signals import SignalDataset
    caches = {"adaptive": ROOT / "data/processed/ptbxl_lead_images_v1",
              "fixed": ROOT / "data/processed/ptbxl_lead_images_fixedgain2p5_v1"}
    rows = {}
    for page, root in caches.items():
        with (root / "cache_manifest.csv").open(newline="") as handle:
            rows[page] = [r for r in csv.DictReader(handle) if r["split"] == "validation"]
    signal_set = SignalDataset("validation", normalize=False)
    signals = np.stack([signal_set[i][0].numpy() for i in range(len(signal_set))])
    ids = [r["ecg_id"] for r in rows["adaptive"]]
    if [r["ecg_id"] for r in rows["fixed"]] != ids or [str(r["ecg_id"]) for r in signal_set.rows] != ids:
        raise ValueError("PTB-XL validation order differs between caches")
    parts = {run_id: [] for run_id, *_ in runs}
    for start in range(0, len(ids), 64):
        stacks = {page: np.stack([_decode(caches[page] / r["cache_path"], r["cache_pixel_sha256"])
                                  for r in rows[page][start:start + 64]]) for page in caches}
        for run_id, p in predict(models, runs, stacks, signals[start:start + 64], device).items():
            parts[run_id].append(p)
    report = {}
    for run_id, family, *_ in runs:
        with (ROOT / "results" / family / run_id / "validation_predictions.csv").open(newline="") as handle:
            saved = [r for r in csv.DictReader(handle)]
        if [r["ecg_id"] for r in saved] != ids:
            raise ValueError(f"{run_id}: saved validation order differs")
        expected = np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in saved])
        report[run_id] = float(np.abs(np.concatenate(parts[run_id]) - expected).max())
    return report


def infer(dataset, device="cuda", tolerance=1e-3):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    check_freeze()
    cache = cache_dir(dataset)
    metadata = read_json(cache / "metadata.json")
    if sha256_file(cache / "manifest.csv") != metadata["manifest_sha256"] or \
            sha256_file(cache / "signals100.npy") != metadata["signals100_sha256"]:
        raise ValueError("External cache changed since it was built")
    runs = completed_runs()
    out = result_dir(dataset)
    done = [r[0] for r in runs if (out / r[0] / "inference_state.json").exists()]
    if done:
        raise FileExistsError(f"External inference already started for {done[:5]}; never repeated")
    models = load_models(runs, torch.device(device))
    check = reproduction_check(runs, models, torch.device(device))
    write_json(experiment_dir(dataset) / "reproduction_check.json", {"checked_utc": utc_now(), "tolerance": tolerance,
                                                                       "max_abs_difference": check})
    failed = {k: v for k, v in check.items() if v > tolerance}
    if failed:
        raise ValueError(f"Reloaded models do not reproduce PTB-XL validation probabilities: {failed}")
    with (cache / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    signals = np.load(cache / "signals100.npy")
    for run_id, *_ in runs:
        (out / run_id).mkdir(parents=True)
        write_json(out / run_id / "inference_state.json", {"status": "started", "started_utc": utc_now(),
                                                          "dataset": dataset, "inference_calls": 1})
    parts = {run_id: [] for run_id, *_ in runs}
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        name = [r["record"].replace("/", "__") for r in batch]
        stacks = {page: np.stack([_decode(cache / page / f"{n}.png", r[f"{page}_pixel_sha256"]) for n, r in zip(name, batch)])
                  for page in ("adaptive", "fixed")}
        for run_id, p in predict(models, runs, stacks, signals[start:start + 64], torch.device(device)).items():
            parts[run_id].append(p)
        if start % 6400 == 0:
            print(f"{utc_now()} inference {start}/{len(rows)}", flush=True)
    for run_id, family, kind, page, head, seed in runs:
        p = np.concatenate(parts[run_id])
        write_csv(out / run_id / "predictions.csv", [{"record": r["record"], **{f"truth_{c}": r[f"label_{c}"] for c in CLASS_NAMES},
                  **{f"probability_{c}": float(p[i, j]) for j, c in enumerate(CLASS_NAMES)}} for i, r in enumerate(rows)])
        write_json(out / run_id / "inference_state.json", {"status": "completed", "completed_utc": utc_now(), "dataset": dataset,
                   "inference_calls": 1, "records": len(rows), "family": family, "kind": kind, "page": page,
                   "head": head, "seed": seed, "predictions_sha256": sha256_file(out / run_id / "predictions.csv")})
    print(f"{utc_now()} external inference complete: {len(runs)} runs x {len(rows)} records")


# ---------------------------------------------------------------- statistics

def auroc(y, p):
    """Rank-based ROC AUC (ties averaged); NaN if only one class present."""
    from scipy.stats import rankdata
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def macro_auroc(y, p, labels):
    return float(np.mean([auroc(y[:, CLASS_NAMES.index(c)], p[:, CLASS_NAMES.index(c)]) for c in labels]))


def holm(p_values):
    """Holm-adjusted p-values in the input order."""
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def hierarchical_difference(y, a, b, metric, seeds, resamples=RESAMPLES, seed=BOOT_SEED, clusters=None):
    """Mean-over-seeds metric(b) - metric(a); seeds resampled, then records, both shared by the two models.

    a, b: {seed: (N, 5) probabilities}. clusters: optional list of record-index arrays (one per patient); then
    patients are resampled and all their records kept. Returns point, CI and two-sided bootstrap p."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([metric(y, b[s]) for s in seeds]) - np.mean([metric(y, a[s]) for s in seeds]))
    draws = []
    for _ in range(resamples):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        if clusters is None:
            index = rng.integers(0, len(y), len(y))
        else:
            index = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        yy = y[index]
        draws.append(np.mean([metric(yy, b[s][index]) for s in chosen]) - np.mean([metric(yy, a[s][index]) for s in chosen]))
    draws = np.asarray(draws)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    p = min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))
    return {"difference": point, "ci_lower": float(lower), "ci_upper": float(upper), "p_bootstrap": float(p),
            "seeds": [int(s) for s in seeds]}


def load_predictions(dataset):
    out, by_run, truth = result_dir(dataset), {}, None
    for state_path in sorted(out.glob("*/inference_state.json")):
        state = read_json(state_path)
        if state["status"] != "completed":
            raise ValueError(f"Incomplete external inference: {state_path.parent.name}")
        with (state_path.parent / "predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        y = np.asarray([[int(r[f"truth_{c}"]) for c in CLASS_NAMES] for r in rows])
        truth = y if truth is None else truth
        if not np.array_equal(truth, y):
            raise ValueError("Truth differs between runs")
        by_run[state_path.parent.name] = (state, np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in rows]))
    return truth, by_run


def ptbxl_test_macro(family, run_id, labels):
    with (ROOT / "results" / family / run_id / "test_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([[int(r[f"truth_{c}"]) for c in CLASS_NAMES] for r in rows])
    return macro_auroc(y, np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in rows]), labels)


def analyze(dataset):
    from src.analyze_calibration_thresholds import ece
    labels = read_json(experiment_dir(dataset) / "inventory.json")["eligible_labels"]
    y, by_run = load_predictions(dataset)
    macro = lambda yy, pp: macro_auroc(yy, pp, labels)
    hyp = lambda yy, pp: auroc(yy[:, 4], pp[:, 4])
    per_run = []
    for run_id, (state, p) in sorted(by_run.items()):
        row = {"run": run_id, "family": state["family"], "kind": state["kind"], "page": state["page"], "head": state["head"],
               "seed": state["seed"], "external_macro_auroc": macro(y, p),
               "ptbxl_test_macro_auroc_same_labels": ptbxl_test_macro(state["family"], run_id, labels)}
        row["external_minus_ptbxl"] = row["external_macro_auroc"] - row["ptbxl_test_macro_auroc_same_labels"]
        for c in CLASS_NAMES:
            j = CLASS_NAMES.index(c)
            row[f"auroc_{c}"] = auroc(y[:, j], p[:, j])
            row[f"ece_{c}"] = ece(y[:, j], p[:, j]) if c in labels else float("nan")
        per_run.append(row)
    res = result_dir(dataset)
    write_csv(res / "per_run.csv", per_run)

    def table(head, page):
        return {s: p for _, (st, p) in by_run.items() for s in [st["seed"]] if st["head"] == head and st["page"] == page}

    h1 = []
    for a, b in itertools.combinations(HEADS, 2):
        ta, tb = table(a, "adaptive"), table(b, "adaptive")
        seeds = sorted(set(ta) & set(tb))
        if seeds:
            h1.append({"family": "H1", "comparison": f"{b} - {a}", "metric": "macro AUROC",
                       **hierarchical_difference(y, ta, tb, macro, seeds)})
    h2 = []
    for head in ("cnn_pool", "cnn_lstm", "temporal_conv", "hybrid_vit"):
        ta, tb = table(head, "adaptive"), table(head, "fixed")
        seeds = sorted(set(ta) & set(tb))
        for name, metric in (("macro AUROC", macro), ("HYP AUROC", hyp)):
            if seeds:
                h2.append({"family": "H2", "comparison": f"{head}: fixed - adaptive", "metric": name,
                           **hierarchical_difference(y, ta, tb, metric, seeds)})
    for family in (h1, h2):
        for row, adjusted in zip(family, holm([r["p_bootstrap"] for r in family])):
            row["p_holm"] = adjusted
            row["significant_at_0.05"] = adjusted < 0.05
    write_csv(res / "hypotheses.csv", h1 + h2)
    lines = [f"# External test: {dataset}", "", f"Eligible labels: {', '.join(labels)}. Frozen plan: docs/external_test_plan.md.",
             "", "## H1 and H2 (hierarchical seed x record bootstrap, Holm within family)", "",
             "| Family | Comparison | Metric | Difference [95% CI] | Holm p |", "|---|---|---|---|---:|"]
    lines += [f"| {r['family']} | {r['comparison']} | {r['metric']} | {r['difference']:+.4f} [{r['ci_lower']:+.4f}, "
              f"{r['ci_upper']:+.4f}] | {r['p_holm']:.4f} |" for r in h1 + h2]
    lines += ["", "## Per run (H3: external minus PTB-XL test, same labels)", "",
              "| Run | Page | Seed | External macro AUROC | PTB-XL test | Difference |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['run']} | {r['page']} | {r['seed']} | {r['external_macro_auroc']:.4f} | "
              f"{r['ptbxl_test_macro_auroc_same_labels']:.4f} | {r['external_minus_ptbxl']:+.4f} |" for r in per_run]
    (res / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["inventory", "render", "infer", "analyze"])
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.step == "inventory":
        inventory(arguments.dataset)
    elif arguments.step == "render":
        render(arguments.dataset, arguments.workers)
    elif arguments.step == "infer":
        infer(arguments.dataset, arguments.device)
    else:
        analyze(arguments.dataset)
