"""Pulse-swap intervention: does the calibration pulse change the HYP output? (docs/secondary_analyses_protocol.md, D)

Usage:
    python -X utf8 -m src.analyze_pulse_swap [--device cuda]

Validation pages only; no training. Only columns 48-87 of every lead row (the 1 mV calibration pulse) are
replaced, in raw ink space, before each model's own normalization. Conditions: original, pulse_10, pulse_5,
pulse_2.5 (block from the first validation page of that v1 gain), sham (block from another page of the
page's own gain) and mask (blank). Seed-42 adaptive-gain networks on adaptive pages; seed-42 fixed-gain
networks on fixed-gain pages as the control. Built-in check: `original` must reproduce each run's saved
validation probabilities. Writes results/secondary/pulse_swap/.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.analyze_saliency import NETWORKS, PULSE
from src.common import ROOT, read_json, write_csv, write_json
from src.ptbxl_fixed_gain import FixedGainDataset
from src.run_ptbxl_experiment import make_dataset
from src.training.ptbxl_trainer import MODELS

OUT = ROOT / "results/secondary/pulse_swap"
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
GAINS = (10.0, 5.0, 2.5)
CONDITIONS = ["original", "pulse_10", "pulse_5", "pulse_2.5", "sham", "mask"]


def gain_of(row):
    return float(row.get("v1_gain_mm_per_mv", row.get("gain_mm_per_mv")))


def reference_indices(rows):
    """First and second validation page (cache order) at each gain: swaps use the first, sham the second."""
    found = {g: [i for i, r in enumerate(rows) if gain_of(r) == g][:2] for g in GAINS}
    if any(len(v) < 2 for v in found.values()):
        raise ValueError("Need two reference pages per gain")
    return found


def apply(x, condition, blocks, sham_blocks):
    """x: raw ink [B, 12, 120, 1120]; returns a copy with only the pulse columns changed."""
    if condition == "original":
        return x
    y = x.clone()
    if condition == "mask":
        y[..., PULSE] = 0
    elif condition == "sham":
        y[..., PULSE] = sham_blocks
    else:
        y[..., PULSE] = blocks[float(condition.split("_")[1])]
    return y


def cluster_bootstrap(values, patients, seed=2024, resamples=1000):
    groups = {}
    for v, p in zip(values, patients):
        groups.setdefault(p, []).append(v)
    sums, counts = np.array([sum(g) for g in groups.values()]), np.array([len(g) for g in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), (resamples, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def saved_validation(relative):
    with (ROOT / "results" / relative / "validation_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(r[f"probability_{n}"]) for n in CLASS_NAMES] for r in rows])


@torch.inference_mode()
def score(setting, dataset, models, reference_rows, device):
    rows = dataset.rows
    refs = reference_indices(reference_rows)
    # Swapped pulses always come from adaptive pages (fixed-gain pages all carry the 2.5 pulse); the sham block
    # comes from the model's own kind of page, so it is a like-for-like copy.
    blocks = {g: dataset_block(reference_rows, refs[g][0], "adaptive") for g in GAINS}
    own = [gain_of(r) for r in rows]
    second = {g: dataset_block(reference_rows, refs[g][1], setting) for g in GAINS}
    out = {(name, c): [] for name in models for c in CONDITIONS}
    for start, (x, _) in zip(range(0, len(rows), 16), DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)):
        sham = torch.stack([second[g] if g in second else torch.zeros_like(blocks[10.0]) for g in own[start:start + len(x)]])
        for condition in CONDITIONS:
            changed = apply(x, condition, blocks, sham).to(device)
            for name, (model, norm) in models.items():
                out[(name, condition)].append(torch.sigmoid(model((changed - norm["mean"]) / norm["std"])).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


_PAGES = {}


def dataset_block(rows, index, setting):
    """Raw-ink pulse block [12, 120, 40] of reference page `index` (adaptive references for the swap gains)."""
    key = (setting, index)
    if key not in _PAGES:
        source = (make_dataset(read_json(ROOT / "experiments/configs/ptbxl_v2_cnn_pool.json"), "validation", normalize=False)
                  if setting == "adaptive" else FixedGainDataset(split="validation", normalize=False))
        _PAGES[key] = source[index][0][..., PULSE].clone()
    return _PAGES[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(parser.parse_args().device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # full float32, as in training
    OUT.mkdir(parents=True, exist_ok=True)
    adaptive =make_dataset(read_json(ROOT / "experiments/configs/ptbxl_v2_cnn_pool.json"), "validation", normalize=False)
    truth = np.asarray([[int(bool(r[f"label_{n}"])) for n in CLASS_NAMES] for r in adaptive.rows])
    patients = [r["patient_id"] for r in adaptive.rows]
    strata = {"all": np.ones(len(truth), bool), "v1 gain 5 or 10": np.array([gain_of(r) >= 5 for r in adaptive.rows]),
              "v1 gain 2.5": np.array([gain_of(r) == 2.5 for r in adaptive.rows])}
    summary, checks = [], {}
    for setting, dataset in (("adaptive", adaptive), ("fixed", FixedGainDataset(split="validation", normalize=False))):
        if [str(r["ecg_id"]) for r in dataset.rows] != [str(r["ecg_id"]) for r in adaptive.rows]:
            raise ValueError("Fixed-gain validation order differs")
        models, saved = {}, {}
        for name, (key, relative) in NETWORKS.items():
            model_relative = relative if setting == "adaptive" else f"ptbxl_v2_fixedgain/ptbxl_v2_fg_{key}_seed42"
            checkpoint = torch.load(ROOT / "models" / model_relative / "best.pt", map_location="cpu", weights_only=True)
            model = MODELS[key]()
            model.load_state_dict(checkpoint["state_dict"])
            models[name] = (model.to(device).eval(), checkpoint["normalization"])
            saved[name] = saved_validation(model_relative)
        probabilities = score(setting, dataset, models, adaptive.rows, device)
        for name in models:
            checks[f"{name} | {setting}"] = float(np.abs(probabilities[(name, "original")] - saved[name]).max())
            base = probabilities[(name, "original")]
            for condition in CONDITIONS:
                p = probabilities[(name, condition)]
                for stratum, mask in strata.items():
                    if condition == "sham":
                        mask = mask & np.array([gain_of(r) in GAINS for r in adaptive.rows])
                    change = cluster_bootstrap((p[mask, 4] - base[mask, 4]).tolist(), [q for q, m in zip(patients, mask) if m])
                    summary.append({
                        "models": f"{setting}-gain", "model": name, "condition": condition, "stratum": stratum, "ecgs": int(mask.sum()),
                        "mean_hyp_probability": float(p[mask, 4].mean()), "hyp_change_mean": change[0],
                        "hyp_change_ci_lower": change[1], "hyp_change_ci_upper": change[2],
                        "norm_change_mean": float((p[mask, 0] - base[mask, 0]).mean()),
                        "hyp_auroc": float(roc_auc_score(truth[mask, 4], p[mask, 4])),
                        "hyp_sensitivity_at_0.5": float(((p[mask, 4] >= .5) & (truth[mask, 4] == 1)).sum() / max(truth[mask, 4].sum(), 1)),
                        "macro_auroc": float(np.mean([roc_auc_score(truth[mask, j], p[mask, j]) for j in range(5)]))})
            print(f"{setting} {name}: original vs saved max diff {checks[f'{name} | {setting}']:.1e}", flush=True)
    write_csv(OUT / "summary.csv", summary)
    write_json(OUT / "checks.json", {"max_abs_difference_original_vs_saved_validation_probabilities": checks,
                                     "pulse_columns": [PULSE.start, PULSE.stop - 1]})
    lines = ["# Pulse-swap intervention (validation, seed 42)", "",
             "Generated by `python -m src.analyze_pulse_swap` (protocol: docs/secondary_analyses_protocol.md, D). Only the "
             "calibration-pulse columns 48-87 are changed. Change = mean HYP probability minus the unchanged page's "
             "[95% patient-cluster bootstrap]. Check: `original` reproduces saved validation probabilities to "
             f"{max(checks.values()):.1e}.", ""]
    for stratum in strata:
        lines += [f"## Pages at {stratum}" if stratum != "all" else "## All validation pages", "",
                  "| Models | Model | Condition | ECGs | HYP change [95% CI] | HYP AUROC | HYP sens. 0.5 | Macro AUROC |",
                  "|---|---|---|---:|---|---:|---:|---:|"]
        lines += [f"| {s['models']} | {s['model']} | {s['condition']} | {s['ecgs']} | {s['hyp_change_mean']:+.3f} "
                  f"[{s['hyp_change_ci_lower']:+.3f}, {s['hyp_change_ci_upper']:+.3f}] | {s['hyp_auroc']:.3f} | "
                  f"{s['hyp_sensitivity_at_0.5']:.3f} | {s['macro_auroc']:.4f} |" for s in summary if s["stratum"] == stratum]
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
