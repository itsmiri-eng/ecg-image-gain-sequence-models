"""Q1 transfer of docs/ptbxl_fixed_gain_protocol.md on any fold, with a one-time guard on the test fold.

Usage:
    python -X utf8 -m src.evaluate_fixed_gain_transfer_test --split validation --models cnn_pool
    python -X utf8 -m src.evaluate_fixed_gain_transfer_test --split test

Adaptive-gain seed-42 models scored on fixed-gain (2.5 mm/mV) pages, exactly as
src/evaluate_fixed_gain_transfer.py does for validation (which stays unchanged). No training; CPU only.
The protocol's four models are PCA + LDA (wide), CNN-LSTM, temporal conv and hybrid ViT. The CNN-only
model did not exist when the protocol was frozen and is a disclosed addition (validation first).

Test fold: each model's fixed-gain test pages are scored exactly once. A test_inference_state.json is
written before inference; if it exists, the model is refused. Outputs go to results/fixed_gain/transfer_<split>[_<tag>]/.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.common import ROOT, read_json, utc_now, write_csv, write_json
from src.compare_ptbxl_models import paired_patient_bootstrap
from src.evaluate_fixed_gain_transfer import CLASS_NAMES, LDA, NETWORKS, lda_probabilities, saved_probabilities
from src.evaluation.multilabel import evaluate_multilabel, save_multilabel_evaluation
from src.ptbxl_fixed_gain import FixedGainDataset
from src.run_ptbxl_experiment import labels
from src.training.ptbxl_trainer import MODELS, predict_proba

ALL = {"lda": (LDA[0], ("lda", LDA[1])), "cnn_pool":("CNN (pooled)", ("cnn_pool", "ptbxl_v2/ptbxl_v2_cnn_pool_seed42")),
       **{spec[0]: (name, spec) for name, spec in NETWORKS.items()}}
PROTOCOL = ["lda", "cnn_lstm", "temporal_conv", "hybrid_vit"]
ADDED = {"cnn_pool": "not in the frozen protocol (model trained after the freeze); disclosed addition"}


def probabilities(key, split, reference):
    if key == "lda":
        return lda_probabilities(reference), LDA[1]
    model_key, result_dir = ALL[key][1]
    checkpoint = torch.load(ROOT / "models" / result_dir / "best.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(0)
    model = MODELS[model_key]()
    model.load_state_dict(checkpoint["state_dict"])
    dataset = FixedGainDataset(split=split, normalization=checkpoint["normalization"])
    return predict_proba(model, dataset, 16, f"transfer-{split}:{model_key}"), result_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--models", nargs="+", choices=list(ALL), default=PROTOCOL + ["cnn_pool"])
    parser.add_argument("--tag", default="", help="suffix for the output folder")
    arguments = parser.parse_args()
    split = arguments.split
    out = ROOT / "results/fixed_gain" / (f"transfer_{split}" + (f"_{arguments.tag}" if arguments.tag else ""))
    if split == "validation" and not arguments.tag:
        raise ValueError("results/fixed_gain/transfer_validation holds the protocol's original run; use --tag")
    for key in arguments.models:  # refuse before any inference if a test transfer already started
        state = out / ALL[key][1][1].split("/")[-1] / "test_inference_state.json"
        if split == "test" and state.exists():
            raise FileExistsError(f"Test transfer for {key} already started at {read_json(state).get('started_utc')}; never repeated")
    reference = FixedGainDataset(split=split, normalize=False)
    truth, patients = labels(reference), [r["patient_id"] for r in reference.rows]
    v1_gain = np.asarray([float(r["v1_gain_mm_per_mv"]) for r in reference.rows])
    strata = {"all": np.ones(len(truth), bool), "v1 gain 2.5 (unchanged)": v1_gain == 2.5,
              "v1 gain 5 or 10 (now smaller)": v1_gain >= 5, "v1 gain <=1.25 (now larger)": v1_gain <= 1.25}
    summary, per_label, checks = [], [], {}
    for key in arguments.models:
        name, result_dir = ALL[key][0], ALL[key][1][1]
        directory = out / result_dir.split("/")[-1]
        if split == "test":
            directory.mkdir(parents=True, exist_ok=False)
            write_json(directory / "test_inference_state.json", {"status": "started", "started_utc": utc_now(),
                                                                 "inference_calls": 1, "model_result_dir": result_dir})
        fixed, _ = probabilities(key, split, reference)
        ids, original = saved_probabilities(result_dir, split)
        if ids != [r["ecg_id"] for r in reference.rows]:
            raise ValueError(f"{split} order differs between saved predictions and fixed-gain cache")
        unchanged = float(np.abs(fixed[strata["v1 gain 2.5 (unchanged)"]] - original[strata["v1 gain 2.5 (unchanged)"]]).max())
        save_multilabel_evaluation(directory, split, reference.rows, truth, fixed, CLASS_NAMES)
        if split == "test":
            write_json(directory / "test_inference_state.json", {"status": "completed", "completed_utc": utc_now(),
                       "inference_calls": 1, "model_result_dir": result_dir, "records": len(truth)})
        boot = paired_patient_bootstrap(truth, original, fixed, patients, CLASS_NAMES, n_resamples=500, seed=2024)
        checks[name] = {"max_abs_probability_change_on_unchanged_gain_pages": unchanged,
                        "macro_auroc_fixed_minus_original": [boot["metrics"]["macro_roc_auc"][k] for k in ("difference", "ci_lower", "ci_upper")],
                        "in_frozen_protocol": key in PROTOCOL, **({"note": ADDED[key]} if key in ADDED else {})}
        for stratum, mask in strata.items():
            before, after = evaluate_multilabel(truth[mask], original[mask], CLASS_NAMES), evaluate_multilabel(truth[mask], fixed[mask], CLASS_NAMES)
            summary.append({"model": name, "stratum": stratum, "ecgs": int(mask.sum()),
                            "macro_auroc_original_pages": before["macro_roc_auc"], "macro_auroc_fixed_gain_pages": after["macro_roc_auc"],
                            "macro_f1_original_pages": before["macro_f1"], "macro_f1_fixed_gain_pages": after["macro_f1"]})
            for label in CLASS_NAMES:
                b, a = before["per_label"][label], after["per_label"][label]
                per_label.append({"model": name, "stratum": stratum, "label": label, "positives": b["support"],
                                  "auroc_original": b["roc_auc"], "auroc_fixed_gain": a["roc_auc"],
                                  "sensitivity_original": b["recall"], "sensitivity_fixed_gain": a["recall"],
                                  "specificity_original": b["specificity"], "specificity_fixed_gain": a["specificity"]})
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_label.csv", per_label)
    write_json(out / "checks.json", {"split": split, "created_utc": utc_now(), "models": checks})
    exploratory = " Test fold 10 was observed before, so these are exploratory." if split == "test" else ""
    lines = [f"# Q1 transfer: adaptive-gain models on fixed-gain (2.5 mm/mV) {split} pages", "",
             f"Generated by `python -m src.evaluate_fixed_gain_transfer_test --split {split}`; no training; CPU.{exploratory} "
             "Built-in check: pages already at 2.5 mm/mV are pixel-identical, so their predictions should not change.", "",
             "| Model | Stratum (v1 gain) | ECGs | Macro AUROC original → fixed | Macro F1 original → fixed |", "|---|---|---:|---|---|"]
    lines += [f"| {s['model']} | {s['stratum']} | {s['ecgs']} | {s['macro_auroc_original_pages']:.4f} → {s['macro_auroc_fixed_gain_pages']:.4f} | "
              f"{s['macro_f1_original_pages']:.4f} → {s['macro_f1_fixed_gain_pages']:.4f} |" for s in summary]
    lines += ["", "Paired patient-bootstrap change in macro AUROC, all ECGs (fixed − original), and the unchanged-page check:", ""]
    lines += [f"- {k}: {v['macro_auroc_fixed_minus_original'][0]:+.4f} [{v['macro_auroc_fixed_minus_original'][1]:+.4f}, "
              f"{v['macro_auroc_fixed_minus_original'][2]:+.4f}]; max change on unchanged pages {v['max_abs_probability_change_on_unchanged_gain_pages']:.1e}"
              + (f" ({v['note']})" if "note" in v else "") for k, v in checks.items()]
    lines += ["", "## HYP, MI and NORM, pages whose gain changed from 5/10 to 2.5 mm/mV", "",
              "| Model | Label | Positives | AUROC original → fixed | Sensitivity original → fixed |", "|---|---|---:|---|---|"]
    lines += [f"| {p['model']} | {p['label']} | {p['positives']} | {p['auroc_original']:.3f} → {p['auroc_fixed_gain']:.3f} | "
              f"{p['sensitivity_original']:.3f} → {p['sensitivity_fixed_gain']:.3f} |"
              for p in per_label if p["stratum"] == "v1 gain 5 or 10 (now smaller)" and p["label"] in ("HYP", "MI", "NORM")]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
