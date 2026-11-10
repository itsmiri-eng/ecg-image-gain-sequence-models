"""Q1 of docs/ptbxl_fixed_gain_protocol.md: adaptive-gain models scored on fixed-gain validation pages.

Usage:
    python -m src.evaluate_fixed_gain_transfer

No training. Each seed-42 model keeps its own training normalization (as a deployed
model would). Validation only; test-fold transfer is deferred to the final analysis.
Writes results/fixed_gain/transfer_validation/.
"""
from __future__ import annotations

import csv

import numpy as np
import torch

from src.common import ROOT, write_csv, write_json
from src.compare_ptbxl_models import paired_patient_bootstrap
from src.evaluation.multilabel import evaluate_multilabel, save_multilabel_evaluation
from src.ptbxl_fixed_gain import FixedGainDataset
from src.run_ptbxl_experiment import labels
from src.run_ptbxl_lda import features
from src.training.ptbxl_trainer import MODELS, predict_proba

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
OUT = ROOT / "results/fixed_gain/transfer_validation"
NETWORKS = {
    "CNN-LSTM": ("cnn_lstm", "ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42"),
    "CNN + temporal conv": ("temporal_conv", "ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42"),
    "CNN + small ViT": ("hybrid_vit", "ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42"),
}
LDA = ("PCA + LDA", "ptbxl_v2/ptbxl_v2_lda_wide_seed42")


def saved_probabilities(result_dir, split="validation"):
    with (ROOT / "results" / result_dir / f"{split}_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return [r["ecg_id"] for r in rows], np.asarray([[float(r[f"probability_{n}"]) for n in CLASS_NAMES] for r in rows])


def lda_probabilities(dataset):
    stored = np.load(ROOT / "models" / LDA[1] / "pca_lda.npz")
    x = features(dataset, directory=ROOT / "data/processed/ptbxl_lda_features_fixedgain2p5_v1")
    z = (x - stored["pca_mean"]) @ stored["pca_components"].T
    return 1 / (1 + np.exp(-(z @ stored["lda_coef"].T + stored["lda_intercept"])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = FixedGainDataset(split="validation", normalize=False)
    truth = labels(reference)
    v1_gain = np.asarray([float(r["v1_gain_mm_per_mv"]) for r in reference.rows])
    strata = {"all": np.ones(len(truth), bool), "v1 gain 2.5 (unchanged)": v1_gain == 2.5,
              "v1 gain 5 or 10 (now smaller)": v1_gain >= 5, "v1 gain <=1.25 (now larger)": v1_gain <= 1.25}
    summary, per_label, checks = [], [], {}
    models = {LDA[0]: None, **NETWORKS}
    for name, spec in models.items():
        if spec is None:
            fixed, result_dir = lda_probabilities(reference), LDA[1]
        else:
            key, result_dir = spec
            checkpoint = torch.load(next((ROOT / "models" / result_dir).glob("best.pt")), map_location="cpu", weights_only=True)
            torch.manual_seed(0)
            model = MODELS[key]()
            model.load_state_dict(checkpoint["state_dict"])
            dataset = FixedGainDataset(split="validation", normalization=checkpoint["normalization"])
            fixed = predict_proba(model, dataset, 16, f"transfer:{key}")
        ids, original = saved_probabilities(result_dir)
        if ids != [r["ecg_id"] for r in reference.rows]:
            raise ValueError("Validation order differs between v1 predictions and fixed-gain cache")
        unchanged = strata["v1 gain 2.5 (unchanged)"]
        checks[name] = float(np.abs(fixed[unchanged] - original[unchanged]).max())
        directory = OUT / result_dir.split("/")[-1]
        save_multilabel_evaluation(directory, "validation", reference.rows, truth, fixed, CLASS_NAMES)
        for stratum, mask in strata.items():
            before = evaluate_multilabel(truth[mask], original[mask], CLASS_NAMES)
            after = evaluate_multilabel(truth[mask], fixed[mask], CLASS_NAMES)
            summary.append({"model": name, "stratum": stratum, "ecgs": int(mask.sum()),
                            "macro_auroc_original_pages": before["macro_roc_auc"],
                            "macro_auroc_fixed_gain_pages": after["macro_roc_auc"],
                            "macro_f1_original_pages": before["macro_f1"], "macro_f1_fixed_gain_pages": after["macro_f1"]})
            for label in CLASS_NAMES:
                b, a = before["per_label"][label], after["per_label"][label]
                per_label.append({"model": name, "stratum": stratum, "label": label, "positives": b["support"],
                                  "auroc_original": b["roc_auc"], "auroc_fixed_gain": a["roc_auc"],
                                  "sensitivity_original": b["recall"], "sensitivity_fixed_gain": a["recall"],
                                  "specificity_original": b["specificity"], "specificity_fixed_gain": a["specificity"]})
        boot = paired_patient_bootstrap(truth, original, fixed, [r["patient_id"] for r in reference.rows], CLASS_NAMES,
                                        n_resamples=500, seed=2024)
        checks[f"{name} macro AUROC change [95% CI]"] = [boot["metrics"]["macro_roc_auc"][k] for k in ("difference", "ci_lower", "ci_upper")]
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "per_label.csv", per_label)
    write_json(OUT / "checks.json", {"max_abs_probability_change_on_unchanged_gain_pages": {k: v for k, v in checks.items() if "CI" not in k},
                                     "paired_bootstrap_macro_auroc_fixed_minus_original": {k: v for k, v in checks.items() if "CI" in k}})
    lines = ["# Q1 transfer: adaptive-gain models on fixed-gain (2.5 mm/mV) validation pages", "",
             "Generated by `python -m src.evaluate_fixed_gain_transfer`; no training. Validation fold 9 (2,146 ECGs). "
             "Built-in check: pages already at 2.5 mm/mV in v1 are pixel-identical, so their predictions must not change "
             f"(maximum absolute probability change: {', '.join(f'{k} {v:.1e}' for k, v in checks.items() if 'CI' not in k)}).", "",
             "| Model | Stratum (v1 gain) | ECGs | Macro AUROC original → fixed | Macro F1 original → fixed |", "|---|---|---:|---|---|"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['stratum']} | {s['ecgs']} | {s['macro_auroc_original_pages']:.4f} → "
                     f"{s['macro_auroc_fixed_gain_pages']:.4f} | {s['macro_f1_original_pages']:.4f} → {s['macro_f1_fixed_gain_pages']:.4f} |")
    lines += ["", "Paired patient-bootstrap change in macro AUROC, all validation ECGs (fixed − original):", ""]
    lines += [f"- {k.replace(' macro AUROC change [95% CI]', '')}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in checks.items() if "CI" in k]
    lines += ["", "## HYP and MI, pages whose gain changed from 5/10 to 2.5 mm/mV", "",
              "| Model | Label | Positives | AUROC original → fixed | Sensitivity original → fixed |", "|---|---|---:|---|---|"]
    for p in per_label:
        if p["stratum"] == "v1 gain 5 or 10 (now smaller)" and p["label"] in ("HYP", "MI", "NORM"):
            lines.append(f"| {p['model']} | {p['label']} | {p['positives']} | {p['auroc_original']:.3f} → {p['auroc_fixed_gain']:.3f} | "
                         f"{p['sensitivity_original']:.3f} → {p['sensitivity_fixed_gain']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
