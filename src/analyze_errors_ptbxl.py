"""Subgroup sensitivity and error-overlap analysis from saved PTB-XL predictions.

Usage:
    python -m src.analyze_errors_ptbxl

Reads saved validation/test predictions of the seed-42 runs plus PTB-XL metadata
(age, sex, human validation, noise annotations) and the rendering manifest (page
gain). No model is run; thresholds stay at 0.5. Validation is the primary
partition for interpretation; test subgroups are descriptive only.
Writes results/error_analysis/ptbxl_seed42/.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from src.common import ROOT, write_csv

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
OUT = ROOT / "results/error_analysis/ptbxl_seed42"
RUNS = {
    "PCA + LDA": "results/ptbxl_v2/ptbxl_v2_lda_wide_seed42",
    "CNN (pooled)": "results/ptbxl_v2/ptbxl_v2_cnn_pool_seed42",
    "CNN-LSTM": "results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42",
    "CNN + temporal conv": "results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42",
    "CNN + small ViT": "results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42",
}
FOCUS = ("MI", "HYP", "CD", "STTC")


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def subgroups():
    """Per-ECG subgroup labels from PTB-XL metadata and rendering provenance."""
    meta = {row["ecg_id"]: row for row in read_csv(
        ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")}
    manifest = {row["ecg_id"]: row for row in read_csv(ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv")}
    groups = {}
    for ecg_id, row in manifest.items():
        m = meta[ecg_id]
        gain = float(row["gain_mm_per_mv"])
        age = float(m["age"]) if m["age"] else None
        n_labels = sum(row[f"label_{n}"] == "True" for n in CLASS_NAMES)
        groups[ecg_id] = {
            "page_gain": "10 or 5 mm/mV" if gain >= 5 else "2.5 mm/mV" if gain == 2.5 else "<=1.25 mm/mV",
            "sex_code": f"sex code {m['sex']}",
            "age": "unknown" if age is None or age > 120 else "<50" if age < 50 else "50-69" if age < 70 else ">=70",
            "label_count": "1 label" if n_labels == 1 else "2+ labels",
            "human_validated": f"validated_by_human={m['validated_by_human']}",
            "noise_annotation": "noise/drift annotated" if any(m[k] for k in ("baseline_drift", "static_noise", "burst_noise")) else "none annotated",
        }
    return groups


def gain_confound():
    """Label prevalence by page gain (training only) and per-label AUROC of page gain alone."""
    from sklearn.metrics import roc_auc_score
    manifest = [r for r in read_csv(ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv")
                if r["eligible_for_classification"] == "True"]
    prevalence, auroc = [], []
    train = [r for r in manifest if r["split"] == "train"]
    for level in ("10", "5", "2.5", "<=1.25"):
        chosen = [r for r in train if (float(r["gain_mm_per_mv"]) <= 1.25 if level == "<=1.25"
                                       else float(r["gain_mm_per_mv"]) == float(level))]
        prevalence.append({"gain_mm_per_mv": level, "train_ecgs": len(chosen),
                           **{n: sum(r[f"label_{n}"] == "True" for r in chosen) / len(chosen) for n in CLASS_NAMES}})
    for split in ("validation", "test"):
        part = [r for r in manifest if r["split"] == split]
        score = -np.log2([float(r["gain_mm_per_mv"]) for r in part])  # lower gain = larger page amplitude
        auroc.append({"partition": split, **{n: roc_auc_score([r[f"label_{n}"] == "True" for r in part], score)
                                             for n in CLASS_NAMES}})
    write_csv(OUT / "gain_prevalence_train.csv", prevalence)
    write_csv(OUT / "gain_only_auroc.csv", auroc)
    lines = ["", "## Page gain as an information channel", "",
             "The renderer lowers the page-wide gain (10 → 5 → 2.5 … mm/mV) when an ECG's maximum amplitude would leave "
             "its row band, and the gain is visible through calibration-pulse height. High QRS voltage is itself a "
             "hypertrophy criterion, so gain carries label information. Training-set prevalence by gain:", "",
             "| Gain (mm/mV) | Train ECGs | " + " | ".join(CLASS_NAMES) + " |", "|---|---:|" + "---:|" * 5]
    for r in prevalence:
        lines.append(f"| {r['gain_mm_per_mv']} | {r['train_ecgs']} | " + " | ".join(f"{r[n]:.3f}" for n in CLASS_NAMES) + " |")
    lines += ["", "AUROC of page gain alone (no image, no model):", "",
              "| Partition | " + " | ".join(CLASS_NAMES) + " |", "|---|" + "---:|" * 5]
    for r in auroc:
        lines.append(f"| {r['partition']} | " + " | ".join(f"{r[n]:.3f}" for n in CLASS_NAMES) + " |")
    lines += ["", "Reading: gain alone reaches HYP AUROC ≈0.72–0.74, and every model's HYP sensitivity is low (0.02–0.19 on validation) on "
              "pages at 5–10 mm/mV (the standard clinical print gain). HYP results on this benchmark therefore partly "
              "reflect amplitude conveyed by the rendering's gain choice and may not transfer to fixed-gain printouts."]
    return lines


def within_gain_auroc(available):
    """Per-label validation AUROC inside single gain strata: discrimination that gain cannot explain."""
    from sklearn.metrics import roc_auc_score
    gain = {r["ecg_id"]: float(r["gain_mm_per_mv"])
            for r in read_csv(ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv")}
    rows = []
    for model, directory in available.items():
        predictions = read_csv(directory / "validation_predictions.csv")
        for level in (5.0, 2.5):
            stratum = [p for p in predictions if gain[p["ecg_id"]] == level]
            for label in ("MI", "HYP"):
                truth = [p[f"truth_{label}"] == "1" for p in stratum]
                rows.append({"model": model, "gain_mm_per_mv": level, "label": label, "positives": sum(truth),
                             "ecgs": len(stratum), "validation_auroc": roc_auc_score(
                                 truth, [float(p[f"probability_{label}"]) for p in stratum])})
    write_csv(OUT / "within_gain_auroc_validation.csv", rows)
    lines = ["", "## Validation AUROC within a single gain level", "",
             "| Model | HYP @5 mm/mV | HYP @2.5 mm/mV | MI @5 mm/mV | MI @2.5 mm/mV |", "|---|---:|---:|---:|---:|"]
    for model in available:
        get = {(r["label"], r["gain_mm_per_mv"]): r for r in rows if r["model"] == model}
        lines.append(f"| {model} | " + " | ".join(f"{get[(l, g)]['validation_auroc']:.3f}" for l, g in
                                                  (("HYP", 5.0), ("HYP", 2.5), ("MI", 5.0), ("MI", 2.5))) + " |")
    positives = {(r["label"], r["gain_mm_per_mv"]): r["positives"] for r in rows}
    lines += ["", f"Positives: HYP {positives[('HYP', 5.0)]} @5, {positives[('HYP', 2.5)]} @2.5; MI {positives[('MI', 5.0)]} @5, "
              f"{positives[('MI', 2.5)]} @2.5. Within a fixed gain the models still rank HYP well above chance, so they use "
              "waveform content beyond gain. The low HYP sensitivity at 5 mm/mV mainly reflects the fixed 0.5 threshold "
              "applied where HYP prevalence is ~6%, not an absence of signal."]
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = subgroups()
    available = {name: ROOT / path for name, path in RUNS.items() if (ROOT / path / "run.json").exists()}
    rows, misses = [], defaultdict(dict)
    for split in ("validation", "test"):
        for model, directory in available.items():
            predictions = read_csv(directory / f"{split}_predictions.csv")
            for label in FOCUS:
                positives = [p for p in predictions if p[f"truth_{label}"] == "1"]
                missed = {p["ecg_id"] for p in positives if p[f"prediction_{label}"] == "0"}
                misses[(split, label)][model] = (missed, len(positives))
                for factor in ("page_gain", "sex_code", "age", "label_count", "human_validated", "noise_annotation"):
                    buckets = defaultdict(lambda: [0, 0])
                    for p in positives:
                        bucket = buckets[groups[p["ecg_id"]][factor]]
                        bucket[0] += p[f"prediction_{label}"] == "1"
                        bucket[1] += 1
                    for level, (hit, n) in sorted(buckets.items()):
                        rows.append({"partition": split, "model": model, "label": label, "factor": factor,
                                     "level": level, "positives": n, "detected": hit,
                                     "sensitivity": hit / n if n else ""})
    write_csv(OUT / "subgroup_sensitivity.csv", rows)
    overlap = []
    for (split, label), by_model in misses.items():
        sets = [s for s, _ in by_model.values()]
        n_pos = next(iter(by_model.values()))[1]
        overlap.append({"partition": split, "label": label, "positives": n_pos,
                        "missed_by_every_model": len(set.intersection(*sets)),
                        "missed_by_at_least_one": len(set.union(*sets)),
                        **{f"missed_{m}": len(s) for m, (s, _) in by_model.items()}})
    write_csv(OUT / "false_negative_overlap.csv", overlap)

    lines = ["# PTB-XL subgroup sensitivity and error overlap (seed 42, threshold 0.5)", "",
             "Generated by `python -m src.analyze_errors_ptbxl` from saved predictions. Validation (fold 9) is the "
             "partition for interpretation; test (fold 10) values are descriptive. Small subgroups are noisy; no "
             "hypothesis test is applied. Sex is reported as the PTB-XL code (0/1) exactly as stored.", "",
             f"Models: {', '.join(available)}.", "", "## False-negative overlap", "",
             "| Partition | Label | Positives | Missed by every model | Missed by ≥1 model |", "|---|---|---:|---:|---:|"]
    for o in overlap:
        lines.append(f"| {o['partition']} | {o['label']} | {o['positives']} | {o['missed_by_every_model']} | {o['missed_by_at_least_one']} |")
    for factor in ("page_gain", "label_count", "age", "sex_code", "human_validated", "noise_annotation"):
        lines += ["", f"## Validation sensitivity by {factor.replace('_', ' ')}", "",
                  "| Label | Level | Positives | " + " | ".join(available) + " |",
                  "|---|---|---:|" + "---:|" * len(available)]
        for label in FOCUS:
            levels = sorted({r["level"] for r in rows if r["partition"] == "validation" and r["label"] == label and r["factor"] == factor})
            for level in levels:
                cells = {r["model"]: r for r in rows if r["partition"] == "validation" and r["label"] == label
                         and r["factor"] == factor and r["level"] == level}
                n = next(iter(cells.values()))["positives"]
                lines.append(f"| {label} | {level} | {n} | " + " | ".join(f"{cells[m]['sensitivity']:.2f}" for m in available) + " |")
    lines += gain_confound() + within_gain_auroc(available)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
