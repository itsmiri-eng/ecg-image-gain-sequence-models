"""POST-HOC (specified 3 Nov 2026, after the pulse-swap and transfer results were seen): split the HYP pulse effect.

Usage:
    python -X utf8 -m src.decompose_pulse_effect

On validation pages originally drawn at 5-10 mm/mV, for each seed-42 adaptive-gain network:
- consistent re-render (fixed-gain transfer pages: waveform and pulse both redrawn at 2.5 mm/mV, true voltage
  unchanged): the change in mean HYP probability can only come from treating "a 2.5 mm/mV page" as a hint
  (the renderer's gain prior), plus any resolution effect;
- pulse swap to 2.5 (only the pulse changed, waveform unchanged): the implied voltage rises 2-4x, so the
  change combines the gain prior with reading the pulse as a voltage scale (calibration, as a cardiologist does).
The difference approximates calibration reading. Approximate: re-rendering also shrinks the waveform.
Reads saved outputs only. Writes results/secondary/pulse_swap/decomposition_posthoc.{csv,md}.
"""
from __future__ import annotations

import csv

import numpy as np

from src.analyze_saliency import NETWORKS
from src.common import ROOT, write_csv

OUT = ROOT / "results/secondary/pulse_swap"


def hyp(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [r["ecg_id"] for r in rows], np.asarray([float(r["probability_HYP"]) for r in rows])


def main():
    with (ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv").open(newline="") as handle:
        gain = {r["ecg_id"]: float(r["gain_mm_per_mv"]) for r in csv.DictReader(handle)}
    with (OUT / "summary.csv").open(newline="") as handle:
        swap = {r["model"]: float(r["hyp_change_mean"]) for r in csv.DictReader(handle)
                if r["models"] == "adaptive-gain" and r["condition"] == "pulse_2.5" and r["stratum"] == "v1 gain 5 or 10"}
    rows = []
    for name, (_, relative) in NETWORKS.items():
        ids, original = hyp(ROOT / "results" / relative / "validation_predictions.csv")
        ids_t, redrawn = hyp(ROOT / "results/fixed_gain/transfer_validation_pc_check" / relative.split("/")[1] / "validation_predictions.csv")
        if ids != ids_t:
            raise ValueError(f"{name}: validation order differs")
        mask = np.asarray([gain[i] >= 5 for i in ids])
        rerender = float((redrawn[mask] - original[mask]).mean())
        rows.append({"model": name, "ecgs": int(mask.sum()), "consistent_rerender_hyp_change": rerender,
                     "pulse_swap_hyp_change": swap[name], "approx_calibration_reading": swap[name] - rerender})
    write_csv(OUT / "decomposition_posthoc.csv", rows)
    lines = ["# POST-HOC: gain prior vs calibration reading (validation, pages originally at 5-10 mm/mV, seed 42)", "",
             "Specified after the pulse-swap and transfer results were seen; see the module docstring of "
             "`src/decompose_pulse_effect.py`. Mean change in HYP probability.", "",
             "| Model | ECGs | Consistent re-render (gain prior) | Pulse swap to 2.5 (prior + calibration) | Difference (≈ calibration reading) |",
             "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['ecgs']} | {r['consistent_rerender_hyp_change']:+.3f} | {r['pulse_swap_hyp_change']:+.3f} | "
              f"{r['approx_calibration_reading']:+.3f} |" for r in rows]
    (OUT / "decomposition_posthoc.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
