"""Does the HYP output use the calibration pulse? (docs/secondary_analyses_protocol.md, C)

Usage:
    python -X utf8 -m src.analyze_saliency [--device cuda]

Integrated gradients (16 midpoint steps, blank-page baseline) of the HYP and NORM logits for seed-42
CNN-only, CNN-LSTM, temporal conv and hybrid ViT, each as the adaptive-gain model on adaptive pages and as
the fixed-gain retrained model on fixed-gain pages. Validation only: all 268 HYP-positive ECGs and the
first 268 HYP-negative ECGs in cache order. Absolute attributions are summed per column of the
12 x 120 x 1120 input; the calibration pulse occupies columns 48-87 and the waveform columns 100-1099.
No model is trained or changed. Writes results/secondary/saliency/.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.common import ROOT, read_json, write_csv, write_json
from src.ptbxl_fixed_gain import FixedGainDataset
from src.run_ptbxl_experiment import make_dataset
from src.training.ptbxl_trainer import MODELS

OUT = ROOT / "results/secondary/saliency"
PULSE, WAVEFORM = slice(48, 88), slice(100, 1100)
STEPS, PER_GROUP = 16, 268
LOGITS = {"HYP": 4, "NORM": 0}
NETWORKS = {
    "CNN-only": ("cnn_pool", "ptbxl_v2/ptbxl_v2_cnn_pool_seed42"),
    "CNN-LSTM": ("cnn_lstm", "ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42"),
    "Temporal conv": ("temporal_conv", "ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42"),
    "Hybrid ViT": ("hybrid_vit", "ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42"),
}
COLORS = {"adaptive gain": "#2a78d6", "fixed gain": "#eb6834"}


def load(key, relative, device):
    checkpoint = torch.load(ROOT / "models" / relative / "best.pt", map_location="cpu", weights_only=True)
    model = MODELS[key]()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["normalization"]


def selection(rows):
    positive = [i for i, r in enumerate(rows) if r["label_HYP"]]
    negative = [i for i, r in enumerate(rows) if not r["label_HYP"]][:PER_GROUP]
    if len(positive) != PER_GROUP:
        raise ValueError(f"Expected {PER_GROUP} HYP-positive validation ECGs, found {len(positive)}")
    return positive + negative


def column_attributions(model, x, baseline, logit, device):
    """|Integrated gradients| of one logit, summed over leads and rows -> 1120 column totals."""
    x, baseline = x.to(device), baseline.to(device)
    alphas = ((torch.arange(STEPS, device=device) + 0.5) / STEPS).view(-1, 1, 1, 1)
    path = (baseline + alphas * (x - baseline)).requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):  # cuDNN's LSTM has no backward pass in eval mode
        model(path)[:, logit].sum().backward()
    attribution = path.grad.mean(dim=0) * (x - baseline)[0]  # [12, 120, 1120]
    return attribution.abs().sum(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)


def bootstrap(values, seed=2024, resamples=1000):
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(resamples)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot(profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(len(NETWORKS), 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, NETWORKS):
        axis.axvspan(PULSE.start, PULSE.stop, color="#e5e4df", zorder=0)
        for condition, color in COLORS.items():
            profile = profiles[(name, condition)]
            axis.plot(np.arange(len(profile)), profile, color=color, linewidth=1.2, label=condition)
        axis.set_title(name, loc="left", color="#0b0b0b")
        axis.set_ylabel("Share per column (%)")
        axis.grid(alpha=.15)
    axes[0].text(PULSE.start + 2, axes[0].get_ylim()[1] * 0.85, "calibration pulse", color="#52514e", fontsize=7)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Column of the 1120-pixel lead row (pulse 48-87, waveform 100-1099)")
    fig.suptitle("Where the HYP output looks: mean |integrated gradients| per column, HYP-positive validation ECGs",
                 color="#0b0b0b")
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"hyp_column_profiles.{suffix}", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(parser.parse_args().device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # full float32, as in training
    OUT.mkdir(parents=True, exist_ok=True)
    adaptive_rows = make_dataset(read_json(ROOT / "experiments/configs/ptbxl_v2_cnn_pool.json"), "validation").rows
    chosen = selection(adaptive_rows)
    records, summary, profiles = [], [], {}
    for name, (key, relative) in NETWORKS.items():
        for condition, model_relative in (("adaptive gain", relative), ("fixed gain", f"ptbxl_v2_fixedgain/ptbxl_v2_fg_{key}_seed42")):
            model, normalization = load(key, model_relative, device)
            dataset = (make_dataset(read_json(ROOT / "experiments/configs/ptbxl_v2_cnn_pool.json"), "validation")
                       if condition == "adaptive gain" else FixedGainDataset(split="validation", normalization=normalization))
            if (dataset.normalization != normalization
                    or [str(r["ecg_id"]) for r in dataset.rows] != [str(r["ecg_id"]) for r in adaptive_rows]):
                raise ValueError(f"{name} {condition}: normalization or row order differs from the model's")
            baseline = torch.full((1, 12, 120, 1120), -normalization["mean"] / normalization["std"])
            hyp_profiles = []
            for n, index in enumerate(chosen, start=1):
                x = dataset[index][0].unsqueeze(0)
                for label, logit in LOGITS.items():
                    columns = column_attributions(model, x, baseline, logit, device)
                    total = columns.sum()
                    positive = bool(adaptive_rows[index]["label_HYP"])  # same ECG and label in both caches (order checked)
                    records.append({"model": name, "condition": condition, "ecg_id": adaptive_rows[index]["ecg_id"],
                                    "hyp_positive": int(positive), "logit": label,
                                    "pulse_share": columns[PULSE].sum() / total, "waveform_share": columns[WAVEFORM].sum() / total})
                    if label == "HYP" and positive:
                        hyp_profiles.append(columns / total * 100)
                if n % 100 == 0:
                    print(f"{name} {condition}: {n}/{len(chosen)}", flush=True)
            profiles[(name, condition)] = np.mean(hyp_profiles, axis=0)
            for label in LOGITS:
                for group, flag in (("HYP-positive", 1), ("HYP-negative", 0)):
                    mine = [r for r in records if r["model"] == name and r["condition"] == condition
                            and r["logit"] == label and r["hyp_positive"] == flag]
                    pulse = bootstrap(np.asarray([r["pulse_share"] for r in mine]))
                    summary.append({"model": name, "condition": condition, "logit": label, "ecgs": group, "n": len(mine),
                                    "pulse_share_mean": pulse[0], "pulse_share_ci_lower": pulse[1], "pulse_share_ci_upper": pulse[2],
                                    "waveform_share_mean": float(np.mean([r["waveform_share"] for r in mine]))})
            print(f"{name} {condition}: HYP pulse share (HYP+) {summary[-4]['pulse_share_mean']:.4f}", flush=True)
    write_csv(OUT / "per_ecg.csv", records)
    write_csv(OUT / "summary.csv", summary)
    write_json(OUT / "column_profiles_hyp_positive.json", {f"{k[0]} | {k[1]}": v.tolist() for k, v in profiles.items()})
    plot(profiles)
    area = (PULSE.stop - PULSE.start) / 1120
    lines = ["# Saliency: does the HYP output use the calibration pulse?", "",
             "Generated by `python -m src.analyze_saliency` (protocol: docs/secondary_analyses_protocol.md, C). Validation "
             f"ECGs only; integrated gradients, {STEPS} steps, blank-page baseline. Pulse columns cover {area:.1%} of each row.", "",
             "| Model | Pages / model | Logit | ECGs | Pulse share [95% CI] | Waveform share |", "|---|---|---|---|---|---:|"]
    lines += [f"| {s['model']} | {s['condition']} | {s['logit']} | {s['ecgs']} (n={s['n']}) | {s['pulse_share_mean']:.2%} "
              f"[{s['pulse_share_ci_lower']:.2%}, {s['pulse_share_ci_upper']:.2%}] | {s['waveform_share_mean']:.2%} |" for s in summary]
    lines += ["", "![HYP column profiles](hyp_column_profiles.png)", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
