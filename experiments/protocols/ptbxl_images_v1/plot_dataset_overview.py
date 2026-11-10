"""Plot frozen PTB-XL cohort counts; does not read images or model results.

Reproduce: /opt/anaconda3/bin/python experiments/protocols/ptbxl_images_v1/plot_dataset_overview.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ecg-ptbxl-figure-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COUNTS = Path(__file__).with_name("label_counts.json")
DEFAULT_OUTPUT = ROOT / "results/ptbxl_images_v1"


def plot(counts_path=DEFAULT_COUNTS, output=DEFAULT_OUTPUT):
    counts_path, output = Path(counts_path), Path(output)
    data_bytes = counts_path.read_bytes()
    data = json.loads(data_bytes)
    source_hash = hashlib.sha256(data_bytes).hexdigest()
    splits = ("train", "validation", "test")
    split_values = [data["partitions"][split]["classification_eligible_records"] for split in splits]
    classes = ("NORM", "MI", "STTC", "CD", "HYP")
    label_values = [data["positive_label_counts"][label] for label in classes]
    assert sum(split_values) == data["classification_eligible_records"]
    assert data["classification_eligible_records"] + data["classification_ineligible_records"] == data["all_source_records"]
    assert all(data["positive_label_counts"][label] == sum(data["partitions"][split]["positive_label_counts"][label] for split in splits) for label in classes)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "semibold", "axes.labelsize": 11,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#617083", "axes.labelcolor": "#263445",
        "text.color": "#17283A", "xtick.color": "#35485C", "ytick.color": "#35485C",
        "svg.fonttype": "none", "svg.hashsalt": "ptbxl_images_v1_dataset_overview",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 7.0), gridspec_kw={"width_ratios": [1.0, 1.18]})
    figure.subplots_adjust(left=.08, right=.95, bottom=.28, top=.76, wspace=.65)
    figure.suptitle("PTB-XL image cohort", x=.08, y=.965, ha="left", fontsize=22, fontweight="semibold")
    figure.text(.08, .895,
                f"{data['all_source_records']:,} source records  ·  {data['classification_eligible_records']:,} classification-eligible images",
                fontsize=12.5, color="#435469")

    left, right = axes
    bars = left.bar(range(3), split_values, width=.61, color="#31587A", zorder=3)
    left.bar_label(bars, labels=[f"{n:,}" for n in split_values], padding=6, fontsize=12, fontweight="semibold")
    left.set_xticks(range(3), ["Train\nFolds 1–8", "Validation\nFold 9", "Test\nFold 10"])
    left.tick_params(axis="x", length=0, pad=9)
    left.set_ylim(0, 20500)
    left.set_ylabel("Eligible images", labelpad=8)
    left.set_title("(a) Official patient partitions", loc="left", pad=17)
    left.yaxis.set_major_locator(MultipleLocator(5000))
    left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    left.grid(axis="y", color="#E1E6EB", linewidth=.8, zorder=0)

    bars = right.barh(range(5), label_values, height=.58, color="#178278", zorder=3)
    right.bar_label(bars, labels=[f"{n:,}" for n in label_values], padding=6, fontsize=11.5, fontweight="semibold")
    right.set_yticks(range(5), ["NORM\nNormal ECG", "MI\nMyocardial infarction", "STTC\nST/T changes", "CD\nConduction disturbance", "HYP\nHypertrophy"])
    right.invert_yaxis()
    right.tick_params(axis="y", length=0, pad=10)
    right.set_xlim(0, 11000)
    right.set_xlabel("Images positive for each label", labelpad=9)
    right.set_title("(b) Diagnostic superclass labels", loc="left", pad=17)
    right.xaxis.set_major_locator(MultipleLocator(2500))
    right.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    right.grid(axis="x", color="#E1E6EB", linewidth=.8, zorder=0)
    right.spines["left"].set_visible(False)

    figure.text(.08, .154, "Label counts overlap: an ECG can contribute to more than one diagnostic class.",
                fontsize=11.5, fontweight="semibold")
    figure.text(.08, .106,
                f"{data['classification_ineligible_records']:,} records have no mapped diagnostic target: retained as images, excluded from classification.",
                fontsize=11.2)
    figure.text(.08, .052,
                "PTB-XL v1.0.3 · All present diagnostic SCP keys; no confidence threshold · Official patient folds preserved",
                fontsize=9.4, color="#596A7A")
    description = (f"Frozen source: {counts_path.name}; SHA-256 {source_hash}. "
                   "Panel A: eligible train/validation/test counts. Panel B: overlapping diagnostic superclass positives. "
                   f"{data['classification_ineligible_records']} unmapped records excluded from classification but retained as images.")
    output.mkdir(parents=True, exist_ok=True)
    png = output / "dataset_overview.png"
    svg = output / "dataset_overview.svg"
    figure.savefig(png, dpi=220, metadata={"Title": "PTB-XL image cohort", "Description": description})
    figure.savefig(svg, metadata={"Title": "PTB-XL image cohort", "Description": description, "Date": None})
    plt.close(figure)
    print(json.dumps({"png": str(png), "svg": str(svg), "counts_sha256": source_hash,
                      "eligible_partition_counts": dict(zip(splits, split_values)),
                      "overlapping_label_counts": dict(zip(classes, label_values))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    plot(arguments.counts, arguments.output)
