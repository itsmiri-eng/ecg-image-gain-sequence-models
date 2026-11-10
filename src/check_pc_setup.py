"""Check that a copied project is complete and ready to train on this computer.

Usage (inside the copied project folder):
    python -X utf8 -m src.check_pc_setup [--device cuda|cpu] [--skip-files]

1. every file in TRANSFER_MANIFEST.json is present and unchanged (src.transfer verify);
2. Python 3.12, UTF-8 mode on Windows, and the package versions pinned in requirements.txt;
3. both image caches open with all their integrity checks;
4. the seed-42 hybrid ViT re-predicts the validation fold on this device; the largest difference
   from the probabilities saved on the Mac measures hardware drift at inference;
5. three training steps run twice from the same seed; the largest parameter difference between
   the repeats shows whether training on this device is bit-reproducible.
Validation and training data only; the test fold is never opened. The report is written to
experiments/setup_checks/<computer>_<UTC time>.json.
"""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import os
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.common import ROOT, utc_now, write_json
from src.portable_cache import PortableCachedECGDataset
from src.ptbxl_fixed_gain import FixedGainDataset
from src.training.ptbxl_trainer import CLASS_NAMES, MODELS, predict_proba, seed_everything
from src import transfer

VIT_RUN = "ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42"


def step(title):
    print(f"\n== {title}", flush=True)


def package_versions():
    problems, found = [], {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, pinned = line.strip().split("==")
        try:
            found[name] = version(name)
        except Exception:
            problems.append(f"{name} not installed (need {pinned})")
            continue
        if found[name].split("+")[0] != pinned:
            problems.append(f"{name} {found[name]} installed, protocol pins {pinned}")
    return found, problems


def training_repeat(device, dataset):
    """Parameters after three identical seeded steps, run twice."""
    finals = []
    for _ in range(2):
        seed_everything(42, 4)
        model = MODELS["hybrid_vit"](dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(Subset(dataset, range(48)), batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(42))
        model.train()
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images.to(device)), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        finals.append(torch.cat([p.detach().cpu().flatten() for p in model.parameters()]))
    return float((finals[0] - finals[1]).abs().max())


def main(device, skip_files):
    report = {"started_utc": utc_now(), "computer": platform.node(), "platform": platform.platform(),
              "python": sys.version, "project_root": str(ROOT), "problems": []}
    problems = report["problems"]

    step("1/5 Copied files")
    if skip_files:
        print("skipped")
    elif not (ROOT / transfer.MANIFEST).exists():
        problems.append("TRANSFER_MANIFEST.json missing: this folder was not made by `python -m src.transfer copy`")
    else:
        manifest, damaged = transfer.verify()
        problems += damaged[:50]
        report["files_checked"] = len(manifest["files"])
        print(f"{len(manifest['files']):,} files checked, {len(damaged)} problems")

    step("2/5 Python and packages")
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {platform.python_version()}; the project was verified on Python 3.12")
    if os.name == "nt" and not sys.flags.utf8_mode:
        problems.append("Windows without UTF-8 mode: run `setx PYTHONUTF8 1`, reopen the terminal, or use `python -X utf8`")
    report["packages"], package_problems = package_versions()
    problems += package_problems
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    report["device"] = device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but PyTorch sees no NVIDIA GPU (CPU-only PyTorch installed?)")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["cuda_runtime"] = torch.version.cuda
    else:
        problems.append("No NVIDIA GPU visible to PyTorch; training would run on the CPU")
    print(f"Python {platform.python_version()}, torch {torch.__version__}, device {device}"
          + (f" ({report['gpu']})" if device == "cuda" else ""))

    step("3/5 Image caches")
    validation = PortableCachedECGDataset(split="validation")
    train = PortableCachedECGDataset(split="train")
    fixed = FixedGainDataset(split="validation")
    for dataset in (validation, train, fixed):
        dataset[0], dataset[len(dataset) - 1]
    report["v1_cache_binding_check"] = validation.binding_check
    print(f"v1 cache: {len(train):,} train / {len(validation):,} validation ({validation.binding_check['mode']} check); "
          f"fixed-gain validation {len(fixed):,}")

    step("4/5 Saved model re-predicts validation on this device")
    checkpoint = torch.load(ROOT / "models" / VIT_RUN / "best.pt", map_location="cpu", weights_only=True)
    torch.manual_seed(0)
    model = MODELS["hybrid_vit"]()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    started = time.perf_counter()
    probabilities = predict_proba(model, validation, 16, f"check:{device}")
    with (ROOT / "results" / VIT_RUN / "validation_predictions.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    if [r["ecg_id"] for r in saved] != [str(r["ecg_id"]) for r in validation.rows]:
        problems.append("Validation order differs from the saved predictions")
    else:
        mac = np.asarray([[float(r[f"probability_{n}"]) for n in CLASS_NAMES] for r in saved])
        report["max_abs_probability_difference_vs_mac"] = float(np.abs(probabilities - mac).max())
        report["validation_inference_seconds"] = time.perf_counter() - started
        print(f"largest probability difference from the Mac: {report['max_abs_probability_difference_vs_mac']:.2e} "
              f"({report['validation_inference_seconds']:.0f} s)")
        if report["max_abs_probability_difference_vs_mac"] > 1e-3:
            problems.append("Saved model gives noticeably different probabilities on this computer (>1e-3)")

    step("5/5 Training steps, repeated")
    report["training_repeat_max_parameter_difference"] = training_repeat(device, train)
    print(f"largest parameter difference between two identical repeats: {report['training_repeat_max_parameter_difference']:.2e}"
          + (" (bit-reproducible)" if report["training_repeat_max_parameter_difference"] == 0 else
             " (not bit-reproducible on this device; repeated runs differ slightly, as expected on GPUs)"))

    report["completed_utc"] = utc_now()
    report["ready"] = not [p for p in problems if not p.startswith("No NVIDIA GPU")]
    name = f"{platform.node() or 'computer'}_{report['completed_utc'].replace(':', '')}.json"
    write_json(ROOT / "experiments/setup_checks" / name, report)
    print("\nREADY to train." if report["ready"] else "\nNOT READY:\n- " + "\n- ".join(problems))
    if problems and report["ready"]:
        print("Notes:\n- " + "\n- ".join(problems))
    print(f"Report: experiments/setup_checks/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--skip-files", action="store_true")
    arguments = parser.parse_args()
    main(arguments.device, arguments.skip_files)
