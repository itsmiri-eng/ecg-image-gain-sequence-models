"""Fixed-gain PTB-XL rendering and learning cache (dataset ptbxl_images_fixedgain2p5_v1).

Usage:
    python -m src.ptbxl_fixed_gain build --workers 3
    python -m src.ptbxl_fixed_gain verify

Protocol: docs/ptbxl_fixed_gain_protocol.md. Identical to the v1 renderer except
that every page is drawn at one fixed gain (2.5 mm/mV). Points beyond a lead's
30-mm row are clipped to the row edge and counted; no ECG is excluded, so the
cohort, labels and official patient folds equal the v1 cache exactly. The v1
renderer module is reused unchanged (its hash binds the frozen v1 dataset), and
only the gain policy in `fixed_gain_geometry` differs. Only the cached model
inputs and a small training-only preview of full pages are written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import ptbxl_rendering
from src.common import ROOT, read_json, sha256_file, utc_now
from src.ptbxl_learning_data import CLASS_NAMES, INPUT_SHAPE, LABEL_COLUMNS, preprocess_page

V1_CACHE = ROOT / "data/processed/ptbxl_lead_images_v1"
OUT = ROOT / "data/processed/ptbxl_lead_images_fixedgain2p5_v1"
SOURCE = ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
FIXED_GAIN = 2.5
DATASET_ID = "ptbxl_images_fixedgain2p5_v1"
CODE_FILES = ("src/ptbxl_fixed_gain.py", "src/ptbxl_rendering.py", "src/ptbxl_learning_data.py")
PREVIEW_TRAIN_ECGS = 6  # first training ECGs whose v1 gain differed from 2.5 mm/mV


def fixed_gain_geometry(signal, config):
    """v1 geometry with gain fixed at FIXED_GAIN; clip to the lead's row instead of reducing gain."""
    signal = np.asarray(signal, dtype=np.float64)
    ppm, pitch = config["pixels_per_mm"], config["row_pitch_mm"]
    gain = FIXED_GAIN
    duration = len(signal) / config["sampling_rate_hz"]
    width = round((config["left_margin_mm"] + duration * config["speed_mm_per_second"] + config["right_margin_mm"]) * ppm)
    height = round((config["header_mm"] + 12 * pitch + config["footer_mm"]) * ppm)
    x = np.rint((config["left_margin_mm"] + np.arange(len(signal)) / config["sampling_rate_hz"]
                 * config["speed_mm_per_second"]) * ppm).astype(np.int32)
    baselines = (config["header_mm"] + (np.arange(12) + 0.5) * pitch) * ppm
    points, bounds, clipped = [], [], 0
    for i in range(12):
        row_top = (config["header_mm"] + i * pitch) * ppm
        row_bottom = row_top + pitch * ppm
        y_float = baselines[i] - signal[:, i] * gain * ppm
        outside = (y_float < row_top) | (y_float > row_bottom - 1)
        clipped += int(np.count_nonzero(outside))
        y = np.rint(np.clip(y_float, row_top, row_bottom - 1)).astype(np.int32)
        points.append(np.column_stack((x, y)))
        bounds.append({"lead": ptbxl_rendering.LEADS[i], "baseline_y_pixels": float(baselines[i]),
                       "row_top_y_pixels": float(row_top), "row_bottom_y_pixels": float(row_bottom)})
    maximum = float(np.abs(signal).max())
    return points, {"width": width, "height": height, "gain_mm_per_mv": gain, "duration_seconds": duration,
                    "last_sample_time_seconds": (len(signal) - 1) / config["sampling_rate_hz"],
                    "maximum_absolute_mv": maximum, "clipped_point_count": clipped,
                    "flat_leads": [ptbxl_rendering.LEADS[i] for i in range(12) if np.ptp(signal[:, i]) == 0],
                    "lead_panels": bounds}


def render_fixed_gain(signal, config):
    """Run the unchanged v1 page drawing with the fixed-gain geometry substituted."""
    original = ptbxl_rendering.geometry
    ptbxl_rendering.geometry = fixed_gain_geometry
    try:
        image, meta = ptbxl_rendering.render_page(signal, config)
    finally:
        ptbxl_rendering.geometry = original
    return image, meta


def _job(task):
    row, config, out = task
    cv2.setNumThreads(1)
    base = SOURCE / row["filename_hr"]
    for suffix, key in ((".hea", "source_header_sha256"), (".dat", "source_waveform_sha256")):
        if sha256_file(f"{base}{suffix}") != row[key]:
            raise ValueError(f"Source checksum changed: ECG {row['ecg_id']}")
    signal, _ = ptbxl_rendering.load_waveform(base, config)
    page, meta = render_fixed_gain(signal, config)
    stack = preprocess_page(page).reshape(1440, 1120)
    ok, encoded = cv2.imencode(".png", stack, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    payload = encoded.tobytes()
    relative = f"cachedimages/{row['split']}/ecg_{int(row['ecg_id']):05d}.png"
    destination = Path(out) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    page_payload = None
    if row.get("_preview") == "1":
        ok, page_encoded = cv2.imencode(".png", page, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        page_payload = page_encoded.tobytes()
    result = {"ecg_id": row["ecg_id"], "cache_path": relative, "cache_sha256": hashlib.sha256(payload).hexdigest(),
              "cache_pixel_sha256": hashlib.sha256(stack.tobytes()).hexdigest(),
              "gain_mm_per_mv": FIXED_GAIN, "v1_gain_mm_per_mv": row["gain_mm_per_mv"],
              "clipped_point_count": meta["clipped_point_count"], "maximum_absolute_mv": meta["maximum_absolute_mv"],
              "page_payload": page_payload}
    if row["split"] == "train":
        mean, std = cv2.meanStdDev(stack)
        m, s = float(mean[0, 0]) / 255, float(std[0, 0]) / 255
        result.update(train_sum=m * stack.size, train_sum_squares=(s * s + m * m) * stack.size)
    return result


def build(workers=3):
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"{OUT} is not empty; never overwrite a built cache")
    started = time.perf_counter()
    config = read_json(ROOT / "experiments/configs/ptbxl_images_v1.json")
    with (V1_CACHE / "cache_manifest.csv").open(newline="") as handle:
        v1_rows = list(csv.DictReader(handle))
    previews = 0
    for row in v1_rows:
        if row["split"] == "train" and float(row["gain_mm_per_mv"]) != FIXED_GAIN and previews < PREVIEW_TRAIN_ECGS:
            row["_preview"], previews = "1", previews + 1
    OUT.mkdir(parents=True)
    protocol = {"created_utc": utc_now(), "dataset_id": DATASET_ID, "fixed_gain_mm_per_mv": FIXED_GAIN,
                "protocol_document": "docs/ptbxl_fixed_gain_protocol.md",
                "protocol_document_sha256": sha256_file(ROOT / "docs/ptbxl_fixed_gain_protocol.md"),
                "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_FILES},
                "v1_cache_manifest_sha256": sha256_file(V1_CACHE / "cache_manifest.csv"),
                "render_configuration": {**config, "gain_policy": f"Fixed {FIXED_GAIN} mm/mV for every page; "
                                         "points beyond the 30-mm lead row are clipped to its edge and counted"},
                "test_information_used_for_choices": False}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tasks = [(row, config, str(OUT)) for row in v1_rows]
    results = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(_job, tasks, chunksize=16):
            if result["page_payload"] is not None:
                (OUT / "preview_train_pages").mkdir(exist_ok=True)
                (OUT / f"preview_train_pages/ecg_{int(result['ecg_id']):05d}.png").write_bytes(result["page_payload"])
            results.append(result)
            if len(results) % 2000 == 0:
                print(f"{utc_now()} rendered {len(results)}/{len(tasks)}", flush=True)
    by_id = {r["ecg_id"]: r for r in results}
    keep = [c for c in v1_rows[0] if c not in ("_preview",)]
    rows = []
    for row in v1_rows:
        r = by_id[row["ecg_id"]]
        new = {key: row[key] for key in keep}
        new.update(cache_path=r["cache_path"], cache_sha256=r["cache_sha256"], cache_pixel_sha256=r["cache_pixel_sha256"],
                   gain_mm_per_mv=FIXED_GAIN, clipped_point_count=r["clipped_point_count"])
        new["v1_gain_mm_per_mv"] = r["v1_gain_mm_per_mv"]
        rows.append(new)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (OUT / "cache_manifest.csv").write_bytes(buffer.getvalue().encode())
    train = [r for r in results if "train_sum" in r]
    count = len(train) * math.prod(INPUT_SHAPE)
    mean = math.fsum(r["train_sum"] for r in train) / count
    std = math.sqrt(max(0.0, math.fsum(r["train_sum_squares"] for r in train) / count - mean * mean))
    clipped = [r for r in rows if int(r["clipped_point_count"]) > 0]
    metadata = {"status": "complete", "completed_utc": utc_now(), "dataset_id": DATASET_ID,
                "input_shape": list(INPUT_SHAPE), "images": len(rows),
                "counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
                "normalization": {"method": "global training-only ink mean/std after division by 255",
                                  "fit_partition": "train", "mean": mean, "std": max(std, 1e-6),
                                  "training_images": len(train), "training_pixels": count},
                "pages_with_clipping": {s: sum(r["split"] == s for r in clipped) for s in ("train", "validation", "test")},
                "cache_manifest_sha256": sha256_file(OUT / "cache_manifest.csv"),
                "build_seconds": time.perf_counter() - started, "workers": workers}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("images", "counts", "normalization", "pages_with_clipping", "build_seconds")}, indent=2))
    return metadata


def verify():
    """Cohort identity with v1, hashes of every cached image, and identical pixels for unchanged-gain pages."""
    metadata = read_json(OUT / "metadata.json")
    if sha256_file(OUT / "cache_manifest.csv") != metadata["cache_manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    with (OUT / "cache_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (V1_CACHE / "cache_manifest.csv").open(newline="") as handle:
        v1 = list(csv.DictReader(handle))
    identity = ("ecg_id", "patient_id", "official_fold", "split", *LABEL_COLUMNS)
    if [{k: r[k] for k in identity} for r in rows] != [{k: r[k] for k in identity} for r in v1]:
        raise ValueError("Cohort, folds or labels differ from v1")
    same_gain_identical, same_gain = 0, 0
    for row, old in zip(rows, v1):
        payload = (OUT / row["cache_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["cache_sha256"]:
            raise ValueError(f"Cache hash mismatch: ECG {row['ecg_id']}")
        if float(old["gain_mm_per_mv"]) == FIXED_GAIN:
            same_gain += 1
            same_gain_identical += row["cache_pixel_sha256"] == old["cache_pixel_sha256"]
    report = {"status": "passed" if same_gain_identical == same_gain else "failed", "checked_utc": utc_now(),
              "images_hashed": len(rows), "cohort_folds_labels_identical_to_v1": True,
              "v1_pages_already_at_fixed_gain": same_gain, "of_which_pixel_identical_to_v1": same_gain_identical}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise ValueError("Unchanged-gain pages are not pixel-identical to v1")
    return report


class FixedGainDataset(Dataset):
    """Same interface as CachedECGDataset for the fixed-gain cache (hash-checked on every access).

    `normalization` defaults to the fixed-gain training statistics; transfer evaluation passes a
    model's original v1 normalization instead.
    """
    def __init__(self, cache_root=OUT, split="train", normalize=True, normalization=None):
        self.cache_root, self.split, self.normalize = Path(cache_root), split, bool(normalize)
        self.metadata = read_json(self.cache_root / "metadata.json")
        if self.metadata.get("status") != "complete" or sha256_file(self.cache_root / "cache_manifest.csv") != self.metadata["cache_manifest_sha256"]:
            raise ValueError("Fixed-gain cache incomplete or manifest changed")
        self.normalization = normalization or self.metadata["normalization"]
        with (self.cache_root / "cache_manifest.csv").open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["split"] == split]
        for row in rows:
            for column in LABEL_COLUMNS:
                row[column] = row[column] == "True"
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = (self.cache_root / row["cache_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["cache_sha256"]:
            raise ValueError(f"Cache checksum mismatch: ECG {row['ecg_id']}")
        stack = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
        image = stack.reshape(INPUT_SHAPE).astype(np.float32) / 255
        if self.normalize:
            image -= self.normalization["mean"]
            image /= self.normalization["std"]
        targets = np.asarray([row[column] for column in LABEL_COLUMNS], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["build", "verify"])
    parser.add_argument("--workers", type=int, default=3)
    arguments = parser.parse_args()
    build(arguments.workers) if arguments.mode == "build" else verify()
