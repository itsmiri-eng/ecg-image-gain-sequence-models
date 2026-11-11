"""Immutable, lazy image inputs for the five-label PTB-XL experiment.

Only generated PNG pixels are model inputs. Twelve complete lead rows are
inverted to ink and reduced by 2x2 maximum pooling; this retains every source
pixel's opportunity to contribute, including calibration, but thickens strokes
and loses detail. This is image preprocessing, not waveform reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/derived/ptbxl_images_v1"
DEFAULT_CACHE = ROOT / "data/processed/ptbxl_lead_images_v1"
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
LABEL_COLUMNS = tuple(f"label_{name}" for name in CLASS_NAMES)
INPUT_SHAPE = (12, 120, 1120)
PAGE_SHAPE = (3024, 2240)
EXPECTED_COUNTS = {"train": 17084, "validation": 2146, "test": 2158}
EXPECTED_SOURCE_RECORDS = 21799
PREPROCESSING = {
    "version": "ptbxl_full_lead_ink_maxpool2_v1",
    "source_page_shape": list(PAGE_SHAPE), "row_crop_y": [80, 2960],
    "row_crop_x": [0, 2240], "source_lead_shape": [12, 240, 2240],
    "input_shape": list(INPUT_SHAPE), "ink_transform": "255 - uint8 grayscale",
    "reduction": "Nonoverlapping 2x2 maximum ink pooling; no threshold or filter",
    "cache_storage_shape": [1440, 1120], "png_compression": 6,
    "lead_order": ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "calibration": "Every complete lead row, including its 1-mV calibration pulse, is retained; page header/footer omitted",
    "limitations": "Raster image representation; not native waveforms. Pooling thickens strokes and loses subpixel morphology. Original page-wide gain varies and low-gain detail may disappear. No numeric source metadata is fed to the model.",
}


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha(path):
    return _sha_bytes(Path(path).read_bytes())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as handle:
        handle.write(payload)


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Missing or duplicate CSV column names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    return rows


def _strict_bool(value):
    if value not in ("True", "False"):
        raise ValueError(f"Expected canonical CSV boolean True/False, received {value!r}")
    return value == "True"


def _positive_int(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"Invalid canonical positive integer {field}: {value!r}")
    return int(value)


def _check_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA-256 value")


def _parse_rows(raw_rows, cached=False):
    rows, ids, patient_folds = [], set(), defaultdict(set)
    for raw in raw_rows:
        row = dict(raw)
        for field in ("ecg_id", "official_fold", "width", "height", "image_bytes"):
            row[field] = _positive_int(row[field], field)
        _positive_int(row["patient_id"], "patient_id")
        for field in (*LABEL_COLUMNS, "eligible_for_classification"):
            row[field] = _strict_bool(row[field])
        labels = [name for name, column in zip(CLASS_NAMES, LABEL_COLUMNS) if row[column]]
        if (not row["eligible_for_classification"] or not labels or row["exclusion_reason"]
                or json.loads(row["labels_json"]) != labels):
            raise ValueError("Invalid classification eligibility or multilabel target mapping")
        fold = row["official_fold"]
        if not 1 <= fold <= 10:
            raise ValueError("Invalid official fold")
        split = "train" if fold <= 8 else "validation" if fold == 9 else "test"
        if row["split"] != split:
            raise ValueError("Split does not match official patient fold")
        if row["ecg_id"] in ids:
            raise ValueError("Duplicate ECG identifier")
        ids.add(row["ecg_id"])
        patient_folds[row["patient_id"]].add(fold)
        if (row["height"], row["width"]) != PAGE_SHAPE or row["clipped_point_count"] != "0":
            raise ValueError("Source page dimensions or clipping differ from frozen renderer")
        expected = f"images/{split}/ecg_{row['ecg_id']:05d}.png"
        if row["image_path"] != expected:
            raise ValueError("Unsafe or unexpected source image path")
        for field in ("image_sha256", "pixel_sha256"):
            _check_hash(row[field])
        if cached:
            if row["cache_path"] != f"cachedimages/{split}/ecg_{row['ecg_id']:05d}.png":
                raise ValueError("Unsafe or unexpected cache image path")
            _check_hash(row["cache_sha256"])
            _check_hash(row["cache_pixel_sha256"])
        rows.append(row)
    if any(len(folds) != 1 for folds in patient_folds.values()):
        raise ValueError("Patient crosses official folds")
    if dict(Counter(row["split"] for row in rows)) != EXPECTED_COUNTS:
        raise ValueError("Eligible cohort size or split counts differ from frozen PTB-XL manifest")
    if [row["ecg_id"] for row in rows] != sorted(ids):
        raise ValueError("ECG rows must retain numeric source order")
    return rows


def load_source_rows(dataset_root=DEFAULT_SOURCE):
    """Read the verified 21,388-row image manifest with strict typed targets."""
    root = Path(dataset_root)
    verification = _read_json(root / "verification.json")
    state = _read_json(root / "generation_state.json")
    summary = _read_json(root / "generation_summary.json")
    protocol = _read_json(root / "protocol.json")
    required_flags = ("all_output_files_accounted_for", "all_source_records_accounted_for",
                      "manifest_hashes_verified", "official_patient_grouping_verified",
                      "classification_target_mapping_verified")
    if (verification.get("status") != "passed" or verification.get("failures") != []
            or verification.get("pages_decoded_and_hashed") != EXPECTED_SOURCE_RECORDS
            or any(verification.get(flag) is not True for flag in required_flags)
            or verification.get("duplicates_crossing_partitions") != []
            or state.get("status") != "complete" or state.get("images") != EXPECTED_SOURCE_RECORDS):
        raise ValueError("Source rendering cohort has not completed full verification")
    if (summary["binding"] != protocol["binding"] or summary["images"] != EXPECTED_SOURCE_RECORDS
            or summary["classification_eligible"] != sum(EXPECTED_COUNTS.values())
            or summary["eligible_partition_counts"] != EXPECTED_COUNTS):
        raise ValueError("Source summary and frozen protocol disagree")
    cfg = protocol["configuration"]
    if (cfg["class_names"] != list(CLASS_NAMES) or cfg["pixels_per_mm"] != 8
            or cfg["lead_order"] != PREPROCESSING["lead_order"]
            or [cfg[key] for key in ("header_mm", "row_pitch_mm", "footer_mm", "left_margin_mm", "right_margin_mm")]
            != [10, 30, 8, 25, 5] or cfg["samples_per_lead"] != 5000
            or cfg["sampling_rate_hz"] != 500 or cfg["speed_mm_per_second"] != 25):
        raise ValueError("Source render configuration is incompatible with fixed lead geometry")
    hashes = summary["manifest_sha256"]
    required_manifests = {"all_images.csv", "classification_manifest.csv", "classification_exclusions.csv",
                          "splits/train.csv", "splits/validation.csv", "splits/test.csv"}
    if set(hashes) != required_manifests:
        raise ValueError("Missing or unexpected verified source manifest")
    for name, digest in hashes.items():
        _check_hash(digest)
        if _sha(root / name) != digest:
            raise ValueError(f"Source manifest hash mismatch: {name}")
    return _parse_rows(_read_csv(root / "classification_manifest.csv"))


def preprocess_page(page):
    """3024x2240 grayscale -> twelve complete 120x1120 uint8 ink images."""
    if not isinstance(page, np.ndarray) or page.dtype != np.uint8 or page.shape != PAGE_SHAPE:
        raise ValueError("Expected 3024x2240 uint8 grayscale generated page")
    ink = 255 - page[80:2960, :]
    leads = ink.reshape(12, 120, 2, 1120, 2)
    return np.ascontiguousarray(leads.max(axis=(2, 4)), dtype=np.uint8)


def _source_binding(source_root):
    return {
        "source_root": str(Path(source_root).resolve()),
        "source_files_sha256": {name: _sha(Path(source_root) / name) for name in
                                ("classification_manifest.csv", "protocol.json", "generation_summary.json",
                                 "verification.json", "generation_state.json")},
        "preprocessing": PREPROCESSING, "class_names": list(CLASS_NAMES),
        "code_sha256": _sha(Path(__file__)),
        "runtime": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__},
    }


def _cache_job(job):
    row, source_root, output = job
    cv2.setNumThreads(1)
    payload = (Path(source_root) / row["image_path"]).read_bytes()
    if _sha_bytes(payload) != row["image_sha256"]:
        raise ValueError(f"Source image checksum mismatch: ECG {row['ecg_id']}")
    page = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if page is None or _sha_bytes(page.tobytes()) != row["pixel_sha256"]:
        raise ValueError(f"Source pixel checksum/decode mismatch: ECG {row['ecg_id']}")
    leads = preprocess_page(page)
    stack = leads.reshape(1440, 1120)
    ok, encoded = cv2.imencode(".png", stack, [cv2.IMWRITE_PNG_COMPRESSION, PREPROCESSING["png_compression"]])
    if not ok:
        raise RuntimeError("Cache PNG encoding failed")
    cache_payload = encoded.tobytes()
    relative = f"cachedimages/{row['split']}/ecg_{row['ecg_id']:05d}.png"
    destination = Path(output) / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(cache_payload)
    result = {"row": {**row, "cache_path": relative, "cache_sha256": _sha_bytes(cache_payload),
                      "cache_pixel_sha256": _sha_bytes(stack.tobytes())}, "train_pixels": 0}
    if row["split"] == "train":
        mean, std = cv2.meanStdDev(stack)
        normalized_mean, normalized_std = float(mean[0, 0]) / 255, float(std[0, 0]) / 255
        result.update(train_pixels=stack.size, train_sum=normalized_mean * stack.size,
                      train_sum_squares=(normalized_std ** 2 + normalized_mean ** 2) * stack.size)
    return result


def _training_normalization(results):
    training = [item for item in results if item["row"]["split"] == "train"]
    if not training or any(item["train_pixels"] != math.prod(INPUT_SHAPE) for item in training):
        raise ValueError("Missing training-only pixel statistics")
    count = sum(item["train_pixels"] for item in training)
    mean = math.fsum(item["train_sum"] for item in training) / count
    second = math.fsum(item["train_sum_squares"] for item in training) / count
    raw_std = math.sqrt(max(0, second - mean * mean))
    return {"method": "global training-only ink mean/std after division by 255", "fit_partition": "train",
            "mean": mean, "std": max(raw_std, 1e-6), "raw_std": raw_std, "std_floor": 1e-6,
            "training_images": len(training), "training_pixels": count}


def _validate_cache(cache_root, source_rows=None):
    root = Path(cache_root)
    metadata, protocol = _read_json(root / "metadata.json"), _read_json(root / "protocol.json")
    if (metadata.get("status") != "complete" or metadata.get("input_shape") != list(INPUT_SHAPE)
            or metadata.get("binding") != protocol.get("binding")):
        raise ValueError("Incomplete or incompatible learning cache")
    binding = protocol["binding"]
    if binding != _source_binding(binding["source_root"]):
        raise ValueError("Cache code, source, configuration or runtime binding changed")
    if _sha(root / "cache_manifest.csv") != metadata["cache_manifest_sha256"]:
        raise ValueError("Cache manifest hash mismatch")
    rows = _parse_rows(_read_csv(root / "cache_manifest.csv"), cached=True)
    source_rows = load_source_rows(binding["source_root"]) if source_rows is None else source_rows
    if [{key: row[key] for key in source} for row, source in zip(rows, source_rows)] != source_rows:
        raise ValueError("Cached identities/targets differ from verified source manifest")
    normalization = metadata["normalization"]
    if (normalization.get("fit_partition") != "train" or normalization.get("training_images") != EXPECTED_COUNTS["train"]
            or normalization.get("training_pixels") != EXPECTED_COUNTS["train"] * math.prod(INPUT_SHAPE)
            or not np.isfinite([normalization["mean"], normalization["std"]]).all()
            or normalization["std"] <= 0 or not 0 <= normalization["mean"] <= 1):
        raise ValueError("Invalid training normalization")
    return rows, metadata


def prepare_cache(output_dir=DEFAULT_CACHE, workers=4, dataset_root=DEFAULT_SOURCE):
    """Build once, or validate an existing complete cache; never resume partial data.

    Every source PNG is hashed before decoding. Partial/nonempty output requires
    a new cache directory, keeping failed output available for investigation.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    output, source_root = Path(output_dir), Path(dataset_root).resolve()
    rows = load_source_rows(source_root)
    binding = _source_binding(source_root)
    if output.exists() and any(output.iterdir()):
        if not (output / "metadata.json").exists():
            raise FileExistsError("Incomplete/nonempty cache: choose a new output directory")
        _, metadata = _validate_cache(output, rows)
        if metadata["binding"] != binding:
            raise ValueError("Existing cache belongs to a different source/protocol")
        return metadata
    output.mkdir(parents=True, exist_ok=True)
    _write_new_json(output / "protocol.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "binding": binding})
    results = []
    try:
        jobs = ((row, str(source_root), str(output)) for row in rows)
        if workers == 1:
            iterator = map(_cache_job, jobs)
            for item in iterator:
                results.append(item)
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                for item in pool.map(_cache_job, jobs, chunksize=8):
                    results.append(item)
                    if len(results) % 1000 == 0:
                        print(f"Cached {len(results)}/{len(rows)} PTB-XL image inputs", flush=True)
        if _source_binding(source_root) != binding:
            raise ValueError("Source/preprocessing binding changed during cache creation")
        cached_rows = [item["row"] for item in results]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(cached_rows[0]))
        writer.writeheader()
        writer.writerows(cached_rows)
        with (output / "cache_manifest.csv").open("xb") as handle:
            handle.write(buffer.getvalue().encode())
        metadata = {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "input_shape": list(INPUT_SHAPE), "counts": dict(Counter(row["split"] for row in rows)),
                    "images": len(rows), "class_names": list(CLASS_NAMES), "binding": binding,
                    "normalization": _training_normalization(results),
                    "source_manifest_sha256": binding["source_files_sha256"]["classification_manifest.csv"],
                    "cache_manifest_sha256": _sha(output / "cache_manifest.csv"),
                    "source_png_hashes_checked_before_decoding": len(rows),
                    "cache_integrity_policy": "Encoded and decoded pixel hashes checked on every Dataset access",
                    "model_input_metadata": "None; only cached PNG pixels are model inputs",
                    "preprocessing": PREPROCESSING}
        _write_new_json(output / "metadata.json", metadata)
        return metadata
    except Exception as exc:
        _write_new_json(output / "failure.json", {"status": "failed", "error": repr(exc),
                        "successful_records_returned": len(results), "retry_policy": "Use a new cache directory"})
        raise


class CachedECGDataset(Dataset):
    """Lazily decode/hash one image; return float32 (12,120,1120), five targets."""
    def __init__(self, cache_root=DEFAULT_CACHE, split="train", normalize=True):
        if split not in EXPECTED_COUNTS:
            raise ValueError("split must be train, validation, or test")
        self.cache_root = Path(cache_root)
        rows, self.metadata = _validate_cache(self.cache_root)
        self.rows = [row for row in rows if row["split"] == split]
        self.normalization = self.metadata["normalization"]
        self.normalize = bool(normalize)
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = (self.cache_root / row["cache_path"]).read_bytes()
        if _sha_bytes(payload) != row["cache_sha256"]:
            raise ValueError(f"Cache encoded checksum mismatch: ECG {row['ecg_id']}")
        stack = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
        if stack is None or stack.dtype != np.uint8 or stack.shape != (1440, 1120):
            raise ValueError("Wrong cache image shape/mode or decoding failure")
        if _sha_bytes(stack.tobytes()) != row["cache_pixel_sha256"]:
            raise ValueError(f"Cache pixel checksum mismatch: ECG {row['ecg_id']}")
        image = stack.reshape(INPUT_SHAPE).astype(np.float32) / 255
        if self.normalize:
            image -= self.normalization["mean"]
            image /= self.normalization["std"]
        targets = np.asarray([row[column] for column in LABEL_COLUMNS], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = prepare_cache(args.output, args.workers, args.source)
    print(json.dumps({key: summary[key] for key in ("status", "images", "counts", "normalization")}, indent=2))
