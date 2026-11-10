"""Recheck frozen publication data without rewriting any prior artifact.

Requires an explicit, new JSON output path. ``metadata`` validates bindings,
labels, inventories and the small Mendeley input cache; ``full`` additionally
decodes/hashes every PTB-XL page/cache PNG, checks 500-Hz waveform/header bytes,
and independently recomputes training-only normalization. Neither mode fits,
loads, scores or runs a model. The old generator's mutating verify() is never
called because verification.json is itself an immutable downstream binding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from src.common import ROOT, read_json, sha256_file
from src.generate_ptbxl_images import CONFIG_PATH, DEFAULT_OUT, _binding, _csv_bytes, _verify_job
from src.ptbxl_image_manifest import prepare_manifest
from src.ptbxl_learning_data import DEFAULT_CACHE, _validate_cache, preprocess_page
from src.verify import load_verified_data


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _duplicate_summary(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    return {"duplicate_groups": len(duplicates),
            "redundant_records": sum(len(group) - 1 for group in duplicates),
            "groups_crossing_splits": sum(len({r["split"] for r in group}) > 1 for group in duplicates)}


def _cache_record(row, cache_root):
    """Check one existing encoded/decoded cache file; return train-only moments."""
    payload = (Path(cache_root) / row["cache_path"]).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == row["cache_sha256"], "Cache encoded hash mismatch")
    pixels = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    _require(pixels is not None and pixels.dtype == np.uint8 and pixels.shape == (1440, 1120),
             "Cache shape/mode/decode mismatch")
    _require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["cache_pixel_sha256"],
             "Cache decoded-pixel hash mismatch")
    result = {"split": row["split"], "ecg_id": row["ecg_id"]}
    if row["split"] == "train":
        mean, std = cv2.meanStdDev(pixels)
        mean, std = float(mean[0, 0]) / 255, float(std[0, 0]) / 255
        result.update(pixels=pixels.size, total=mean * pixels.size,
                      squared_total=(std ** 2 + mean ** 2) * pixels.size)
    return result


def _training_moments(checks):
    train = [row for row in checks if row["split"] == "train"]
    _require(bool(train), "No training normalization observations")
    count = sum(row["pixels"] for row in train)
    mean = math.fsum(row["total"] for row in train) / count
    std = math.sqrt(max(0, math.fsum(row["squared_total"] for row in train) / count - mean ** 2))
    return {"fit_partition": "train", "images": len(train), "pixels": count, "mean": mean, "std": std}


def _full_ptbxl_scan(source, all_rows, cached, metadata, protocol, workers):
    """Use existing read-only verification helpers, never generator.verify()."""
    manifest_by_id = {int(row["ecg_id"]): row for row in all_rows}
    source_root = ROOT / protocol["configuration"]["source_directory"]
    jobs = ((row, str(DEFAULT_OUT), str(source_root), protocol["binding"]["configuration_sha256"],
             {row["filename_hr"] + ext: source.expected_checksums[row["filename_hr"] + ext]
              for ext in (".hea", ".dat")}, manifest_by_id[row["ecg_id"]]["provenance_sha256"])
            for row in source.rows)
    verified = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_verify_job, jobs), 1):
            _require(result["ok"], f"Source verification failed: {result}")
            verified.append(result["manifest_row"])
            if index % 4000 == 0:
                print(f"Source pages verified: {index}/{len(source.rows)}", flush=True)
    eligible = [row for row in verified if row["eligible_for_classification"]]
    expected = {"all_images.csv": verified, "classification_manifest.csv": eligible,
                "classification_exclusions.csv": [row for row in verified if not row["eligible_for_classification"]]}
    expected.update({f"splits/{split}.csv": [row for row in eligible if row["split"] == split]
                     for split in ("train", "validation", "test")})
    for name, rows in expected.items():
        _require((DEFAULT_OUT / name).read_bytes() == _csv_bytes(rows), f"Reconstructed manifest mismatch: {name}")
    checks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(lambda row: _cache_record(row, DEFAULT_CACHE), cached), 1):
            checks.append(result)
            if index % 4000 == 0:
                print(f"Cache PNGs verified: {index}/{len(cached)}", flush=True)
    moments = _training_moments(checks)
    saved = metadata["normalization"]
    _require(moments["images"] == saved["training_images"] and moments["pixels"] == saved["training_pixels"],
             "Training normalization cohort mismatch")
    _require(abs(moments["mean"] - saved["mean"]) < 1e-14 and abs(moments["std"] - saved["std"]) < 1e-14,
             "Training normalization differs from cached training pixels")
    train = [row for row in cached if row["split"] == "train"]
    probes = [train[0], train[len(train) // 2], train[-1]]
    for row in probes:
        page = cv2.imread(str(DEFAULT_OUT / row["image_path"]), cv2.IMREAD_UNCHANGED)
        actual = cv2.imread(str(DEFAULT_CACHE / row["cache_path"]), cv2.IMREAD_UNCHANGED)
        _require(np.array_equal(preprocess_page(page).reshape(1440, 1120), actual), "Preprocessing probe differs")
    return {"source_pages_decoded_and_hashed": len(verified), "source_sidecars_hashed": len(verified),
            "source_500hz_waveform_header_pairs_hashed": len(verified), "manifests_reconstructed": len(expected),
            "cache_images_decoded_and_hashed": len(checks), "train_normalization_recomputed": moments,
            "preprocessing_training_probe_ecg_ids": [row["ecg_id"] for row in probes]}


def _collect_report(mode, workers):
    cv2.setNumThreads(1)
    frozen = [DEFAULT_OUT / name for name in (
        "verification.json", "protocol.json", "generation_summary.json", "generation_state.json", "all_images.csv",
        "classification_manifest.csv", "classification_exclusions.csv", "splits/train.csv", "splits/validation.csv", "splits/test.csv")]
    frozen += [DEFAULT_CACHE / name for name in ("metadata.json", "protocol.json", "cache_manifest.csv")]
    frozen += [ROOT / name for name in ("data/splits/fixed_split.csv", "data/splits/protocol.json",
               "data/processed/inputs_v1.npz", "data/audit/source_manifest.csv", "data/audit/header_ids.csv",
               "data/audit/near_duplicate_links.csv")]
    before = {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen}
    source = prepare_manifest()
    cached, metadata = _validate_cache(DEFAULT_CACHE)
    protocol = read_json(DEFAULT_OUT / "protocol.json")
    _require(_binding(CONFIG_PATH, source) == protocol["binding"], "Renderer/source/runtime binding mismatch")
    all_rows = _rows(DEFAULT_OUT / "all_images.csv")
    _require([int(row["ecg_id"]) for row in all_rows] == [row["ecg_id"] for row in source.rows],
             "Source rows missing/duplicated/reordered")
    for saved, fresh in zip(all_rows, source.rows):
        _require(all(saved[key] == str(value) for key, value in fresh.items()), "Fresh source labels/identity mismatch")
    for directory, pattern, expected in (
        (DEFAULT_OUT / "images", "*.png", {row["image_path"] for row in all_rows}),
        (DEFAULT_OUT / "provenance", "*.json", {row["provenance_path"] for row in all_rows}),
        (DEFAULT_CACHE / "cachedimages", "*.png", {row["cache_path"] for row in cached}),
    ):
        _require({str(path.relative_to(directory.parent)) for path in directory.rglob(pattern)} == expected,
                 f"Missing or unexpected files: {directory}")
    splits = {split: [row for row in cached if row["split"] == split] for split in ("train", "validation", "test")}
    overlaps = {f"{a}_vs_{b}": {key: len({row[key] for row in splits[a]} & {row[key] for row in splits[b]})
                for key in ("ecg_id", "patient_id", "image_sha256", "pixel_sha256", "cache_sha256", "cache_pixel_sha256")}
                for a, b in combinations(splits, 2)}
    _require(not any(value for group in overlaps.values() for value in group.values()), "Cross-partition identity/pixel overlap")
    full_checks = None
    if mode == "full":
        full_checks = _full_ptbxl_scan(source, all_rows, cached, metadata, protocol, workers)
    x, y, rows, subsets, _ = load_verified_data(verify_originals=mode == "full")
    train_x = x[subsets["train"]]
    normalization = {"mean": float(train_x.mean(dtype=np.float64)), "std": float(train_x.std(dtype=np.float64))}
    for folder, run in (("cnn", "cnn_v1_seed42"), ("cnn_lstm", "cnn_lstm_image_v1_seed42")):
        saved = read_json(ROOT / f"results/{folder}/{run}/training_metadata.json")["normalization"]
        _require(all(abs(saved[key] - normalization[key]) < 1e-14 for key in normalization), "Mendeley normalization mismatch")
    for folder, run in (("lda", "lda_v1_seed42"), ("cnn", "cnn_v1_seed42"), ("cnn_lstm", "cnn_lstm_image_v1_seed42")):
        _require((ROOT / f"results/{folder}/{run}/split_used.csv").read_bytes() == (ROOT / "data/splits/fixed_split.csv").read_bytes(),
                 f"Mendeley saved run split changed: {run}")
    headers = {row["file_sha256"]: row["printed_id"] for row in _rows(ROOT / "data/audit/header_ids.csv")}
    groups = defaultdict(list)
    for row in rows:
        groups[headers[row["file_sha256"]]].append(row)
    _require(all(len({row["split"] for row in group}) == len({row["group_id"] for row in group}) == 1
                 for group in groups.values()), "Mendeley printed-ID proxy crosses partitions/groups")
    raw_count = None
    if mode == "full":
        raw = _rows(ROOT / "data/audit/source_manifest.csv")
        for row in raw:
            _require(sha256_file(ROOT / row["path"]) == row["file_sha256"], f"Mendeley raw image changed: {row['path']}")
        raw_count = len(raw)
    _require(before == {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen}, "Frozen evidence changed during audit")
    return {"status": "passed", "mode": mode,
            "scope": "Data integrity only; no model fitting/inference/evaluation, clinical adjudication, prior artifact writes, or full independent waveform rerender.",
            "ptbxl": {"source_metadata": source.summary, "cache_records": len(cached),
                       "source_and_cache_inventories_complete": True, "cross_partition_overlap": overlaps,
                       "source_pixel_duplicates_from_manifest": _duplicate_summary(all_rows, "pixel_sha256"),
                       "cache_pixel_duplicates_from_manifest": _duplicate_summary(cached, "cache_pixel_sha256"),
                       "full_file_checks_performed": mode == "full", "full_file_checks": full_checks,
                       "100hz_waveform_bytes_rechecked": False},
            "mendeley": {"retained_cache_records_verified": len(rows), "raw_source_file_hashes_rechecked": raw_count,
                         "counts": {split: dict(Counter(rows[i]["class_name"] for i in indices)) for split, indices in subsets.items()},
                         "train_normalization_recomputed": normalization, "two_saved_normalizations_match": True,
                         "three_saved_run_splits_match": True, "printed_id_proxy_groups": len(groups),
                         "mixed_folder_label_proxy_groups": sum(len({row["class_name"] for row in group}) > 1 for group in groups.values()),
                         "patient_identity_verified": False},
            "protected_files_unchanged": before,
            "not_checked": ["Model outputs or checkpoint validity", "Clinical label validity", "Unrecognized patient relationships in Mendeley", "Near-duplicate waveform similarity", "100-Hz waveform bytes"]
                           + (["PTB-XL PNG contents, sidecar contents, 500-Hz waveform/header bytes, training-pixel normalization, Mendeley original bytes"] if mode == "metadata" else [])}


def run_audit(output, mode="metadata", workers=4):
    """Write only one explicit new JSON report; reject existing paths before work."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("Audit output must be a new .json file")
    if mode not in ("metadata", "full"):
        raise ValueError("mode must be metadata or full")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer from 1 through 8")
    started = time.monotonic()
    report = _collect_report(mode, workers)
    report.update(checked_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                  audit_code_sha256=sha256_file(Path(__file__)))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a path created while checks ran.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Explicit NEW JSON report path; existing paths are rejected")
    parser.add_argument("--mode", choices=("metadata", "full"), default="metadata", help="metadata does not rehash PTB-XL image contents")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    report = run_audit(args.output, args.mode, args.workers)
    print(json.dumps({"status": report["status"], "mode": report["mode"], "report": str(args.output.resolve()),
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
