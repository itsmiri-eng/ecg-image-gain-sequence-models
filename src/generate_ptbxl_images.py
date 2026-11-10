"""Generate and verify the full, separately versioned PTB-XL image cohort."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import cv2
import numpy as np
import wfdb

from src.common import ROOT, read_json, sha256_file, utc_now, write_csv, write_json
from src.ptbxl_image_manifest import prepare_manifest
from src.ptbxl_rendering import LEADS, load_waveform, render_page, encode_png

CONFIG_PATH = ROOT / "experiments/configs/ptbxl_images_v1.json"
DEFAULT_OUT = ROOT / "data/derived/ptbxl_images_v1"
CODE_FILES = ["src/generate_ptbxl_images.py", "src/ptbxl_rendering.py", "src/ptbxl_image_manifest.py", "src/common.py"]


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False)+"\n").encode())


def _render_job(job):
    row, source, out, config, checksums, config_hash = job
    cv2.setNumThreads(1)
    source, out = Path(source), Path(out)
    image_relative = f"images/{row['split']}/ecg_{row['ecg_id']:05d}.png"
    sidecar_relative = f"provenance/{row['split']}/ecg_{row['ecg_id']:05d}.json"
    image_path, sidecar = out/image_relative, out/sidecar_relative
    try:
        source_hashes = {}
        for suffix in [".hea", ".dat"]:
            relative = row["filename_hr"] + suffix
            actual = sha256_file(source/relative)
            if actual != checksums[relative]:
                raise ValueError(f"Source waveform/header checksum changed: {relative}")
            source_hashes[relative] = actual
        if sidecar.exists():
            saved = read_json(sidecar)
            if (saved["source_record"] != row or saved["configuration_sha256"] != config_hash
                    or saved["source_sha256"] != source_hashes or saved["image_path"] != image_relative
                    or saved["sidecar_path"] != sidecar_relative
                    or sha256_file(image_path) != saved["image_sha256"]):
                raise ValueError("Existing page/provenance does not match the frozen generation protocol")
            return {"ok": True, "metadata": saved, "reused": True}
        signal, signal_metadata = load_waveform(source/row["filename_hr"], config)
        image, geometry = render_page(signal, config)
        payload = encode_png(image, config)
        image_hash = hashlib.sha256(payload).hexdigest()
        if image_path.exists():
            if sha256_file(image_path) != image_hash:
                raise ValueError("Unverified existing image differs from deterministic rendering")
        else:
            _atomic_bytes(image_path, payload)
        metadata = {"source_record": row, "image_path": image_relative,
                    "sidecar_path": sidecar_relative, "configuration_sha256": config_hash,
                    "source_sha256": source_hashes, "signal": signal_metadata, **geometry,
                    "image_sha256": image_hash, "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "image_bytes": len(payload), "renderer": config["renderer"]}
        _atomic_json(sidecar, metadata)
        return {"ok": True, "metadata": metadata, "reused": False}
    except Exception as exc:
        return {"ok": False, "ecg_id": row["ecg_id"], "error": repr(exc), "traceback": traceback.format_exc()}


def _verify_job(job):
    row, out, source, config_hash, expected_checksums, sidecar_hash = job
    out = Path(out)
    try:
        sidecar_path = out/f"provenance/{row['split']}/ecg_{row['ecg_id']:05d}.json"
        if sha256_file(sidecar_path) != sidecar_hash:
            raise ValueError("Per-record provenance hash mismatch")
        sidecar = read_json(sidecar_path)
        if sidecar["source_record"] != row:
            raise ValueError("Manifest/source identity mismatch")
        if (sidecar["image_path"] != f"images/{row['split']}/ecg_{row['ecg_id']:05d}.png"
                or sidecar["sidecar_path"] != f"provenance/{row['split']}/ecg_{row['ecg_id']:05d}.json"):
            raise ValueError("Per-record image/provenance path mismatch")
        if sidecar["configuration_sha256"] != config_hash or sidecar["source_sha256"] != expected_checksums:
            raise ValueError("Per-record source/configuration binding mismatch")
        for relative, digest in expected_checksums.items():
            if sha256_file(Path(source)/relative) != digest:
                raise ValueError("Source waveform/header changed after generation")
        payload = (out/sidecar["image_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sidecar["image_sha256"]:
            raise ValueError("Encoded image hash mismatch")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("Image decode/mode failure")
        if image.shape != (sidecar["height"], sidecar["width"]):
            raise ValueError("Image dimension mismatch")
        if hashlib.sha256(image.tobytes()).hexdigest() != sidecar["pixel_sha256"]:
            raise ValueError("Decoded pixel hash mismatch")
        if sidecar["clipped_point_count"] or sidecar["nonfinite_point_count"] or sidecar["out_of_row_point_count"]:
            raise ValueError("Recorded rendering-integrity issue")
        if sidecar["source_points_plotted"] != 60000 or len(sidecar["lead_panels"]) != 12:
            raise ValueError("Incomplete lead/sample coverage")
        if tuple(panel["lead"] for panel in sidecar["lead_panels"]) != LEADS:
            raise ValueError("Lead names/order differ from the canonical 12-lead layout")
        for panel in sidecar["lead_panels"]:
            if panel["sample_start"] != 0 or panel["sample_stop_exclusive"] != 5000:
                raise ValueError("Lead interval is not the complete waveform")
        return {"ok": True, "ecg_id": row["ecg_id"], "image_sha256": sidecar["image_sha256"],
                "pixel_sha256": sidecar["pixel_sha256"], "split": row["split"], "patient_id": row["patient_id"],
                "manifest_row": {**_manifest_row(sidecar), "provenance_sha256": sidecar_hash}}
    except Exception as exc:
        return {"ok": False, "ecg_id": row["ecg_id"], "error": repr(exc)}


def _runtime():
    return {"python": sys.version, "platform": platform.platform(), "wfdb": wfdb.__version__,
            "numpy": np.__version__, "opencv": cv2.__version__}


def _binding(config_path, source_manifest):
    return {"configuration_sha256": sha256_file(config_path),
            "source_code_sha256": {p: sha256_file(ROOT/p) for p in CODE_FILES},
            "source_rows_sha256": hashlib.sha256(json.dumps(source_manifest.rows, sort_keys=True).encode()).hexdigest(),
            "runtime": _runtime()}


def _manifest_row(meta):
    return {**meta["source_record"], "image_path": meta["image_path"], "provenance_path": meta["sidecar_path"],
            "image_sha256": meta["image_sha256"], "pixel_sha256": meta["pixel_sha256"],
            "width": meta["width"], "height": meta["height"], "gain_mm_per_mv": meta["gain_mm_per_mv"],
            "seconds_per_pixel": meta["seconds_per_pixel"], "mv_per_pixel": meta["mv_per_pixel"],
            "image_bytes": meta["image_bytes"], "quality_flags_json": json.dumps(meta["quality_flags"]),
            "flat_leads_json": json.dumps(meta["flat_leads"]), "clipped_point_count": meta["clipped_point_count"],
            "source_header_sha256": meta["source_sha256"][meta["source_record"]["filename_hr"]+".hea"],
            "source_waveform_sha256": meta["source_sha256"][meta["source_record"]["filename_hr"]+".dat"]}


def _csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def generate(config_path=CONFIG_PATH, output=DEFAULT_OUT, workers=4):
    started = time.perf_counter()
    output, config_path = Path(output), Path(config_path)
    config = read_json(config_path)
    source = ROOT/config["source_directory"]
    source_manifest = prepare_manifest(source)
    binding = _binding(config_path, source_manifest)
    review = read_json(ROOT/"results/ptbxl_images_v1/preview/preview_review.json")
    if (not review["passed"] or review["config_sha256"] != binding["configuration_sha256"]
            or review["renderer_sha256"] != binding["source_code_sha256"]["src/ptbxl_rendering.py"]):
        raise ValueError("Development preview review is missing or does not match this renderer/config")
    if shutil.disk_usage(ROOT).free < 8*2**30:
        raise RuntimeError("Need at least 8 GiB free before generating this cohort")
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/"protocol.json"
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        if protocol["binding"] != binding:
            raise ValueError("Resume requires exactly the saved config, renderer, metadata rows and runtime")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Nonempty output directory without a verified generation protocol")
        protocol = {"created_utc": utc_now(), "dataset_id": config["dataset_id"], "binding": binding,
                    "configuration": config, "source_manifest_summary": source_manifest.summary,
                    "preview_review_sha256": sha256_file(ROOT/"results/ptbxl_images_v1/preview/preview_review.json"),
                    "test_style_selection": False, "source_signals_modified": False, "model_training_performed": False}
        write_json(protocol_path, protocol)
    _atomic_json(output/"generation_state.json", {"status": "running", "updated_utc": utc_now(),
                 "records_expected": len(source_manifest.rows), "binding": binding})
    jobs = ((row, str(source), str(output), config,
             {row["filename_hr"]+ext: source_manifest.expected_checksums[row["filename_hr"]+ext] for ext in [".hea", ".dat"]},
             binding["configuration_sha256"]) for row in source_manifest.rows)
    metadata, failures, reused = [], [], 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for count, result in enumerate(pool.map(_render_job, jobs, chunksize=12), 1):
            if result["ok"]:
                metadata.append(result["metadata"])
                reused += int(result["reused"])
            else:
                failures.append(result)
            if count % 500 == 0 or count == len(source_manifest.rows):
                elapsed = time.perf_counter()-started
                print(f"Rendered/checked {count}/{len(source_manifest.rows)}; errors={len(failures)}; elapsed={elapsed:.1f}s", flush=True)
                _atomic_json(output/"generation_state.json", {"status": "running", "updated_utc": utc_now(),
                             "records_checked": count, "successful_records": len(metadata), "errors": len(failures),
                             "records_expected": len(source_manifest.rows), "elapsed_seconds": elapsed})
    if failures:
        write_json(output/"generation_failures.json", failures)
        _atomic_json(output/"generation_state.json", {"status": "failed", "failed_records": len(failures), "updated_utc": utc_now()})
        raise RuntimeError(f"{len(failures)} pages failed; successful outputs preserved for checked resume")
    if _binding(config_path, source_manifest) != binding:
        raise ValueError("Renderer/configuration changed during generation")
    rows = [{**_manifest_row(meta), "provenance_sha256": sha256_file(output/meta["sidecar_path"])} for meta in metadata]
    write_csv(output/"all_images.csv", rows)
    eligible = [r for r in rows if r["eligible_for_classification"]]
    excluded = [r for r in rows if not r["eligible_for_classification"]]
    write_csv(output/"classification_manifest.csv", eligible)
    write_csv(output/"classification_exclusions.csv", excluded)
    for split in ["train", "validation", "test"]:
        write_csv(output/f"splits/{split}.csv", [r for r in eligible if r["split"] == split])
    gain_counts = Counter(str(meta["gain_mm_per_mv"]) for meta in metadata)
    flags = Counter(flag for meta in metadata for flag in meta["quality_flags"])
    summary = {"status": "generated_pending_full_verification", "dataset_id": config["dataset_id"],
               "completed_utc": utc_now(), "images": len(rows), "classification_eligible": len(eligible),
               "classification_ineligible": len(excluded), "eligible_partition_counts": dict(Counter(r["split"] for r in eligible)),
               "all_partition_counts": dict(Counter(r["split"] for r in rows)),
               "source_manifest_summary": source_manifest.summary, "gain_counts": dict(gain_counts),
               "quality_flag_counts": dict(flags), "all_waveform_points_within_rows": True,
               "total_image_bytes": sum(meta["image_bytes"] for meta in metadata),
               "render_runtime_seconds": time.perf_counter()-started, "reused_pages": reused,
               "workers": workers, "binding": binding,
               "manifest_sha256": {name: sha256_file(output/name) for name in ["all_images.csv", "classification_manifest.csv", "classification_exclusions.csv", "splits/train.csv", "splits/validation.csv", "splits/test.csv"]},
               "model_training_performed": False, "test_images_used_for_rendering_choices": False}
    write_json(output/"generation_summary.json", summary)
    _atomic_json(output/"generation_state.json", {"status": "generated_pending_full_verification", "updated_utc": utc_now(), "images": len(rows)})
    print(json.dumps({k:summary[k] for k in ["images", "classification_eligible", "total_image_bytes", "render_runtime_seconds"]}, indent=2))
    return summary


def verify(output=DEFAULT_OUT, workers=4, config_path=CONFIG_PATH):
    output = Path(output)
    started = time.perf_counter()
    protocol, summary = read_json(output/"protocol.json"), read_json(output/"generation_summary.json")
    config = protocol["configuration"]
    if read_json(config_path) != config or summary["binding"] != protocol["binding"]:
        raise ValueError("Embedded configuration or generation binding differs from frozen protocol")
    source_manifest = prepare_manifest(ROOT/config["source_directory"])
    if _binding(config_path, source_manifest) != protocol["binding"]:
        raise ValueError("Current renderer/config/source binding differs from generated cohort")
    for name, digest in summary["manifest_sha256"].items():
        if sha256_file(output/name) != digest:
            raise ValueError(f"Generated manifest changed: {name}")
    with (output/"all_images.csv").open() as f:
        manifest_rows = list(csv.DictReader(f))
    if [int(r["ecg_id"]) for r in manifest_rows] != [r["ecg_id"] for r in source_manifest.rows]:
        raise ValueError("Generated cohort missing, duplicate or reordered source ECGs")
    expected_png = {row["image_path"] for row in manifest_rows}
    actual_png = {str(p.relative_to(output)) for p in (output/"images").rglob("*.png")}
    if actual_png != expected_png:
        raise ValueError("Missing or unexpected generated page files")
    expected_sidecars = {f"provenance/{r['split']}/ecg_{r['ecg_id']:05d}.json" for r in source_manifest.rows}
    actual_sidecars = {str(p.relative_to(output)) for p in (output/"provenance").rglob("*.json")}
    if actual_sidecars != expected_sidecars:
        raise ValueError("Missing or unexpected per-record provenance files")
    failures, checks, hash_groups, verified_rows = [], 0, defaultdict(list), []
    context = multiprocessing.get_context("spawn")
    manifest_by_id = {int(row["ecg_id"]): row for row in manifest_rows}
    jobs = ((r, str(output), str(ROOT/config["source_directory"]), protocol["binding"]["configuration_sha256"],
             {r["filename_hr"]+ext: source_manifest.expected_checksums[r["filename_hr"]+ext] for ext in [".hea", ".dat"]},
             manifest_by_id[r["ecg_id"]]["provenance_sha256"]) for r in source_manifest.rows)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for result in pool.map(_verify_job, jobs, chunksize=16):
            if not result["ok"]:
                failures.append(result)
            else:
                hash_groups[result["pixel_sha256"]].append({k:result[k] for k in ["ecg_id", "patient_id", "split"]})
                verified_rows.append(result["manifest_row"])
            checks += 1
            if checks % 1000 == 0:
                print(f"Decoded/hashed {checks}/{len(source_manifest.rows)} pages; errors={len(failures)}", flush=True)
    collisions = [records for records in hash_groups.values() if len(records)>1]
    crossing = [records for records in collisions if len({r["split"] for r in records})>1]
    if crossing:
        failures.append({"error": "Exact rendered pixel collisions cross partitions", "groups": crossing})
    if not failures:
        eligible = [r for r in verified_rows if r["eligible_for_classification"]]
        expected_manifests = {"all_images.csv": verified_rows, "classification_manifest.csv": eligible,
                              "classification_exclusions.csv": [r for r in verified_rows if not r["eligible_for_classification"]]}
        expected_manifests.update({f"splits/{split}.csv": [r for r in eligible if r["split"] == split]
                                  for split in ["train", "validation", "test"]})
        for name, records in expected_manifests.items():
            if (output/name).read_bytes() != _csv_bytes(records):
                failures.append({"error": "Manifest content differs from verified source/provenance", "path": name})
    report = {"status": "passed" if not failures else "failed", "checked_utc": utc_now(),
              "pages_decoded_and_hashed": checks, "failures": failures, "decoded_image_duplicates": collisions,
              "duplicates_crossing_partitions": crossing, "all_output_files_accounted_for": True,
              "all_source_records_accounted_for": True, "manifest_hashes_verified": True,
              "official_patient_grouping_verified": True, "classification_target_mapping_verified": not failures,
              "model_training_performed": False, "test_images_visually_inspected": False,
              "runtime_seconds": time.perf_counter()-started}
    write_json(output/"verification.json", report)
    if failures:
        _atomic_json(output/"generation_state.json", {"status": "verification_failed", "updated_utc": utc_now()})
        raise RuntimeError(f"Full verification failed; inspect {output/'verification.json'}")
    _atomic_json(output/"generation_state.json", {"status": "complete", "updated_utc": utc_now(), "images": checks})
    print(f"PASS: all {checks} images decode, match hashes and source identities; no cross-partition pixel collisions.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "verify"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use between 1 and 8 bounded worker processes")
    if args.mode == "generate":
        generate(args.config, args.output, args.workers)
    else:
        verify(args.output, args.workers, args.config)
