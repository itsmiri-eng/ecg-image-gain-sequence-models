"""Offline, exact-glyph header transcription for this fixed rendered image cohort.

This is a narrowly scoped typography lookup, not a general OCR system. Manually
read reference headers cover digits 0..9. Unknown/non-exact glyphs are rejected.
Printed IDs remain unverified report identifiers, not established patient IDs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from src.common import ROOT, sha256_file, utc_now, write_csv, write_json


REFERENCES = {
    "ecg files/Normal/Normal(1).jpg": "168590",
    "ecg files/Normal/Normal(2).jpg": "168597",
    "ecg files/Abnormal/HB(1).jpg": "168594",
    "ecg files/HIstory of MI/PMI(1).jpg": "168592",
    "ecg files/MI/MI(3).jpg": "177073",
}
FOLDERS = ["Normal", "MI", "Abnormal", "HIstory of MI"]
ROI_XYXY = (125, 65, 245, 100)
THRESHOLD = 130


def extract_glyphs(path):
    with Image.open(path) as image:
        if image.size != (2213, 1572):
            raise ValueError(f"Unexpected image dimensions: {image.size}")
        pixels = np.asarray(image.convert("L").crop(ROI_XYXY))
    binary = (pixels < THRESHOLD).astype(np.uint8)
    labels, _ = ndimage.label(binary, structure=np.ones((3, 3)))
    boxes = []
    for number, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        yy, xx = slices
        height = yy.stop - yy.start
        area = int((labels[slices] == number).sum())
        if height >= 12 and area >= 15:
            boxes.append((xx.start, yy.start, xx.stop - xx.start, height))
    boxes.sort()
    if len(boxes) != 6:
        raise ValueError(f"Expected six digit components, found {len(boxes)}")
    glyphs = []
    for x, y, width, height in boxes:
        patch = binary[y : y + height, x : x + width].astype(np.float32)
        scaled_width = round(width * 32 / height)
        if scaled_width > 28:
            raise ValueError("Digit component is unexpectedly wide")
        canvas = np.zeros((36, 28), dtype=np.float32)
        start = (28 - scaled_width) // 2
        canvas[2:34, start : start + scaled_width] = np.asarray(
            Image.fromarray(patch).resize((scaled_width, 32), Image.Resampling.BILINEAR)
        )
        glyphs.append(canvas)
    return np.stack(glyphs)


def main():
    output = ROOT / "data/audit/header_ids.csv"
    if output.exists():
        raise FileExistsError("The final identity audit already exists; preserve it")
    templates, digits, reference_rows = [], [], []
    for relative, label in REFERENCES.items():
        path = ROOT / relative
        templates.extend(extract_glyphs(path))
        digits.extend(label)
        reference_rows.append({
            "path": relative, "file_sha256": sha256_file(path), "printed_id": label,
            "annotation_method": "direct_visual_reading_of_original_header_by_cnn_baseline_agent",
        })
    templates, digits = np.stack(templates), np.asarray(digits)
    if set(digits) != set("0123456789"):
        raise ValueError("Reference set does not cover every digit")
    file_groups = {}
    for folder in FOLDERS:
        for path in sorted((ROOT / "ecg files" / folder).glob("*.jpg")):
            file_groups.setdefault(sha256_file(path), path)
    rows, glyph_errors, margins = [], [], []
    for file_hash, path in sorted(file_groups.items()):
        candidate, maximum_error, minimum_margin, error = "", None, None, ""
        try:
            glyphs = extract_glyphs(path)
            distances = np.mean((glyphs[:, None] - templates[None]) ** 2, axis=(2, 3))
            per_digit = np.stack([distances[:, digits == str(d)].min(axis=1) for d in range(10)], axis=1)
            ranking = per_digit.argsort(axis=1)
            best = per_digit[np.arange(6), ranking[:, 0]]
            second = per_digit[np.arange(6), ranking[:, 1]]
            maximum_error = float(best.max())
            minimum_margin = float((second - best).min())
            candidate = "".join(map(str, ranking[:, 0]))
            glyph_errors.extend(map(float, best))
            margins.extend(map(float, second - best))
            accepted = maximum_error == 0.0 and minimum_margin >= 0.05
            status = "exact_template_match" if accepted else "requires_visual_review"
        except Exception as exc:
            accepted, status, error = False, "segmentation_error", repr(exc)
        rows.append({
            "path": str(path.relative_to(ROOT)), "file_sha256": file_hash,
            "printed_id": candidate if accepted else "", "confidence": "",
            "status": status, "candidate_id": candidate,
            "maximum_digit_mse": maximum_error, "minimum_competing_digit_margin": minimum_margin,
            "method": "offline_exact_binary_glyph_template_matching",
            "error": error, "manual_correction": "none",
        })
    by_hash = {row["file_sha256"]: row for row in rows}
    validation_path = ROOT / "data/audit/manual_id_validation.csv"
    with validation_path.open() as handle:
        validation = list(csv.DictReader(handle))
    validation_results = []
    reference_hashes = {row["file_sha256"] for row in reference_rows}
    for item in validation:
        digest = sha256_file(ROOT / item["path"])
        result = by_hash[digest]
        validation_results.append({
            **item, "file_sha256": digest, "template_prediction": result["printed_id"],
            "agrees": item["printed_id"] == result["printed_id"],
            "is_reference_image": digest in reference_hashes,
        })
    if not all(item["agrees"] for item in validation_results):
        raise ValueError(f"Manual validation disagrees: {[row for row in validation_results if not row['agrees']]}")
    template_path = ROOT / "data/audit/header_digit_templates.npz"
    np.savez_compressed(template_path, templates=templates, digits=digits)
    write_csv(ROOT / "data/audit/header_digit_references.csv", reference_rows)
    write_csv(ROOT / "data/audit/header_template_validation.csv", validation_results)
    write_csv(output, rows)
    summary = {
        "created_utc": utc_now(), "unique_source_files": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "unique_recognized_ids": len({row["printed_id"] for row in rows if row["printed_id"]}),
        "roi_xyxy_pixels": ROI_XYXY, "grayscale_threshold_below": THRESHOLD,
        "segmentation": "8-connected components with height>=12 and area>=15; exactly six components required",
        "glyph_normalization": "32px tall, aspect-preserving bilinear resize; centered in 36x28 zero canvas",
        "acceptance": "Every digit has exactly zero MSE to a visually labeled template, and next digit margin>=0.05",
        "reference_pages": len(REFERENCES), "reference_glyphs": len(digits),
        "maximum_digit_mse": max(glyph_errors) if glyph_errors else None,
        "minimum_competing_digit_margin": min(margins) if margins else None,
        "manual_validation_rows": len(validation_results),
        "manual_validation_agreements": sum(row["agrees"] for row in validation_results),
        "nonreference_manual_validation_rows": sum(not row["is_reference_image"] for row in validation_results),
        "validation_policy": "Root agent independently visually transcribed headers before template predictions were available; reference overlaps are explicitly marked",
        "confidence_policy": "No calibrated OCR probability is claimed; exact distance and separation are reported directly",
        "identity_limitation": "Transcribed printed report IDs do not establish unique patient identity. Group equal IDs conservatively across classes.",
        "header_ids_sha256": sha256_file(output), "templates_sha256": sha256_file(template_path),
        "manual_validation_sha256": sha256_file(validation_path),
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(ROOT / "data/audit/header_template_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
