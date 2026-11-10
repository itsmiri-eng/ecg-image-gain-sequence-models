"""Prepare deterministic PTB-XL rendering and classification metadata.

All source records are returned for rendering. Five diagnostic superclass
targets use all present diagnostic SCP keys, including confidence zero, as in
the distributed example. No-Mapped-class rows remain present but are ineligible
for classification. No labels or source metadata should be printed on images.

Interface: result = prepare_manifest(dataset_dir, audit_path). The result has
rows, summary, and expected_checksums attributes. This function is read-only.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DEFAULT_AUDIT = ROOT / "data/audit/ptbxl_inventory.json"
CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
SPLITS = ("train", "validation", "test")
LABEL_COLUMNS = tuple(f"label_{label}" for label in CLASS_NAMES)


@dataclass(frozen=True)
class ManifestResult:
    rows: list[dict]
    summary: dict
    expected_checksums: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _positive_integer(value: str, name: str) -> int:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ValueError(f"Invalid {name}: {value!r}") from None
    if not number.is_finite() or number != number.to_integral_value() or number <= 0:
        raise ValueError(f"Invalid {name}: {value!r}")
    return int(number)


def _relative_path(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Unsafe or noncanonical dataset path: {value!r}")
    return path


def _verify_metadata(dataset: Path, audit_path: Path) -> tuple[dict, dict, dict]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = audit["source_hashes"]
    required = {"ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt"}
    if not required <= set(expected):
        raise ValueError("Prior verified audit lacks required metadata hashes")
    actual = {}
    for name, digest in expected.items():
        actual[name] = _sha256(dataset / _relative_path(name))
        if actual[name] != digest:
            raise ValueError(f"Metadata differs from prior verified audit: {name}")
    checksums = {}
    for line in (dataset / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        _relative_path(name)
        if name in checksums or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid or duplicate checksum entry: {name}")
        checksums[name] = digest
    for name in ("ptbxl_database.csv", "scp_statements.csv"):
        if checksums.get(name) != actual[name]:
            raise ValueError(f"Metadata differs from shipped checksum: {name}")
    return audit, actual, checksums


def prepare_manifest(
    dataset_dir: str | Path = DEFAULT_DATASET,
    audit_path: str | Path = DEFAULT_AUDIT,
) -> ManifestResult:
    """Return every source row in numeric ECG-ID order without writing files.

    IDs are unique ECG integers and normalized patient-ID strings. Boolean
    targets are label_NORM, label_MI, label_STTC, label_CD and label_HYP.
    `scp_codes_json` retains every source code and confidence value; `labels_json`
    lists present diagnostic targets in CLASS_NAMES order. The official fold is
    retained even for rows ineligible for classification. expected_checksums maps
    shipped dataset-relative file paths to SHA-256 for the renderer's file checks.
    Actual waveform bytes are not rehashed here: that is the renderer's role.
    """
    dataset = Path(dataset_dir).resolve()
    audit_path = Path(audit_path).resolve()
    audit, hashes, checksums = _verify_metadata(dataset, audit_path)
    statements = _read_csv(dataset / "scp_statements.csv")
    all_codes, diagnostic = set(), {}
    for statement in statements:
        code = statement[""]
        if not code or code in all_codes:
            raise ValueError(f"Missing or duplicate SCP code: {code!r}")
        all_codes.add(code)
        indicator = statement["diagnostic"].strip()
        if indicator not in {"", "0", "0.0", "1", "1.0"}:
            raise ValueError(f"Invalid diagnostic indicator for SCP code {code}")
        if indicator in {"1", "1.0"}:
            label = statement["diagnostic_class"]
            if label not in CLASS_NAMES:
                raise ValueError(f"Unknown diagnostic superclass {label!r}")
            diagnostic[code] = label

    rows = []
    ecg_ids, waveform_paths = set(), set()
    patient_folds = defaultdict(set)
    for source in _read_csv(dataset / "ptbxl_database.csv"):
        ecg_id = _positive_integer(source["ecg_id"], "ecg_id")
        patient = str(_positive_integer(source["patient_id"], "patient_id"))
        fold = _positive_integer(source["strat_fold"], "strat_fold")
        if ecg_id in ecg_ids:
            raise ValueError(f"Duplicate ECG ID: {ecg_id}")
        if not 1 <= fold <= 10:
            raise ValueError(f"Official fold outside 1..10: {fold}")
        ecg_ids.add(ecg_id)
        patient_folds[patient].add(fold)
        split = "train" if fold <= 8 else "validation" if fold == 9 else "test"
        try:
            codes = ast.literal_eval(source["scp_codes"])
        except (ValueError, SyntaxError):
            raise ValueError(f"Unparseable SCP dictionary for ECG {ecg_id}") from None
        if not isinstance(codes, dict) or any(not isinstance(code, str) for code in codes):
            raise ValueError(f"Invalid SCP dictionary for ECG {ecg_id}")
        unknown = set(codes) - all_codes
        if unknown:
            raise ValueError(f"Unmapped SCP codes for ECG {ecg_id}: {sorted(unknown)}")
        for value in codes.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid SCP confidence for ECG {ecg_id}")
        targets = {diagnostic[code] for code in codes if code in diagnostic}
        filenames = {}
        for key, directory, suffix in (("filename_hr", "records500", "hr"), ("filename_lr", "records100", "lr")):
            filename = source[key]
            path = _relative_path(filename)
            if len(path.parts) != 3 or path.parts[0] != directory or path.name != f"{ecg_id:05d}_{suffix}":
                raise ValueError(f"Waveform path disagrees with record/rate: {filename}")
            if filename in waveform_paths:
                raise ValueError(f"Waveform path reused by source rows: {filename}")
            waveform_paths.add(filename)
            for extension in (".hea", ".dat"):
                relative = filename + extension
                if relative not in checksums or not (dataset / relative).is_file():
                    raise ValueError(f"Waveform file missing from disk or checksum manifest: {relative}")
            filenames[key] = filename
        rows.append({
            "ecg_id": ecg_id, "patient_id": patient, "official_fold": fold, "split": split,
            **filenames,
            **{f"label_{label}": label in targets for label in CLASS_NAMES},
            "labels_json": json.dumps([label for label in CLASS_NAMES if label in targets], separators=(",", ":")),
            "scp_codes_json": json.dumps(codes, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "eligible_for_classification": bool(targets),
            "exclusion_reason": "" if targets else "no_mapped_diagnostic_superclass",
        })
    if len(rows) != audit["record_count"]:
        raise ValueError("Source record count differs from prior verified audit")
    if any(len(folds) > 1 for folds in patient_folds.values()):
        raise ValueError("A patient appears in multiple official folds")
    rows.sort(key=lambda row: row["ecg_id"])
    partitions = {}
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        eligible = [row for row in selected if row["eligible_for_classification"]]
        partitions[split] = {
            "all_source_records": len(selected), "all_source_patients": len({r["patient_id"] for r in selected}),
            "classification_eligible_records": len(eligible),
            "classification_eligible_patients": len({r["patient_id"] for r in eligible}),
            "classification_ineligible_records": len(selected) - len(eligible),
            "positive_label_counts": {label: sum(row[f"label_{label}"] for row in selected) for label in CLASS_NAMES},
        }
    eligible = [row for row in rows if row["eligible_for_classification"]]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    summary = {
        "dataset_release": "PTB-XL 1.0.3", "render_protocol": "ptbxl_images_v1",
        "all_source_records": len(rows), "all_source_patients": len(patient_folds),
        "records_to_render": len(rows), "records_filtered_from_rendering": 0,
        "classification_eligible_records": len(eligible),
        "classification_eligible_patients": len({row["patient_id"] for row in eligible}),
        "classification_ineligible_records": len(rows) - len(eligible),
        "exclusion_reason_counts": dict(Counter(row["exclusion_reason"] for row in rows if row["exclusion_reason"])),
        "class_names": list(CLASS_NAMES), "label_columns": list(LABEL_COLUMNS),
        "positive_label_counts": {label: sum(row[f"label_{label}"] for row in rows) for label in CLASS_NAMES},
        "diagnostic_superclasses_per_record": dict(sorted(Counter(sum(row[column] for column in LABEL_COLUMNS) for row in rows).items())),
        "partitions": partitions,
        "official_fold_counts": dict(sorted(Counter(row["official_fold"] for row in rows).items())),
        "patients_spanning_official_folds": 0, "duplicate_ecg_ids": 0,
        "policy": "All present diagnostic SCP keys; no confidence threshold; diagnostic superclass union; multilabel; no class priority, balancing, or History_MI derivation",
        "zero_confidence_codes_included": True,
        "no_mapped_class_policy": "Render page; keep source record and original SCP dictionary; mark classification ineligible rather than treating as an all-negative labeled target",
        "rendering_signal": "filename_hr: 500 Hz waveform; filename_lr retained as metadata",
        "patient_id_policy": "Normalize integer-valued CSV IDs to decimal integer strings; preserve official fold; all record variants inherit patient fold",
        "image_text_policy": "Patient IDs, ECG IDs, labels, SCP codes and other source metadata must not be printed on generated pages",
        "metadata_sha256": hashes, "prior_audit_sha256": _sha256(audit_path),
        "canonical_rows_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_waveform_checks": "All referenced rate-specific .hea/.dat files exist and are listed in shipped checksums; actual waveform byte verification belongs to renderer",
        "previous_same_rate_exact_duplicate_evidence": audit.get("same_rate_exact_waveform_duplicates", {}),
        "same_rate_duplicate_evidence_scope": "Prior verified full audit, not a new waveform duplicate scan",
        "model_outputs_used": False, "source_files_changed": False,
    }
    return ManifestResult(rows=rows, summary=summary, expected_checksums=checksums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--counts-json", type=Path, default=ROOT / "experiments/protocols/ptbxl_images_v1/label_counts.json")
    arguments = parser.parse_args()
    result = prepare_manifest(arguments.dataset_dir, arguments.audit_path)
    arguments.counts_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.counts_json.write_text(json.dumps(result.summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result.summary[key] for key in ("all_source_records", "classification_eligible_records", "classification_ineligible_records", "partitions")}, indent=2))
