"""Read-only inventory of the separate, unused local PTB-XL waveform dataset."""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ROOT, sha256_file, utc_now, write_json


def audit_ptbxl():
    started = time.perf_counter()
    dataset = ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    output = ROOT / "data/audit/ptbxl_inventory.json"
    document = ROOT / "docs/ptbxl_secondary_audit.md"
    if output.exists() or document.exists():
        raise FileExistsError("Preserve the existing secondary dataset audit; archive it explicitly before a rerun")
    database = pd.read_csv(dataset / "ptbxl_database.csv")
    statements = pd.read_csv(dataset / "scp_statements.csv", index_col=0)
    manifest = []
    for line in (dataset / "SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = Path(relative)
        if len(expected) != 64 or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid manifest entry: {line!r}")
        manifest.append((expected, relative))
    missing, mismatches, errors, header_errors = [], [], [], []
    header_specs, dimensions, formats, lead_orders = {}, Counter(), Counter(), Counter()
    for _, relative in manifest:
        if not relative.endswith(".hea"):
            continue
        path = dataset / relative
        if not path.exists():
            continue
        try:
            lines = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
            record, n_signals, sample_rate, n_samples = lines[0][:4]
            n_signals, sample_rate, n_samples = int(n_signals), float(sample_rate), int(n_samples)
            if len(lines) != n_signals + 1 or record != path.stem:
                raise ValueError("Record name/signal-line count does not match header")
            signal_lines = lines[1:]
            data_names = {line[0] for line in signal_lines}
            signal_formats = {line[1] for line in signal_lines}
            if data_names != {path.with_suffix(".dat").name} or signal_formats != {"16"}:
                raise ValueError("Audit reader supports only the observed single-file WFDB format 16")
            lead_names = tuple(line[-1] for line in signal_lines)
            dimensions[f"{n_signals} leads x {n_samples} samples at {sample_rate:g} Hz"] += 1
            formats.update(line[1] for line in signal_lines)
            lead_orders[",".join(lead_names)] += 1
            header_specs[str(Path(relative).with_suffix(".dat"))] = {
                "header": relative,
                "shape": (n_samples, n_signals),
                "rate": int(sample_rate),
                "initial": np.array([int(line[5]) for line in signal_lines]),
                "checksums": np.array([int(line[6]) % 65536 for line in signal_lines]),
            }
        except Exception as error:
            header_errors.append({"path": relative, "error": repr(error)})
    exact_data_groups = {100: defaultdict(list), 500: defaultdict(list)}
    byte_sizes = Counter()
    byte_size_errors, signal_integrity_errors, flat_channel_records = [], [], []
    file_counts, verified_bytes, matched_count = Counter(), 0, 0
    int16_min_count = 0
    for index, (expected, relative) in enumerate(manifest, 1):
        path = dataset / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            content = path.read_bytes()
            verified_bytes += len(content)
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected:
                mismatches.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual})
            else:
                matched_count += 1
            file_counts[path.suffix or "no_extension"] += 1
            if relative.endswith(".dat"):
                spec = header_specs.get(relative)
                if spec is None:
                    signal_integrity_errors.append({"path": relative, "error": "Missing/unparseable associated header"})
                    continue
                rate = spec["rate"]
                exact_data_groups.setdefault(rate, defaultdict(list))[actual].append(relative)
                byte_sizes[f"{rate} Hz: {len(content)} bytes"] += 1
                expected_bytes = int(np.prod(spec["shape"])) * 2
                if len(content) != expected_bytes:
                    byte_size_errors.append({"path": relative, "expected": expected_bytes, "actual": len(content)})
                    continue
                samples = np.frombuffer(content, dtype="<i2").reshape(spec["shape"])
                initial_ok = np.array_equal(samples[0], spec["initial"])
                checksum_ok = np.array_equal(samples.sum(axis=0, dtype=np.int64) % 65536, spec["checksums"])
                if not initial_ok or not checksum_ok:
                    signal_integrity_errors.append({"path": relative, "initial_sample_matches": initial_ok, "channel_checksum_matches": checksum_ok})
                flat_channels = np.flatnonzero(np.all(samples == samples[0], axis=0)).tolist()
                if flat_channels:
                    flat_channel_records.append({"path": relative, "zero_based_channel_indices": flat_channels})
                int16_min_count += int((samples == -32768).sum())
        except Exception as error:
            errors.append({"path": relative, "error": repr(error)})
        if index % 10000 == 0:
            print(f"PTB-XL verified {index}/{len(manifest)} entries ({verified_bytes / 1e9:.2f} GB)", flush=True)
    expected_paths = {
        str(value) + suffix
        for field in ("filename_lr", "filename_hr")
        for value in database[field]
        for suffix in (".hea", ".dat")
    }
    listed_waveforms = {relative for _, relative in manifest if relative.endswith((".hea", ".dat"))}
    patient_fold_counts = database.groupby("patient_id")["strat_fold"].nunique()
    official_parts = {
        "train_folds_1_to_8": database[database.strat_fold <= 8],
        "validation_fold_9": database[database.strat_fold == 9],
        "test_fold_10": database[database.strat_fold == 10],
    }
    partitions = {name: {"records": len(frame), "patients": int(frame.patient_id.nunique())} for name, frame in official_parts.items()}
    patient_sets = {name: set(frame.patient_id) for name, frame in official_parts.items()}
    patient_overlap = {
        f"{a} vs {b}": len(patient_sets[a] & patient_sets[b])
        for index, a in enumerate(patient_sets) for b in list(patient_sets)[index + 1:]
    }
    code_counts, superclass_counts = Counter(), Counter()
    code_cardinality, superclass_cardinality = Counter(), Counter()
    parse_errors, unmapped_codes = [], set()
    diagnostic_mapping = statements.loc[statements.diagnostic == 1, "diagnostic_class"].dropna().to_dict()
    for row in database.itertuples(index=False):
        try:
            codes = ast.literal_eval(row.scp_codes)
            if not isinstance(codes, dict):
                raise ValueError("scp_codes is not a dictionary")
            code_counts.update(codes.keys())
            code_cardinality[len(codes)] += 1
            unmapped_codes.update(set(codes) - set(statements.index))
            classes = {diagnostic_mapping[code] for code in codes if code in diagnostic_mapping}
            superclass_counts.update(classes)
            superclass_cardinality[len(classes)] += 1
        except Exception as error:
            parse_errors.append({"ecg_id": int(row.ecg_id), "error": repr(error)})
    duplicates = {}
    path_to_fold = {
        str(value) + ".dat": int(fold)
        for field in ("filename_lr", "filename_hr")
        for value, fold in zip(database[field], database.strat_fold)
    }
    for rate, groups in exact_data_groups.items():
        repeated = [{"sha256": digest, "paths": paths, "official_folds": sorted({path_to_fold[p] for p in paths if p in path_to_fold})}
                    for digest, paths in groups.items() if len(paths) > 1]
        duplicates[str(rate)] = {
            "data_files": sum(map(len, groups.values())), "unique_data_sha256": len(groups),
            "redundant_exact_data_copies": sum(len(group["paths"]) - 1 for group in repeated),
            "groups_crossing_official_folds": sum(len(group["official_folds"]) > 1 for group in repeated),
            "duplicate_groups": repeated,
        }
    inventory = {
        "created_utc": utc_now(), "dataset_directory": str(dataset.relative_to(ROOT)),
        "status": "SECONDARY UNUSED DATASET; excluded from all current image LDA/CNN experiments",
        "record_count": len(database), "unique_ecg_ids": int(database.ecg_id.nunique()),
        "unique_patients": int(database.patient_id.nunique()),
        "missing_patient_ids": int(database.patient_id.isna().sum()),
        "patients_in_multiple_official_folds": int((patient_fold_counts > 1).sum()),
        "official_fold_counts": {str(k): int(v) for k, v in database.strat_fold.value_counts().sort_index().items()},
        "official_partitions": partitions, "patient_overlap_between_partitions": patient_overlap,
        "manifest": {"entries": len(manifest), "matched_files": matched_count,
                     "duplicate_path_entries": len(manifest) - len({p for _, p in manifest}),
                     "verified_bytes": verified_bytes, "missing_files": missing,
                     "checksum_mismatches": mismatches, "read_errors": errors,
                     "file_extension_counts": dict(file_counts)},
        "header_dimensions": dict(dimensions), "signal_format_counts": dict(formats),
        "lead_order_counts": dict(lead_orders), "data_byte_size_counts": dict(byte_sizes),
        "header_parse_errors": header_errors, "data_byte_size_errors": byte_size_errors,
        "header_signal_integrity_errors": signal_integrity_errors,
        "integrity_checks": "All observed WFDB format-16 files decoded as little-endian int16; shape/byte size, first samples, and per-channel sums modulo 65536 compared to header fields.",
        "flat_channel_records": flat_channel_records, "int16_min_value_count": int16_min_count,
        "metadata_referenced_files_missing_from_manifest": sorted(expected_paths - listed_waveforms),
        "manifest_waveforms_not_referenced_by_database": sorted(listed_waveforms - expected_paths),
        "same_rate_exact_waveform_duplicates": duplicates,
        "duplicate_scope": "SHA256 equality of same-rate .dat bytes; 100-Hz and 500-Hz files are alternative representations of the same ECG, not independent samples. Near-duplicate signal similarity was not assessed.",
        "scp_label_parse_errors": parse_errors, "unmapped_scp_codes": sorted(unmapped_codes),
        "scp_code_counts": dict(code_counts.most_common()),
        "scp_codes_per_record": {str(k): int(v) for k, v in sorted(code_cardinality.items())},
        "diagnostic_superclass_counts": dict(superclass_counts.most_common()),
        "diagnostic_superclasses_per_record": {str(k): int(v) for k, v in sorted(superclass_cardinality.items())},
        "diagnostic_label_policy": "Count each diagnostic superclass at most once per ECG from all present scp_codes keys where local scp_statements.diagnostic==1; no likelihood threshold. Labels are multilabel and counts overlap.",
        "source_hashes": {name: sha256_file(dataset / name) for name in ["SHA256SUMS.txt", "ptbxl_database.csv", "scp_statements.csv", "RECORDS", "ptbxl_v103_changelog.txt"]},
        "script_sha256": sha256_file(Path(__file__)), "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output, inventory)
    duplicate_summary = ", ".join(f"{rate} Hz: {item['redundant_exact_data_copies']}" for rate, item in duplicates.items())
    lines = [
        "# Secondary PTB-XL inventory — unused in current experiments", "",
        "The local PTB-XL 1.0.3 directory was audited independently. Its waveform records were not mixed with the ECG image dataset, and no PTB-XL model was trained.", "",
        f"- Metadata: **{len(database):,} ECG records**, **{database.patient_id.nunique():,} patients**; {database.patient_id.isna().sum()} missing patient IDs.",
        "- Input: 12 simultaneous leads, 10 seconds; 1,000 samples per lead at 100 Hz or 5,000 samples per lead at 500 Hz. These are two representations of each ECG.",
        f"- SHA256 manifest: **{matched_count:,}/{len(manifest):,} files matched**, {len(missing)} missing, {len(mismatches)} checksum mismatches, {len(errors)} read errors; {verified_bytes / 1e9:.3f} GB read.",
        f"- Header/byte checks: {len(header_errors)} unparseable headers, {len(byte_size_errors)} byte-size mismatches, {len(signal_integrity_errors)} initial-value/channel-checksum errors.",
        f"- Metadata coverage: {len(expected_paths - listed_waveforms)} referenced files absent from manifest; {len(listed_waveforms - expected_paths)} waveform/header files outside the database.",
        f"- Same-rate exact waveform duplicates: {duplicate_summary}. No near-duplicate waveform search was performed.",
        f"- Constant-channel records: {len(flat_channel_records)} rate-specific files; raw int16 minimum (-32768) values: {int16_min_count}. These are descriptive signal checks, not clinical-quality adjudication.", "",
        "## Existing official partition metadata", "",
        "| Partition | Records | Patients |", "|---|---:|---:|",
    ]
    lines.extend(f"| {name} | {part['records']:,} | {part['patients']:,} |" for name, part in partitions.items())
    lines.extend(["", f"Patients appearing in multiple official folds: **{int((patient_fold_counts > 1).sum())}**. Patient overlap between every train/validation/test pair: **{list(patient_overlap.values())}**.", "", "## Local diagnostic superclass labels", "", "| Superclass | ECG records containing label |", "|---|---:|"])
    lines.extend(f"| {name} | {count:,} |" for name, count in superclass_counts.most_common())
    lines.extend(["", inventory["diagnostic_label_policy"],
                  f"Records with multiple diagnostic superclasses: **{sum(n for k, n in superclass_cardinality.items() if k > 1):,}**; without a mapped diagnostic superclass: **{superclass_cardinality[0]:,}**.",
                  "", "These labels differ from the four image folders (Normal, MI, Abnormal, History_MI). A future waveform experiment needs a separate prespecified task, preprocessing, and evaluation protocol. Keep the existing patient-based folds and never treat both sampling rates as separate patients.",
                  "", "The complete SCP-code frequency table, input dimensions, exact-duplicate groups, anomaly lists, and source checksums are saved in `data/audit/ptbxl_inventory.json`. This audit verifies consistency with the local shipped manifest; it does not independently authenticate that manifest against an online source.",
                  "", "Reproduce after explicitly archiving the existing audit outputs:", "", "```bash", "/opt/anaconda3/bin/python -m src.audit_ptbxl", "```", ""])
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "matched_files": matched_count, "manifest_entries": len(manifest), "runtime_seconds": inventory["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    audit_ptbxl()
