"""Read-only label readiness review of the frozen image cohort.

Writes aggregate group evidence and an unfilled development-only adjudication
queue. It never imports model outputs, changes labels, or rebuilds the split.
Uses only Python's standard library: python3 -m src.review_labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("Normal", "MI", "Abnormal", "History_MI")
PARTITIONS = ("train", "validation", "test")
REVIEW_FIELDS = ("verified_label", "reviewer", "evidence", "status")


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def composition(rows):
    present = {row["class_name"] for row in rows}
    return "+".join(label for label in CLASSES if label in present)


def review(root):
    root = root.resolve()
    out = root / "results/readiness/labels"
    queue_path = out / "development_adjudication_queue.csv"
    if queue_path.exists():
        previous = read_csv(queue_path)
        require(not any(row.get(field, "").strip() for row in previous for field in REVIEW_FIELDS),
                "Adjudication queue contains reviewer entries; refusing to overwrite them")

    inputs = {
        "split": "data/splits/fixed_split.csv",
        "protocol": "data/splits/protocol.json",
        "dataset_audit": "data/audit/dataset_audit.json",
        "manifest": "data/audit/source_manifest.csv",
        "exclusions": "data/audit/exclusions.csv",
        "header_ids": "data/audit/header_ids.csv",
        "header_summary": "data/audit/header_template_summary.json",
        "archive": "data/audit/archive_provenance.json",
        "near_copy_summary": "data/audit/near_duplicate_summary.json",
        "near_copy_links": "data/audit/near_duplicate_links.csv",
        "config": "experiments/configs/baseline_v1.json",
        "cache": "data/processed/inputs_v1.npz",
    }
    before = {path: sha256(root / path) for path in inputs.values()}
    audit = read_json(root / inputs["dataset_audit"])
    expected = {"split": "split_sha256", "manifest": "source_manifest_sha256",
                "header_ids": "header_ids_sha256", "near_copy_links": "near_duplicate_links_sha256",
                "config": "config_sha256", "cache": "cache_sha256"}
    for name, field in expected.items():
        require(before[inputs[name]] == audit[field], f"Frozen input mismatch: {inputs[name]}")
    require(read_json(root / inputs["protocol"])["audit"] == audit, "Protocol/audit mismatch")

    split = read_csv(root / inputs["split"])
    manifest = read_csv(root / inputs["manifest"])
    exclusions = read_csv(root / inputs["exclusions"])
    headers = read_csv(root / inputs["header_ids"])
    source = {row["path"]: row for row in manifest}
    require(len(source) == len(manifest), "Source paths are not unique")
    for path, row in source.items():
        require(sha256(root / path) == row["file_sha256"], f"Changed source image: {path}")
    ids = {row["file_sha256"]: row for row in headers}
    require(len(ids) == len(headers), "Header hash rows are not unique")
    for row in headers:
        require(row["status"] == "exact_template_match" and row["printed_id"], "Unresolved printed ID")

    exact_file, exact_pixel = defaultdict(list), defaultdict(list)
    for row in manifest:
        require(row["decode_status"] == "valid", "Invalid source image in verified cohort")
        require(row["class_name"] in CLASSES, "Unexpected folder class")
        exact_file[row["file_sha256"]].append(row)
        exact_pixel[row["pixel_sha256"]].append(row)
    file_conflicts = [rows for rows in exact_file.values() if len({r["class_name"] for r in rows}) > 1]
    pixel_conflicts = [rows for rows in exact_pixel.values() if len({r["class_name"] for r in rows}) > 1]
    require(not file_conflicts and not pixel_conflicts, "New exact-copy label conflict; review manually")

    retained = {row["path"]: row for row in split}
    require(len(retained) == len(split), "Duplicate retained source path")
    require(len({row["sample_id"] for row in split}) == len(split), "Duplicate retained sample ID")
    excluded = {row["path"]: row for row in exclusions}
    require(len(excluded) == len(exclusions), "Duplicate exclusion row")
    require(set(retained).isdisjoint(excluded), "Retained/excluded source overlap")
    require(set(retained) | set(excluded) == set(source), "Incomplete source-to-canonical accounting")
    for row in exclusions:
        require(row["reason"] == "exact_decoded_pixel_duplicate", "Unexpected exclusion policy")
        canonical = retained[row["canonical_path"]]
        original = source[row["path"]]
        require(canonical["pixel_sha256"] == original["pixel_sha256"], "Invalid canonical mapping")
        require(canonical["class_name"] == original["class_name"], "Canonical/source label mismatch")

    groups, printed, model_inputs = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in split:
        require(row["split"] in PARTITIONS, "Unexpected partition")
        require(row["class_name"] in CLASSES, "Unexpected analysis label")
        for field in ("class_name", "label", "file_sha256", "pixel_sha256"):
            require(row[field] == source[row["path"]][field], f"Changed canonical {field}")
        require(row["sample_id"] == row["pixel_sha256"][:20], "Changed canonical sample ID")
        row["printed_proxy"] = ids[row["file_sha256"]]["printed_id"]
        groups[row["group_id"]].append(row)
        printed[row["printed_proxy"]].append(row)
        model_inputs[row["input_sha256"]].append(row)
    for rows in list(groups.values()) + list(printed.values()) + list(model_inputs.values()):
        require(len({row["split"] for row in rows}) == 1, "Identity or model-input overlap across partitions")
    for rows in printed.values():
        require(len({row["group_id"] for row in rows}) == 1, "Printed proxy spans frozen groups")
    for group_id, rows in groups.items():
        expected_id = hashlib.sha256(json.dumps(sorted(row["sample_id"] for row in rows), sort_keys=True).encode()).hexdigest()[:20]
        require(group_id == expected_id, "Opaque group ID differs from frozen membership digest")

    group_rows = []
    for group_id, rows in sorted(groups.items(), key=lambda item: (PARTITIONS.index(item[1][0]["split"]), item[0])):
        counts = Counter(row["class_name"] for row in rows)
        group_rows.append({
            "group_id": group_id, "partition": rows[0]["split"], "n_records": len(rows),
            "n_printed_id_proxies": len({row["printed_proxy"] for row in rows}),
            "n_supplied_labels": len(counts), "mixed_folder_labels": str(len(counts) > 1).lower(),
            "class_composition": composition(rows),
            **{f"{label}_records": counts[label] for label in CLASSES},
        })
    group_lookup = {row["group_id"]: row for row in group_rows}
    class_counts = []
    partition_counts = []
    for partition in (*PARTITIONS, "all"):
        rows = [row for row in split if partition == "all" or row["split"] == partition]
        relevant = {row["group_id"] for row in rows}
        mixed = {group_id for group_id in relevant if group_lookup[group_id]["n_supplied_labels"] > 1}
        partition_counts.append({
            "partition": partition, "records": len(rows), "distinct_frozen_groups": len(relevant),
            "distinct_printed_id_proxies": len({row["printed_proxy"] for row in rows}),
            "mixed_folder_groups": len(mixed),
            "records_in_mixed_folder_groups": sum(row["group_id"] in mixed for row in rows),
            "repeated_record_groups": sum(group_lookup[g]["n_records"] > 1 for g in relevant),
        })
        for label in CLASSES:
            labeled = [row for row in rows if row["class_name"] == label]
            label_groups = {row["group_id"] for row in labeled}
            class_counts.append({
                "partition": partition, "class_name": label, "records": len(labeled),
                "distinct_printed_id_proxies": len({row["printed_proxy"] for row in labeled}),
                "distinct_frozen_groups": len(label_groups),
                "single_label_groups": len(label_groups - mixed),
                "mixed_folder_groups": len(label_groups & mixed),
            })

    composition_rows = []
    for partition in (*PARTITIONS, "all"):
        relevant = [row for row in group_rows if partition == "all" or row["partition"] == partition]
        for value in sorted({row["class_composition"] for row in relevant}):
            selected = [row for row in relevant if row["class_composition"] == value]
            composition_rows.append({"partition": partition, "class_composition": value,
                                     "n_groups": len(selected), "n_records": sum(row["n_records"] for row in selected)})
    queue = []
    for row in split:
        if row["split"] == "test":
            continue
        group = group_lookup[row["group_id"]]
        reasons = []
        if group["n_supplied_labels"] > 1:
            reasons.append("mixed_folder_labels_within_printed_id_group")
        if row["class_name"] == "MI":
            reasons.append("small_MI_class")
        if not reasons:
            reasons.append("repeated_ID_same_folder" if group["n_records"] > 1 else "routine_label_provenance_review")
        queue.append({
            "group_id": row["group_id"], "partition": row["split"], "sample_id": row["sample_id"],
            "source_path": row["path"], "supplied_label": row["class_name"],
            "file_sha256": row["file_sha256"], "pixel_sha256": row["pixel_sha256"], "input_sha256": row["input_sha256"],
            "group_n_records": group["n_records"], "group_class_composition": group["class_composition"],
            "source_exact_copy_count": len(exact_pixel[row["pixel_sha256"]]),
            "review_reason": ";".join(reasons), **{field: "" for field in REVIEW_FIELDS},
        })
    queue.sort(key=lambda row: (
        "mixed_folder_labels" not in row["review_reason"], "small_MI_class" not in row["review_reason"],
        PARTITIONS.index(row["partition"]), row["group_id"], row["sample_id"],
    ))
    require(not any(row["partition"] == "test" for row in queue), "Test sample entered adjudication queue")
    require(all(not row[field] for row in queue for field in REVIEW_FIELDS), "Queue is not blank")
    require(len(queue) == sum(row["split"] != "test" for row in split), "Development queue coverage mismatch")

    after = {path: sha256(root / path) for path in inputs.values()}
    require(before == after, "A frozen input changed during the review")
    near_copy = read_json(root / inputs["near_copy_summary"])
    archive = read_json(root / inputs["archive"])
    summary = {
        "review_scope": "metadata and grouping consistency; no clinical adjudication or label changes",
        "dataset_version": audit["dataset_version"], "split_version": audit["split_version"],
        "source_image_hashes_reverified": len(manifest), "source_images": len(manifest),
        "retained_canonical_images": len(split), "excluded_exact_copies": len(exclusions),
        "exact_file_groups_with_label_conflicts": len(file_conflicts),
        "exact_pixel_groups_with_label_conflicts": len(pixel_conflicts),
        "identical_model_input_groups_with_label_conflicts": sum(len({r["class_name"] for r in rows}) > 1 for rows in model_inputs.values()),
        "printed_ID_proxies": len(printed), "frozen_groups": len(groups),
        "groups_with_multiple_printed_ID_proxies": sum(row["n_printed_id_proxies"] > 1 for row in group_rows),
        "groups_crossing_partitions": 0, "printed_IDs_crossing_partitions": 0,
        "partition_counts": partition_counts, "class_group_counts": class_counts,
        "group_compositions": composition_rows,
        "development_adjudication_queue_records": len(queue),
        "development_adjudication_queue_groups": len({row["group_id"] for row in queue}),
        "test_records_in_adjudication_queue": 0,
        "verified_clinical_labels": 0, "adjudication_fields": list(REVIEW_FIELDS),
        "labels_changed": False, "split_changed": False,
        "queue_scope": "train_and_validation_only; all canonical development samples; no test rows or model predictions",
        "model_predictions_read": False,
        "near_copy_evidence": {"source": inputs["near_copy_summary"], "candidate_pairs": near_copy["candidate_pairs"],
                               "accepted_links": near_copy["conservative_grouping_links"], "limitation": near_copy["limitation"]},
        "archive_evidence": {"source": inputs["archive"], "exact_byte_matches": archive["exact_byte_matches"],
                             "scope": "Existing archive/source verification; all current enumerated raw hashes also reverified in this review"},
        "limitations": [
            "Distinct printed IDs or frozen groups are independence proxies, not verified independent patients.",
            "Repeated IDs with different folder labels do not establish that either label is wrong.",
            "Supplied folder categories are not independently adjudicated diagnoses.",
            "The local source metadata cannot settle whether IDs denote patients, examinations, or reused counters.",
            "The similarity screen is not exhaustive near-copy detection.",
            "Per-class group counts overlap where the same frozen group contains multiple labels.",
            "Header transcription verification supports ID reading, not diagnosis or patient identity.",
        ],
        "frozen_input_sha256": before, "frozen_inputs_unchanged": before == after,
        "script_sha256": sha256(Path(__file__).resolve()),
        "reproduce": "python3 -m src.review_labels",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "group_review.csv", group_rows)
    write_csv(out / "class_group_counts.csv", class_counts)
    write_csv(out / "group_compositions.csv", composition_rows)
    write_csv(queue_path, queue)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(root / "docs/label_consistency_review.md", summary)
    print(json.dumps({"records": len(split), "groups": len(groups), "development_queue_rows": len(queue),
                      "partition_counts": partition_counts, "frozen_inputs_unchanged": True}, indent=2))


def write_report(path, summary):
    lines = [
        "# Label consistency and readiness review", "",
        "The technical label review is complete. The supplied folder labels remain unchanged; clinical label validity is unresolved. This review verifies provenance, duplicate-label agreement, grouping and queue coverage. It does not assign diagnoses or decide that a record is mislabeled.", "",
        f"Frozen dataset: `{summary['dataset_version']}`. Frozen split: `{summary['split_version']}`. Reproduce with `python3 -m src.review_labels` from the project root; only standard-library Python is required.", "",
        "## Verified evidence", "",
        f"All {summary['source_image_hashes_reverified']} enumerated raw image hashes were rechecked against the source manifest. The retained {summary['retained_canonical_images']} canonical paths, labels, sample identifiers and hashes agree with the fixed split. All {summary['excluded_exact_copies']} exclusions map to a retained representative with the same pixel hash and folder label. The fixed split, configuration, input cache, metadata and grouping-evidence hashes agree with the frozen audit and remained unchanged during this review.", "",
        f"There are {summary['exact_file_groups_with_label_conflicts']} exact-file duplicate groups, {summary['exact_pixel_groups_with_label_conflicts']} decoded-pixel duplicate groups and {summary['identical_model_input_groups_with_label_conflicts']} identical-model-input groups containing conflicting folder labels. The archive audit already records 928 byte-for-byte matches between the ZIP and source files; this review rechecks the current files against their frozen manifest rather than treating the archive as an independent cohort.", "",
        f"The existing near-copy screen considered {summary['near_copy_evidence']['candidate_pairs']:,} pairs and accepted {summary['near_copy_evidence']['accepted_links']} extra grouping links. Its fixed-crop perceptual-hash, correlation and absolute-difference thresholds may miss shifted, rescaled or redrawn copies. No absence-of-near-copies or patient-independence claim follows from a zero-link result.", "",
        "## Distinct identifier groups", "",
        f"There are {summary['printed_ID_proxies']} printed-ID proxies and {summary['frozen_groups']} frozen groups; no group combines multiple distinct printed IDs in this cohort. Equal printed IDs remain together in one partition, and no frozen group crosses partitions. These are separate identifier groups, not confirmed independent patients.", "",
        "| Partition | Records | Distinct groups | Groups spanning folder labels | Records in those groups | Groups with multiple records |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["partition_counts"]:
        lines.append(f"| {row['partition']} | {row['records']} | {row['distinct_frozen_groups']} | {row['mixed_folder_groups']} | {row['records_in_mixed_folder_groups']} | {row['repeated_record_groups']} |")
    lines += ["", "Per-class counts of distinct groups overlap: a group represented in two labels contributes once to each of their rows. Summing a partition's class-level group counts therefore overstates its total distinct groups.", "",
              "| Partition | Supplied label | Records | Distinct ID / frozen groups | Single-label groups | Groups also containing another label |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary["class_group_counts"]:
        lines.append(f"| {row['partition']} | {row['class_name']} | {row['records']} | {row['distinct_frozen_groups']} | {row['single_label_groups']} | {row['mixed_folder_groups']} |")
    lines += ["", "The overall group compositions are:", "",
              "| Supplied labels within group | Groups | Records |", "| --- | ---: | ---: |"]
    for row in summary["group_compositions"]:
        if row["partition"] == "all":
            lines.append(f"| {row['class_composition']} | {row['n_groups']} | {row['n_records']} |")
    lines += ["", "A repeated ID spanning folder labels is a provenance question, not proof of a wrong label. Possible explanations include different examinations, changes over time, identifier reuse, source-label conventions, or inconsistent labeling. The available metadata does not distinguish these explanations. No automatic majority vote, relabeling or group deletion was performed.", "",
              "## Development-only adjudication queue", "",
              f"`results/readiness/labels/development_adjudication_queue.csv` contains every one of the {summary['development_adjudication_queue_records']} canonical training/validation records from {summary['development_adjudication_queue_groups']} groups. It contains zero test records. Each row preserves its canonical source path, sample ID, opaque group ID, supplied label and file/pixel/model-input hashes. It includes no model prediction, confidence, error flag or ranking based on model results.", "",
              "Rows from mixed-folder groups are listed first, followed by the small MI class and routine provenance checks. This ordering directs review effort without claiming an error. The fields `verified_label`, `reviewer`, `evidence` and `status` are all blank: a prepared queue is not a completed adjudication. The script refuses to overwrite the queue once any of these fields has been filled.", "",
              "A qualified reviewer should record the available original evidence, verify what the printed ID represents, and assess the supplied categories using a documented labeling policy. Review should allow an unresolved outcome when the source information is insufficient. The four broad folder categories do not establish a unified clinical taxonomy; an ECG page alone may not support the clinical history encoded by a folder name. Metadata consistency and accurate header transcription cannot clinically validate these labels.", "",
              "Any accepted corrections require a separately authorized, versioned label revision with its evidence and a declared evaluation policy. Preserve this frozen baseline cohort for comparison; do not silently edit its labels or tune corrections using test errors. The current review exports test information only as group-level technical metadata and aggregate counts, and creates no test adjudication queue.", "",
              "## Readiness and artifacts", "",
              "The dataset is technically ready for a prespecified next exploratory model on the same frozen split, subject to the project's model-definition decision. Clinical-ground-truth validation remains open. Results must continue to be described as prediction of supplied dataset categories, with limited minority-class support, uncertain patient identity and source-label limitations.", "",
              "- `results/readiness/labels/group_review.csv`: all opaque groups, partition, record counts and supplied-label composition; no printed ID values or model outcomes.",
              "- `results/readiness/labels/class_group_counts.csv`: distinct ID/group counts per class and partition, including mixed-group overlap.",
              "- `results/readiness/labels/group_compositions.csv`: group-composition counts for every partition and the complete cohort.",
              "- `results/readiness/labels/development_adjudication_queue.csv`: unfilled training/validation review form with original paths and hashes.",
              "- `results/readiness/labels/summary.json`: computed findings, limitations and input hashes.",
              "- `src/review_labels.py`: reproducible review, integrity checks and queue generation.", "",
              "Local evidence: `docs/dataset_audit.md`, `docs/source_references.md`, `data/audit/archive_provenance.json`, `data/audit/header_template_summary.json` and `data/audit/near_duplicate_summary.json`. The documented source attribution does not substitute for patient-level source metadata or independent label adjudication. No model outputs were opened by the helper and no external service was contacted.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    review(parser.parse_args().root)
