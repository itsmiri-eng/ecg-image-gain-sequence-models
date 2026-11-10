"""Verify baseline preservation and completed technical reviews, without fitting models."""
import argparse
import csv
from pathlib import Path

from src.common import ROOT, read_json, sha256_file, utc_now, write_json
from src.verify import load_verified_data


def check(protocol_path, output_path, require_model=False):
    protocol = read_json(protocol_path)
    lock_path = ROOT / protocol["baseline_lock"]
    lock = read_json(lock_path)
    changes = [name for name, expected in lock["files"].items()
               if not (ROOT / name).is_file() or sha256_file(ROOT / name) != expected]
    log_preservation = {"mode": "exact original log"}
    if changes == ["experiments/experiment_log.csv"]:
        from src.baseline_preservation import verify_baseline_preservation
        log_preservation = {"mode": "original byte prefix plus new experiment rows",
                            **verify_baseline_preservation()}
        changes = []
    if changes:
        raise ValueError(f"Protected baseline artifacts changed: {changes}")
    _, _, rows, parts, frozen = load_verified_data()
    audit = frozen["audit"]
    for field in ["dataset_version", "split_version", "split_sha256"]:
        if protocol[field] != audit[field]:
            raise ValueError(f"Readiness protocol mismatch: {field}")
    if protocol["class_order"] != frozen["config"]["class_names"]:
        raise ValueError("Class order changed")
    if protocol["seed"] != frozen["config"]["seed"]:
        raise ValueError("Readiness seed differs from baseline comparison seed")
    reviews = {
        "label_consistency": ROOT / "results/readiness/labels/summary.json",
        "minority_errors": ROOT / "results/readiness/errors/summary.json",
        "input_representation": ROOT / "docs/input_representation_review.md",
    }
    missing = [name for name, path in reviews.items() if not path.is_file()]
    scope_checks = {}
    if not missing:
        label_summary = read_json(reviews["label_consistency"])
        error_summary = read_json(reviews["minority_errors"])
        if label_summary["labels_changed"] or label_summary["split_changed"]:
            raise ValueError("The review must not silently change labels/splits")
        if not all(error_summary[k] for k in ["no_new_model_inference", "no_retraining", "no_test_case_gallery"]):
            raise ValueError("Error review scope differs from the declared post-baseline review")
        development = {r["sample_id"]: r for r in rows if r["split"] != "test"}
        queues = {
            "label_queue": ROOT / "results/readiness/labels/development_adjudication_queue.csv",
            "error_queue": ROOT / "results/readiness/errors/development_cases.csv",
        }
        for name, path in queues.items():
            with path.open() as f:
                records = list(csv.DictReader(f))
            if len(records) != len(development) or {r["sample_id"] for r in records} != set(development):
                raise ValueError(f"{name} is not exactly the development cohort")
            for r in records:
                source = development[r["sample_id"]]
                partition = r.get("partition", r.get("split"))
                if partition != source["split"] or r["file_sha256"] != source["file_sha256"] or r["supplied_label"] != source["class_name"]:
                    raise ValueError(f"{name}: source identity, partition, or label mismatch")
                if name == "label_queue":
                    if any(r[k] for k in ["verified_label", "reviewer", "evidence", "status"]):
                        raise ValueError("Clinical-adjudication fields unexpectedly populated")
                    if any("prediction" in k or "probability" in k for k in r):
                        raise ValueError("Model predictions must be excluded from the blind label queue")
            scope_checks[name] = {"records": len(records), "test_records": 0, "source_identity_matches": True}
    definition = protocol["model_definition"]
    model_defined = (definition["definition_status"] == "confirmed"
                     and bool(definition["confirmed_name"])
                     and bool(definition["definition_or_reference"]))
    blockers = []
    if missing:
        blockers.append("Incomplete technical review artifacts: " + ", ".join(missing))
    if not model_defined:
        blockers.append("Exact next-model name and sufficient definition/reference remain unconfirmed")
    if protocol["training_specification"] is None:
        blockers.append("Method-specific training configuration and selection rule are not yet frozen")
    result = {
        "checked_utc": utc_now(), "baseline_artifacts_preserved": len(lock["files"]),
        "changed_baseline_artifacts": changes, "data_integrity_checks_passed": True,
        "experiment_log_preservation": log_preservation,
        "partition_counts": {k: len(v) for k, v in parts.items()},
        "technical_reviews_complete": not missing,
        "development_review_queue_checks": scope_checks,
        "model_definition_confirmed": bool(model_defined),
        "ready_for_model_implementation": not missing and bool(model_defined),
        "ready_for_model_execution": not blockers,
        "outstanding_items": blockers,
        "clinical_labels_adjudicated": False,
        "verified_patient_identifiers_available": False,
        "new_model_runs_performed_by_this_check": 0,
        "new_test_inference_performed_by_this_check": False,
        "review_artifact_sha256": {name: sha256_file(path) for name, path in reviews.items() if path.is_file()},
        "protocol_sha256": sha256_file(protocol_path), "baseline_lock_sha256": sha256_file(lock_path),
    }
    write_json(output_path, result)
    print(f"Baseline preservation: PASS ({len(lock['files'])} artifacts)")
    print("Technical reviews:", "complete" if not missing else "pending: " + ", ".join(missing))
    print("Next-model execution:", "ready" if not blockers else "pending")
    for item in blockers:
        print(" -", item)
    if require_model and blockers:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/configs/next_model_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/readiness/readiness.json")
    parser.add_argument("--require-model", action="store_true", help="Exit 2 while a method or execution specification is pending")
    args = parser.parse_args()
    check(args.protocol, args.output, args.require_model)
