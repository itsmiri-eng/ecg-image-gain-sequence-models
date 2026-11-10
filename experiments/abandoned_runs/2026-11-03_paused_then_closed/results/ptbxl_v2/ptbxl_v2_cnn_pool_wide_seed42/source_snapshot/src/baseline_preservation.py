"""Protect the original baselines while allowing new experiment-log rows."""
import csv
import io

from src.common import ROOT, read_json, sha256_file

LOCK = ROOT / "experiments/reviews/2026-09-24/baseline_lock.json"
LOG = ROOT / "experiments/experiment_log.csv"
SNAPSHOT = ROOT / "experiments/protocols/cnn_lstm_image_v1/baseline_log_before.csv"


def verify_baseline_preservation():
    locked = read_json(LOCK)["files"]
    log_name = LOG.relative_to(ROOT).as_posix()  # lock keys use "/" on every OS
    changed = [name for name, digest in locked.items() if name != log_name
               and (not (ROOT / name).is_file() or sha256_file(ROOT / name) != digest)]
    if changed:
        raise ValueError(f"Protected baseline artifacts changed: {changed}")
    if sha256_file(SNAPSHOT) != locked[log_name]:
        raise ValueError("Original experiment-log snapshot does not match the original lock")
    original, current = SNAPSHOT.read_bytes(), LOG.read_bytes()
    if not current.startswith(original):
        raise ValueError("The experiment log must preserve the exact original byte prefix")
    old_reader = csv.DictReader(io.StringIO(original.decode()))
    reader = csv.DictReader(io.StringIO(current.decode()))
    old_rows, rows = list(old_reader), list(reader)
    if reader.fieldnames != old_reader.fieldnames or rows[:len(old_rows)] != old_rows:
        raise ValueError("Baseline log schema or original rows changed")
    ids = [row["experiment_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("Malformed or duplicate experiment-log rows")
    return {"immutable_baseline_artifacts_checked": len(locked) - 1,
            "changed_baseline_artifacts": [], "original_log_sha256": sha256_file(SNAPSHOT),
            "original_log_byte_prefix_preserved": True, "original_log_rows_preserved": len(old_rows),
            "appended_log_rows": len(rows) - len(old_rows), "current_log_sha256": sha256_file(LOG)}


def append_experiment_log(record):
    verify_baseline_preservation()
    with LOG.open(newline="") as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    if set(record) != set(fields):
        raise ValueError("New experiment record must match the existing log schema")
    if any(row["experiment_id"] == record["experiment_id"] for row in rows):
        raise ValueError("Experiment ID already logged")
    buffer = io.StringIO(newline="")
    csv.DictWriter(buffer, fieldnames=fields).writerow(record)
    with LOG.open("ab") as f:
        f.write(buffer.getvalue().encode())
    verify_baseline_preservation()


if __name__ == "__main__":
    import json
    print(json.dumps(verify_baseline_preservation(), indent=2))
