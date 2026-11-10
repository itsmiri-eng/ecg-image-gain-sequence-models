"""Build the SNOMED CT -> PTB-XL superclass mapping for the external test (docs/external_test_plan.md section 3).

Usage:
    python -X utf8 -m src.external_label_mapping

Reads only label text: the "# Dx:" lines of the Challenge 2021 copy of PTB-XL (training/ptb-xl/*.hea, record
HRxxxxx = PTB-XL ecg_id xxxxx) and this project's PTB-XL superclass labels (classification manifest; records
outside the 21,388-record cohort have no superclass). No external test ECG (Georgia, Chapman-Shaoxing, Ningbo)
is opened.

Rule fixed in the plan before this ran: a SNOMED code is assigned to a superclass when at least 90% of the
PTB-XL records carrying the code carry that superclass, among codes seen on at least 10 PTB-XL records.
Alignment check: header age/sex must agree with ptbxl_database.csv for the same ecg_id.
Writes experiments/external/label_mapping.{csv,json}.
"""
from __future__ import annotations

import csv
import json
import re

from src.common import ROOT, sha256_file, utc_now, write_csv

CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
HEADERS = ROOT / "data/external/challenge-2021/training/ptb-xl"
PTBXL = ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT = ROOT / "experiments/external"
SHARE, MIN_RECORDS = 0.90, 10


def read_header(path):
    fields = {}
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        match = re.match(r"#\s*(Age|Sex|Dx):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def main():
    with (ROOT / "data/derived/ptbxl_images_v1/classification_manifest.csv").open(newline="") as handle:
        labels = {int(r["ecg_id"]): {c for c in CLASS_NAMES if r[f"label_{c}"] in ("1", "True", "true")}
                  for r in csv.DictReader(handle)}
    with (PTBXL / "ptbxl_database.csv").open(newline="") as handle:
        database = {int(float(r["ecg_id"])): r for r in csv.DictReader(handle)}
    codes, checked, mismatched, unmatched = {}, 0, [], 0
    for path in sorted(HEADERS.rglob("HR*.hea")):
        ecg_id = int(path.stem[2:])
        if ecg_id not in database:
            unmatched += 1  # in PTB-XL v1.0.1 (used by the Challenge) but removed in v1.0.3
            continue
        fields, row = read_header(path), database[ecg_id]
        sex = {"0": "Male", "1": "Female"}.get(str(int(float(row["sex"]))), "?")
        age_ok = fields.get("Age") in ("Unknown", "NaN") or row["age"] in ("", "nan") or \
            abs(float(fields["Age"]) - float(row["age"])) < 1
        checked += 1
        if fields.get("Sex") != sex or not age_ok:
            mismatched.append(ecg_id)
        for code in filter(None, (c.strip() for c in fields.get("Dx", "").split(","))):
            codes.setdefault(code, []).append(labels.get(ecg_id, set()))
    if checked < 21000 or len(mismatched) > 0.01 * checked:
        raise ValueError(f"Header/PTB-XL alignment failed: {checked} checked, {len(mismatched)} mismatched")
    rows = []
    for code, sets in sorted(codes.items(), key=lambda item: -len(item[1])):
        shares = {c: sum(c in s for s in sets) / len(sets) for c in CLASS_NAMES}
        assigned = [c for c in CLASS_NAMES if len(sets) >= MIN_RECORDS and shares[c] >= SHARE]
        rows.append({"snomed_code": code, "ptbxl_records": len(sets),
                     **{f"share_{c}": round(shares[c], 4) for c in CLASS_NAMES},
                     "assigned_superclasses": "|".join(assigned)})
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "label_mapping.csv", rows)
    summary = {"created_utc": utc_now(), "rule": f"share >= {SHARE} among codes on >= {MIN_RECORDS} PTB-XL records",
               "headers_checked": checked, "age_sex_mismatches": len(mismatched), "mismatch_ids_first20": mismatched[:20],
               "challenge_records_not_in_ptbxl_v103": unmatched, "codes_seen": len(rows),
               "codes_assigned": sum(bool(r["assigned_superclasses"]) for r in rows),
               "by_superclass": {c: [r["snomed_code"] for r in rows if c in r["assigned_superclasses"].split("|")]
                                 for c in CLASS_NAMES},
               "challenge_sha256sums_sha256": sha256_file(ROOT / "data/external/challenge-2021/SHA256SUMS.txt")}
    (OUT / "label_mapping.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
