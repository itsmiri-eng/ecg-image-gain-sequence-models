"""Download the external-test data approved by the user on 3 Nov 2026 (docs/external_test_plan.md).

Usage:
    python -X utf8 -m src.download_external [--workers 8]
    python -X utf8 -m src.download_external --sets ptbxl-headers --report download_report_ptbxl_headers.json

Fetches, from PhysioNet (open access, CC BY 4.0):
- PTB-XL+ v1.0.1 labels/mapping/ptbxlToSNOMED.csv (+ the project's SHA256SUMS.txt to verify it);
- CinC Challenge 2021 v1.0.3 training/georgia, training/chapman_shaoxing, training/ningbo
  (+ the project's SHA256SUMS.txt, which lists every file and its hash).
Every file is checked against PhysioNet's SHA-256 before it is kept; files already present and correct are
skipped, so the script can be rerun after an interruption. Nothing is executed from the downloaded data.
Writes data/external/<project>/... and data/external/download_report.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from src.common import ROOT, utc_now

OUT = ROOT / "data/external"
CHALLENGE = "https://physionet.org/files/challenge-2021/1.0.3/"
SETS = {  # name: (local folder, base URL, wanted paths or folder prefixes, required suffix)
    "ptb-xl-plus": ("ptb-xl-plus", "https://physionet.org/files/ptb-xl-plus/1.0.1/", ("labels/mapping/ptbxlToSNOMED.csv",), ""),
    "challenge-2021": ("challenge-2021", CHALLENGE,
                       ("training/georgia/", "training/chapman_shaoxing/", "training/ningbo/"), ""),
    # Approved separately on 3 Nov 2026: label headers only (no signals) of the Challenge's PTB-XL copy, used to
    # recover the organisers' PTB-XL -> SNOMED CT conversion (docs/external_test_plan.md section 3).
    "ptbxl-headers": ("challenge-2021", CHALLENGE, ("training/ptb-xl/",), ".hea"),
}


def fetch(url, attempts=4):
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ecg-research-download/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def get_file(base, relative, digest, root):
    destination = root / relative
    if destination.is_file() and sha(destination.read_bytes()) == digest:
        return "present", destination.stat().st_size
    payload = fetch(base + relative)
    if sha(payload) != digest:
        raise ValueError(f"checksum mismatch: {relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return "downloaded", len(payload)


def main(workers, sets, report_name):
    report = {"started_utc": utc_now(), "projects": {}}
    for project in sets:
        folder, base, wanted, suffix = SETS[project]
        root = OUT / folder
        root.mkdir(parents=True, exist_ok=True)
        sums_file = root / "SHA256SUMS.txt"
        if sums_file.is_file():  # already fetched by an earlier set from the same project
            sums_payload = sums_file.read_bytes()
        else:
            sums_payload = fetch(base + "SHA256SUMS.txt")
            sums_file.write_bytes(sums_payload)
        entries = {}
        for line in sums_payload.decode("utf-8").splitlines():
            digest, _, relative = line.strip().partition(" ")
            relative = relative.strip().lstrip("*")
            if (relative and relative.endswith(suffix)
                    and any(relative == w or (w.endswith("/") and relative.startswith(w)) for w in wanted)):
                entries[relative] = digest
        if not entries:
            raise ValueError(f"{project}: no wanted files found in SHA256SUMS.txt")
        counts, size, failures = {"present": 0, "downloaded": 0}, 0, []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(get_file, base, r, d, root): r for r, d in entries.items()}
            for done, job in enumerate(as_completed(jobs), start=1):
                try:
                    status, nbytes = job.result()
                    counts[status] += 1
                    size += nbytes
                except Exception as error:
                    failures.append({"file": jobs[job], "error": repr(error)})
                if done % 2000 == 0 or done == len(jobs):
                    print(f"{utc_now()} {project}: {done}/{len(jobs)} files, {size / 1e9:.2f} GB, "
                          f"failures {len(failures)}", flush=True)
        report["projects"][project] = {"base_url": base, "wanted": list(wanted), "files": len(entries), **counts,
                                       "bytes": size, "failures": failures,
                                       "sha256sums_sha256": sha(sums_payload)}
    report["completed_utc"] = utc_now()
    (OUT / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({p: {k: v for k, v in r.items() if k != "failures"} | {"failures": len(r["failures"])}
                      for p, r in report["projects"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sets", nargs="+", choices=list(SETS), default=["ptb-xl-plus", "challenge-2021"])
    parser.add_argument("--report", default="download_report.json")
    arguments = parser.parse_args()
    main(arguments.workers, arguments.sets, arguments.report)
