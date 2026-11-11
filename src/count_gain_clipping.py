"""How many PTB-XL pages would need clipping if every page were drawn at one fixed gain (2.5, 5 or 10 mm/mV).

Usage:
    python -X utf8 -m src.count_gain_clipping --workers 2

Descriptive only; it trains nothing and writes no image. It imports the unchanged renderer
(src/ptbxl_rendering.py, never edited) and the fixed-gain geometry of src/ptbxl_fixed_gain.py, and applies that
geometry's clipping rule (a point clips when it falls outside its lead's 30-mm row) at each candidate gain. The
2.5 mm/mV count must reproduce the clipped_point_count stored in the fixed-gain cache for every ECG, which
checks that the rule is applied exactly as in the dataset build. Writes results/fixed_gain/clipping_by_gain/.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv

from src import ptbxl_fixed_gain, ptbxl_rendering
from src.common import ROOT, read_json, sha256_file, utc_now, write_json

GAINS = (2.5, 5.0, 10.0)
OUT = ROOT / "results/fixed_gain/clipping_by_gain"


def _job(task):
    row, config = task
    base = ptbxl_fixed_gain.SOURCE / row["filename_hr"]
    for suffix, key in ((".hea", "source_header_sha256"), (".dat", "source_waveform_sha256")):
        if sha256_file(f"{base}{suffix}") != row[key]:
            raise ValueError(f"Source checksum changed: ECG {row['ecg_id']}")
    signal, _ = ptbxl_rendering.load_waveform(base, config)
    counts = {}
    original = ptbxl_fixed_gain.FIXED_GAIN
    try:
        for gain in GAINS:
            ptbxl_fixed_gain.FIXED_GAIN = gain
            counts[gain] = ptbxl_fixed_gain.fixed_gain_geometry(signal, config)[1]["clipped_point_count"]
    finally:
        ptbxl_fixed_gain.FIXED_GAIN = original
    return {"ecg_id": row["ecg_id"], "split": row["split"], "adaptive_gain_mm_per_mv": row["v1_gain_mm_per_mv"],
            "label_HYP": row["label_HYP"], **{f"clipped_points_at_{g:g}": counts[g] for g in GAINS}}


def main(workers):
    config = read_json(ROOT / "experiments/configs/ptbxl_images_v1.json")
    with (ptbxl_fixed_gain.OUT / "cache_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_job, [(r, config) for r in rows], chunksize=64))
    mismatches = [r["ecg_id"] for r, m in zip(results, rows) if r["clipped_points_at_2.5"] != int(m["clipped_point_count"])]
    if mismatches:
        raise ValueError(f"2.5 mm/mV counts do not reproduce the fixed-gain cache for {len(mismatches)} ECGs: {mismatches[:5]}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_ecg.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {"created_utc": utc_now(), "ecgs": len(results),
               "rule": "a sample clips when it falls outside its lead's 30-mm row (src.ptbxl_fixed_gain.fixed_gain_geometry)",
               "check": "2.5 mm/mV counts reproduce clipped_point_count of the fixed-gain cache for every ECG",
               "by_gain": {}}
    for g in GAINS:
        clipped = [r for r in results if r[f"clipped_points_at_{g:g}"] > 0]
        summary["by_gain"][f"{g:g}"] = {
            "pages_with_any_clipping": len(clipped),
            "percent_of_pages": 100 * len(clipped) / len(results),
            "by_split": {s: sum(r["split"] == s for r in clipped) for s in ("train", "validation", "test")},
            "hyp_positive_pages_with_clipping": sum(r["label_HYP"] == "True" for r in clipped)}
    write_json(OUT / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args().workers)
