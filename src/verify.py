"""Validate the frozen split and cache without fitting or scoring any model."""
import csv
import hashlib
from itertools import combinations

import numpy as np

from src.common import ROOT, digest, read_json, sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_verified_data(verify_originals=True):
    protocol = read_json(ROOT / "data/splits/protocol.json")
    audit = protocol["audit"]
    split_path = ROOT / "data/splits/fixed_split.csv"
    cache_path = ROOT / "data/processed/inputs_v1.npz"
    require(sha256_file(split_path) == audit["split_sha256"], "Split changed")
    require(sha256_file(cache_path) == audit["cache_sha256"], "Input cache changed")
    require(digest(protocol["config"]) == audit["config_content_sha256"], "Frozen configuration changed")
    require(sha256_file(ROOT / "data/audit/header_ids.csv") == audit["header_ids_sha256"], "Identity audit changed")
    require(sha256_file(ROOT / "data/audit/source_manifest.csv") == audit["source_manifest_sha256"], "Source manifest changed")
    links = ROOT / "data/audit/near_duplicate_links.csv"
    require((sha256_file(links) if links.exists() else None) == audit["near_duplicate_links_sha256"], "Near-copy grouping audit changed")
    with split_path.open() as f:
        rows = list(csv.DictReader(f))
    data = np.load(cache_path, allow_pickle=False)
    x, y = data["images"], data["labels"]
    require(len(x) == len(y) == len(rows) == audit["retained_samples"], "Inconsistent sample counts")
    require(list(x.shape[1:]) == audit["model_input_shape"] and x.dtype == np.float32, "Wrong input shape/dtype")
    require(y.ndim == 1 and np.issubdtype(y.dtype, np.integer), "Invalid labels")
    require(sorted(int(r["array_index"]) for r in rows) == list(range(len(rows))), "Missing/repeated array indices")
    require(np.isfinite(x).all() and float(x.min()) >= 0 and float(x.max()) <= 1, "Invalid image values")
    require(all(r["split"] in {"train", "validation", "test"} for r in rows), "Unknown partition")
    rows.sort(key=lambda r: int(r["array_index"]))
    for i, r in enumerate(rows):
        require(int(r["label"]) == y[i], "Manifest/cache label mismatch")
        require(r["class_name"] == protocol["config"]["class_names"][int(y[i])], "Class name mismatch")
        require(hashlib.sha256(x[i].tobytes()).hexdigest() == r["input_sha256"], "Input row changed")
        if verify_originals:
            require(sha256_file(ROOT / r["path"]) == r["file_sha256"], f"Original changed: {r['path']}")
    subsets = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s])
               for s in ["train", "validation", "test"]}
    for s, indices in subsets.items():
        require(set(y[indices]) == {0, 1, 2, 3}, f"Absent class in {s}")
    for a, b in combinations(subsets, 2):
        for key in ["sample_id", "file_sha256", "pixel_sha256", "input_sha256", "group_id"]:
            require(not {rows[i][key] for i in subsets[a]} & {rows[i][key] for i in subsets[b]}, f"Shared {key} across {a}/{b}")
    return x, y, rows, subsets, protocol


if __name__ == "__main__":
    x, y, rows, subsets, protocol = load_verified_data()
    print(f"PASS: {len(rows)} originals/cache entries verified; no shared identity/duplicate groups across splits.")
    print({s: len(i) for s, i in subsets.items()})
