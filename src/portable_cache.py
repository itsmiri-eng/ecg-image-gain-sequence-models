"""Open the frozen v1 learning cache after the project folder has moved to another computer.

CachedECGDataset (src/ptbxl_learning_data.py) re-derives its frozen binding from the absolute
source path and the exact Python build string recorded on the Mac where the cache was built, so
it refuses to open anywhere else. That file is itself hash-bound and is not edited. This loader
repeats every check of its _validate_cache with two machine-specific exceptions: the source
folder is found relative to this project, and the Python build string may differ. Source-file
hashes, the v1 loader's code hash, preprocessing, class names, the numpy/OpenCV versions and the
per-image byte and pixel hashes (on every access) are still enforced. On the original computer
nothing differs and the check is identical to the strict one.
"""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

import numpy as np

from src import ptbxl_learning_data as v1

MACHINE_SPECIFIC = {"source_root", "python"}


def relocate(recorded_path, root=v1.ROOT):
    """Map an absolute path recorded on the original computer into this copy of the project (`root`)."""
    path = Path(recorded_path)
    if path.exists():
        return path
    parts = PurePosixPath(recorded_path).parts
    if "data" not in parts:
        raise FileNotFoundError(f"Cannot relocate {recorded_path} into {root}")
    return Path(root).joinpath(*parts[parts.index("data"):])


def located(recorded_path, current_path, root=v1.ROOT):
    """`recorded_path` if it names `current_path` (same path below the project folder), else `current_path`.

    Used by verifiers to accept absolute paths a run recorded on the computer it ran on.
    """
    if str(recorded_path) == str(current_path):
        return recorded_path
    tail = Path(current_path).resolve().relative_to(Path(root).resolve()).parts
    recorded = PurePosixPath(str(recorded_path).replace("\\", "/")).parts
    return recorded_path if recorded[-len(tail):] == tail else str(current_path)


def validate_cache(cache_root):
    """Same checks as v1._validate_cache, tolerant only of the machine-specific fields."""
    root = Path(cache_root)
    metadata, protocol = v1._read_json(root / "metadata.json"), v1._read_json(root / "protocol.json")
    if (metadata.get("status") != "complete" or metadata.get("input_shape") != list(v1.INPUT_SHAPE)
            or metadata.get("binding") != protocol.get("binding")):
        raise ValueError("Incomplete or incompatible learning cache")
    recorded = protocol["binding"]
    source_root = relocate(recorded["source_root"])
    current = v1._source_binding(source_root)
    differing = {key for key in recorded if key not in ("source_root", "runtime") and recorded[key] != current[key]}
    differing |= {key for key in recorded["runtime"] if recorded["runtime"][key] != current["runtime"][key]}
    if differing - MACHINE_SPECIFIC:
        raise ValueError(f"Cache code, source, configuration or runtime binding changed: {sorted(differing)}")
    if v1._sha(root / "cache_manifest.csv") != metadata["cache_manifest_sha256"]:
        raise ValueError("Cache manifest hash mismatch")
    rows = v1._parse_rows(v1._read_csv(root / "cache_manifest.csv"), cached=True)
    source_rows = v1.load_source_rows(source_root)
    if [{key: row[key] for key in source} for row, source in zip(rows, source_rows)] != source_rows:
        raise ValueError("Cached identities/targets differ from verified source manifest")
    normalization = metadata["normalization"]
    if (normalization.get("fit_partition") != "train" or normalization.get("training_images") != v1.EXPECTED_COUNTS["train"]
            or normalization.get("training_pixels") != v1.EXPECTED_COUNTS["train"] * math.prod(v1.INPUT_SHAPE)
            or not np.isfinite([normalization["mean"], normalization["std"]]).all()
            or normalization["std"] <= 0 or not 0 <= normalization["mean"] <= 1):
        raise ValueError("Invalid training normalization")
    check = {"mode": "relocated" if differing or str(source_root.resolve()) != recorded["source_root"] else "strict",
             "differences": {"source_root": [recorded["source_root"], str(source_root.resolve())],
                             "python": [recorded["runtime"]["python"], current["runtime"]["python"]]}}
    return rows, metadata, check


class PortableCachedECGDataset(v1.CachedECGDataset):
    """CachedECGDataset that also opens on a computer other than the one that built the cache."""
    def __init__(self, cache_root=v1.DEFAULT_CACHE, split="train", normalize=True):
        if split not in v1.EXPECTED_COUNTS:
            raise ValueError("split must be train, validation, or test")
        self.cache_root = Path(cache_root)
        rows, self.metadata, self.binding_check = validate_cache(self.cache_root)
        self.rows = [row for row in rows if row["split"] == split]
        self.normalization = self.metadata["normalization"]
        self.normalize = bool(normalize)
        self.split = split
