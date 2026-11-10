"""Audit originals, exclude exact copies, freeze one group-aware split, cache tiny inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.common import ROOT, digest, read_json, sha256_file, utc_now, write_csv, write_json
from src.preprocessing.images import preprocess_image


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def prepare(config_path):
    config = read_json(config_path)
    if config["split"]["method"] != "stratified_group_5_fold_then_4_fold":
        raise ValueError("Unsupported split method")
    if config["split"]["target_proportions"] != [0.6, 0.2, 0.2]:
        raise ValueError("This grouping procedure targets 60/20/20 proportions")
    if config["cnn"]["seed"] != config["seed"]:
        raise ValueError("The baseline protocol uses one fixed seed for both models")
    split_path = ROOT / "data/splits/fixed_split.csv"
    if split_path.exists():
        raise FileExistsError("Fixed split already exists; use src.verify to check it. Never silently replace it.")
    id_path = ROOT / "data/audit/header_ids.csv"
    if not id_path.exists():
        raise FileNotFoundError("Complete the printed-ID audit before preparing the split")
    with id_path.open() as f:
        id_rows = list(csv.DictReader(f))
    ids = {r["file_sha256"]: r for r in id_rows}
    inventory, excluded, by_pixel = [], [], defaultdict(list)
    for label, (name, folder) in enumerate(zip(config["class_names"], config["class_folders"])):
        directory = ROOT / config["raw_directory"] / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            row = {"path": str(p.relative_to(ROOT)), "class_name": name, "label": label,
                   "file_sha256": sha256_file(p), "bytes": p.stat().st_size}
            try:
                with Image.open(p) as im:
                    im.load()
                    row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                    rgb = im.convert("RGB")
                    row["pixel_sha256"] = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                row["decode_status"] = "valid"
                by_pixel[row["pixel_sha256"]].append(row)
            except Exception as e:
                row["decode_status"] = f"{type(e).__name__}: {e}"
                row.update(width="", height="", mode="", format="", pixel_sha256="")
                excluded.append({"path": row["path"], "reason": "corrupt", "canonical_path": ""})
            inventory.append(row)
    write_csv(ROOT / "data/audit/source_manifest.csv", inventory)
    kept, duplicates, conflicts, equivalent_hashes = [], [], [], {}
    for pixel_hash, group in sorted(by_pixel.items()):
        group.sort(key=lambda r: r["path"])
        if len({r["label"] for r in group}) > 1:
            for row in group:
                excluded.append({"path": row["path"], "reason": "cross_class_exact_duplicate_conflict", "canonical_path": ""})
                conflicts.append(row["path"])
            continue
        canonical = group[0]
        for row in group[1:]:
            record = {"path": row["path"], "reason": "exact_decoded_pixel_duplicate", "canonical_path": canonical["path"]}
            excluded.append(record)
            duplicates.append(record)
        id_row = ids.get(canonical["file_sha256"])
        if id_row is None:
            # Other compressed versions of identical pixels may have an audited header.
            id_row = next((ids[r["file_sha256"]] for r in group if r["file_sha256"] in ids), None)
        if id_row is None or not id_row.get("printed_id", "").strip():
            excluded.append({"path": canonical["path"], "reason": "unresolved_printed_id", "canonical_path": ""})
            continue
        canonical = dict(canonical)
        canonical.update(sample_id=pixel_hash[:20], printed_id=id_row["printed_id"].strip(),
                         id_status=id_row.get("status", ""), duplicate_group_size=len(group))
        equivalent_hashes[canonical["file_sha256"]] = [r["file_sha256"] for r in group]
        kept.append(canonical)
    kept.sort(key=lambda r: r["sample_id"])
    if len(kept) < 20:
        raise ValueError("Insufficient identified unique images for a four-class three-way split")
    images = np.stack([preprocess_image(ROOT / r["path"], config["preprocessing"]) for r in kept])
    union = UnionFind(len(kept))
    known_id, known_input = {}, {}
    for i, (row, x) in enumerate(zip(kept, images)):
        row["input_sha256"] = hashlib.sha256(x.tobytes()).hexdigest()
        for value, table in [(row["printed_id"], known_id), (row["input_sha256"], known_input)]:
            if value in table:
                union.union(i, table[value])
            table[value] = i
    # Optional audited near-copy links are grouping constraints, never guessed labels/deletions.
    links_path = ROOT / "data/audit/near_duplicate_links.csv"
    if links_path.exists():
        lookup = {h: i for i, r in enumerate(kept) for h in equivalent_hashes[r["file_sha256"]]}
        with links_path.open() as f:
            for link in csv.DictReader(f):
                if link["sha256_a"] in lookup and link["sha256_b"] in lookup:
                    union.union(lookup[link["sha256_a"]], lookup[link["sha256_b"]])
    groups = np.array([str(union.find(i)) for i in range(len(kept))])
    # Replace unstable integer group IDs with reproducible opaque IDs (avoid distributing printed IDs).
    group_names = {g: digest(sorted(r["sample_id"] for r, gg in zip(kept, groups) if gg == g))[:20]
                   for g in set(groups)}
    groups = np.array([group_names[g] for g in groups])
    y = np.array([r["label"] for r in kept], dtype=np.int64)
    for label in range(4):
        if len(set(groups[y == label])) < 5:
            raise ValueError(f"Class {config['class_names'][label]} has fewer than five independent groups")
    seed = config["seed"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, test = list(splitter.split(images, y, groups))[config["split"]["test_fold"]]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_local, val_local = list(inner.split(images[dev], y[dev], groups[dev]))[config["split"]["validation_fold"]]
    partitions = {"train": dev[train_local], "validation": dev[val_local], "test": test}
    split_rows = []
    for subset, indices in partitions.items():
        for label in range(4):
            if not np.any(y[indices] == label):
                raise ValueError(f"Prespecified split lacks {config['class_names'][label]} in {subset}; no model run")
        for i in indices:
            row = kept[i]
            split_rows.append({"array_index": int(i), "sample_id": row["sample_id"], "path": row["path"],
                               "label": row["label"], "class_name": row["class_name"], "split": subset,
                               "group_id": groups[i], "file_sha256": row["file_sha256"],
                               "pixel_sha256": row["pixel_sha256"], "input_sha256": row["input_sha256"]})
    split_rows.sort(key=lambda r: r["array_index"])
    counts = lambda rr: dict(Counter(r["class_name"] for r in rr))
    printed_id_groups = defaultdict(list)
    for row in kept:
        printed_id_groups[row["printed_id"]].append(row)
    dataset_version = "image_dedup_" + digest([{k: r[k] for k in ["path", "file_sha256"]} for r in kept])[:12]
    split_version = "grouped_" + digest(split_rows)[:12]
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "data/processed/inputs_v1.npz", images=images, labels=y)
    write_csv(split_path, split_rows)
    write_csv(ROOT / "data/audit/exclusions.csv", excluded, ["path", "reason", "canonical_path"])
    hidden_images = [str(p.relative_to(ROOT)) for p in (ROOT / config["raw_directory"]).rglob("*.jpg")
                     if any(part.startswith(".") for part in p.relative_to(ROOT / config["raw_directory"]).parts)]
    audit = {
        "created_utc": utc_now(), "dataset_version": dataset_version, "split_version": split_version,
        "source_samples": len(inventory), "source_counts": counts(inventory),
        "decoded_unique_images": len(by_pixel), "redundant_exact_copies_excluded": len(duplicates),
        "cross_class_exact_duplicate_conflicts": conflicts,
        "corrupt_samples": sum(r["decode_status"] != "valid" for r in inventory),
        "missing_samples": 0, "missing_definition": "Every enumerated source manifest path was readable; no claim of source cohort completeness",
        "retained_samples": len(kept), "retained_counts": counts(kept),
        "excluded_counts_by_reason": dict(Counter(r["reason"] for r in excluded)),
        "source_dimensions": dict(Counter(f"{r['width']}x{r['height']} {r['mode']}" for r in inventory)),
        "model_input_shape": list(images.shape[1:]), "ignored_checkpoint_images": hidden_images,
        "printed_id_count": len({r["printed_id"] for r in kept}), "group_count": len(set(groups)),
        "printed_ids_with_multiple_unique_images": sum(len(rr) > 1 for rr in printed_id_groups.values()),
        "maximum_images_per_printed_id": max(map(len, printed_id_groups.values())),
        "printed_ids_spanning_folder_classes": sum(len({r["label"] for r in rr}) > 1 for rr in printed_id_groups.values()),
        "patient_id_limitation": "Printed report IDs are available and grouped conservatively. Their relationship to unique patients is unverified; patient independence is not established.",
        "partitions": {s: {"samples": len(idx), "groups": len(set(groups[idx])),
                            "class_counts": counts([kept[i] for i in idx])} for s, idx in partitions.items()},
        "preprocessing": config["preprocessing"], "seed": seed,
        "split_policy": config["split"], "split_chosen_without_model_results": True,
        "split_sha256": sha256_file(split_path),
        "cache_sha256": sha256_file(ROOT / "data/processed/inputs_v1.npz"),
        "header_ids_sha256": sha256_file(id_path), "config_sha256": sha256_file(config_path),
        "config_content_sha256": digest(config),
        "source_manifest_sha256": sha256_file(ROOT / "data/audit/source_manifest.csv"),
        "near_duplicate_links_sha256": sha256_file(links_path) if links_path.exists() else None,
    }
    for i, a in enumerate(partitions):
        for b in list(partitions)[i + 1:]:
            assert not set(groups[partitions[a]]) & set(groups[partitions[b]])
    write_json(ROOT / "data/audit/dataset_audit.json", audit)
    write_json(ROOT / "data/splits/protocol.json", {"config": config, "audit": audit})
    print(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "experiments/configs/baseline_v1.json"))
    prepare(parser.parse_args().config)
