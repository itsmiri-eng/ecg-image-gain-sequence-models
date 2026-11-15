"""Label-mapping sensitivity: render and score the external records that had no mapped superclass.

Additional analysis of docs/protocol_amendment_2026-11-15_label_mapping_sensitivity.md (frozen before any of
these records was rendered or given to a model). The frozen confirmatory analysis is not touched: its cache,
predictions and outputs are only read.

Usage:
    python -X utf8 -m src.external_allrecords render --dataset georgia --workers 3
    python -X utf8 -m src.external_allrecords infer --dataset georgia --device cuda     # once per cohort

render   every record whose only exclusion reason was "no_mapped_superclass" is drawn with the unchanged adaptive
         and fixed-gain renderers (src.external_test._render_job) into data/external/cache_unmapped/<cohort>/.
infer    every completed run that has frozen external predictions (plus the ResNet-18 runs of the 11 Nov
         amendment) is reloaded, must reproduce its saved PTB-XL validation probabilities, and then gets exactly
         one inference on these records, written to results/external_unmapped/<cohort>/<run>/. Never repeated.
The analysis of the variants is in src/mapping_sensitivity.py.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import multiprocessing

import numpy as np

from src import external_test as ext
from src.common import ROOT, read_json, sha256_file, utc_now, write_csv, write_json

AMENDMENT = "docs/protocol_amendment_2026-11-15_label_mapping_sensitivity.md"
CLASS_NAMES = ext.CLASS_NAMES


def cache_dir(dataset):
    return ROOT / "data/external/cache_unmapped" / dataset


def result_dir(dataset):
    return ROOT / "results/external_unmapped" / dataset


def render(dataset, workers=3):
    ext.check_freeze()
    if not (ROOT / AMENDMENT).exists():
        raise FileNotFoundError(AMENDMENT)
    out = cache_dir(dataset)
    if (out / "manifest.csv").exists():
        raise FileExistsError("Cache already built; never rebuilt in place")
    with (ext.experiment_dir(dataset) / "inclusion.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "no_mapped_superclass"]
    for kind in ("adaptive", "fixed"):
        (out / kind).mkdir(parents=True, exist_ok=True)
        for stale in (out / kind).glob("*.png"):  # leftovers of an interrupted render (pages are written with "xb")
            stale.unlink()
    config = read_json(ROOT / "experiments/configs/ptbxl_images_v1.json")
    results = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(ext._render_job, [(r, config, str(out)) for r in rows], chunksize=16):
            results.append(result)
            if len(results) % 2000 == 0:
                print(f"{utc_now()} rendered {len(results)}/{len(rows)}", flush=True)
    kept = [r for r in results if r["finite"]]
    np.save(out / "signals100.npy", np.stack([r.pop("signal100") for r in kept]))
    by_record = {r["record"]: r for r in kept}
    manifest = [{**{k: row[k] for k in ("record", "source", "dx", *[f"label_{c}" for c in CLASS_NAMES])},
                 **{k: v for k, v in by_record[row["record"]].items() if k not in ("record", "finite")}}
                for row in rows if row["record"] in by_record]
    write_csv(out / "manifest.csv", manifest)
    write_json(out / "metadata.json", {"created_utc": utc_now(), "dataset": dataset, "amendment": AMENDMENT,
               "records": len(manifest), "excluded_nonfinite": len(results) - len(kept),
               "nonfinite_records": [r["record"] for r in results if not r["finite"]],
               "manifest_sha256": sha256_file(out / "manifest.csv"), "signals100_sha256": sha256_file(out / "signals100.npy"),
               "code_sha256": {n: sha256_file(ROOT / n) for n in ("src/external_test.py", "src/external_allrecords.py",
                                                                 "src/ptbxl_rendering.py", "src/ptbxl_fixed_gain.py",
                                                                 "src/ptbxl_learning_data.py")},
               "render_config_sha256": sha256_file(ROOT / "experiments/configs/ptbxl_images_v1.json")})
    print(f"{utc_now()} cache complete: {len(manifest)} records, {len(results) - len(kept)} excluded as non-finite")


def scored_runs(dataset):
    """Runs with completed frozen external predictions, plus the ResNet-18 runs (their own result folder)."""
    from src import amendment_20261111 as amend
    frozen = ext.result_dir(dataset)
    runs = [r for r in ext.completed_runs() if (frozen / r[0] / "inference_state.json").exists()
            and read_json(frozen / r[0] / "inference_state.json")["status"] == "completed"]
    resnets = [r for r in amend.resnet_runs()
               if (amend.OUT / "external" / dataset / r[0] / "inference_state.json").exists()]
    return runs, resnets


def infer(dataset, device="cuda", tolerance=1e-3):
    import torch
    from src import amendment_20261111 as amend
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ext.check_freeze()
    cache = cache_dir(dataset)
    metadata = read_json(cache / "metadata.json")
    if sha256_file(cache / "manifest.csv") != metadata["manifest_sha256"] or \
            sha256_file(cache / "signals100.npy") != metadata["signals100_sha256"]:
        raise ValueError("Unmapped-record cache changed since it was built")
    out = result_dir(dataset)
    runs, resnets = scored_runs(dataset)
    # Only runs that have no folder yet are scored, so a run added later (for example the fixed-gain
    # capacity-matched runs of 15 Nov 2026) gets its single inference without any other run being repeated.
    everything = [r for r in runs + resnets if not (out / r[0]).exists()]
    if not everything:
        raise FileExistsError("Every run already has its inference on the unmapped records; never repeated")
    runs = [r for r in runs if r in everything]
    resnets = [r for r in resnets if r in everything]
    target = torch.device(device)
    models = {**ext.load_models(runs, target), **amend.load_resnets(resnets, target)}
    check = ext.reproduction_check(everything, models, target)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"reproduction_check_{utc_now()[:19].replace(':', '')}.json", {"checked_utc": utc_now(), "tolerance": tolerance, "amendment": AMENDMENT,
                                                  "max_abs_difference": check})
    failed = {k: v for k, v in check.items() if v > tolerance}
    if failed:
        raise ValueError(f"Reloaded models do not reproduce PTB-XL validation probabilities: {failed}")
    with (cache / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    signals = np.load(cache / "signals100.npy")
    for run_id, *_ in everything:
        (out / run_id).mkdir(parents=True)
        write_json(out / run_id / "inference_state.json", {"status": "started", "started_utc": utc_now(), "dataset": dataset,
                                                          "inference_calls": 1, "amendment": AMENDMENT})
    parts = {run_id: [] for run_id, *_ in everything}
    decoders = ThreadPoolExecutor(6)  # page decoding and checksum only; the order of records is unchanged
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        names = [r["record"].replace("/", "__") for r in batch]
        stacks = {page: np.stack(list(decoders.map(
            lambda item, page=page: ext._decode(cache / page / f"{item[0]}.png", item[1][f"{page}_pixel_sha256"]),
            zip(names, batch)))) for page in ("adaptive", "fixed")}
        for run_id, p in ext.predict(models, everything, stacks, signals[start:start + 64], target).items():
            parts[run_id].append(p)
        if start % 1600 == 0:
            print(f"{utc_now()} {dataset} inference {start}/{len(rows)}", flush=True)
    for run_id, family, kind, page, head, seed in everything:
        p = np.concatenate(parts[run_id])
        write_csv(out / run_id / "predictions.csv", [{"record": r["record"],
                  **{f"probability_{c}": float(p[i, j]) for j, c in enumerate(CLASS_NAMES)}} for i, r in enumerate(rows)])
        write_json(out / run_id / "inference_state.json", {"status": "completed", "completed_utc": utc_now(), "dataset": dataset,
                   "inference_calls": 1, "records": len(rows), "family": family, "kind": kind, "page": page, "head": head,
                   "seed": seed, "amendment": AMENDMENT, "predictions_sha256": sha256_file(out / run_id / "predictions.csv")})
    print(f"{utc_now()} {dataset}: inference on unmapped records complete ({len(everything)} runs x {len(rows)} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["render", "infer"])
    parser.add_argument("--dataset", choices=list(ext.DATASETS), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.step == "render":
        render(arguments.dataset, arguments.workers)
    else:
        infer(arguments.dataset, arguments.device)
