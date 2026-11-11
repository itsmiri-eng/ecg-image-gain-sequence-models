"""Analyses of the 11 Nov 2026 amendment (docs/protocol_amendment_2026-11-11.md).

Usage:
    python -X utf8 -m src.amendment_20261111 external-infer --dataset georgia   # once per cohort, ResNet-18 runs
    python -X utf8 -m src.amendment_20261111 analyze                            # PTB-XL and external tables

Part A (40-epoch sensitivity): selected epochs; test/validation macro AUROC per head next to the 20-epoch runs
of the same seeds; 40 - 20 epochs per head and the 10 head pairs at 40 epochs, both with a hierarchical seed x
patient bootstrap (2,000 resamples, seed 2024) and Holm within each family.

Part B (pretrained ResNet-18): PTB-XL test fixed - adaptive gain (macro and HYP AUROC, seed x patient bootstrap,
Holm over 2). On the external cohorts, every ResNet-18 run is first reloaded and must reproduce its saved PTB-XL
validation probabilities; it then gets exactly one inference per cohort, with the frozen inclusion, mapping and
eligibility. Fixed - adaptive (seed x record bootstrap) uses Holm over the 4 cohort x metric comparisons.
Predictions go to results/amendment_20261111/external/<cohort>/<run>/, apart from the frozen external results,
so the confirmatory analysis is untouched.
"""
from __future__ import annotations

import argparse
import csv
import itertools

import numpy as np

from src import external_test as ext
from src.common import ROOT, read_json, sha256_file, utc_now, write_csv, write_json
from src.ptbxl_learning_data import CLASS_NAMES

OUT = ROOT / "results/amendment_20261111"
AMENDMENT = "docs/protocol_amendment_2026-11-11.md"
HEADS = ("cnn_pool", "cnn_pool_wide", "cnn_lstm", "temporal_conv", "hybrid_vit")
SEEDS = (42, 43, 44)
RESNET = {"adaptive": "ptbxl_v2_resnet18_seed{}", "fixed": "ptbxl_v2_fg_resnet18_seed{}"}


FAMILIES = ("ptbxl_v2", "ptbxl_v2_fixedgain", "ptbxl_v2_long40", "ptbxl_v2_resnet18", "ptbxl_cnn_lstm",
            "ptbxl_spatiotemporal", "ptbxl_hybrid_vit")
FG_HEADS = ("cnn_pool", "cnn_lstm", "temporal_conv", "hybrid_vit")
# The 20-epoch seed-42 runs of three heads are the original v1 runs, as in the paper (src/compare_heads_hierarchical.py)
V1_SEED42 = {"cnn_lstm": "ptbxl_cnn_lstm_image_v1_seed42", "temporal_conv": "ptbxl_spatiotemporal_image_v1_seed42",
             "hybrid_vit": "ptbxl_hybrid_vit_image_v1_seed42"}


def twenty_epoch_run(head, seed):
    return V1_SEED42[head] if seed == 42 and head in V1_SEED42 else f"ptbxl_v2_{head}_seed{seed}"


def run_dir(run_id):
    matches = [ROOT / "results" / f / run_id for f in FAMILIES if (ROOT / "results" / f / run_id / "run.json").exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"{run_id}: {len(matches)} completed runs found")
    return matches[0]


def test_predictions(run_id):
    with (run_dir(run_id) / "test_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["ecg_id"] for r in rows]
    y = np.asarray([[int(r[f"truth_{c}"]) for c in CLASS_NAMES] for r in rows])
    p = np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in rows])
    return ids, [r["patient_id"] for r in rows], y, p


def collect(run_ids):
    """{key: probabilities} on the PTB-XL test fold, with shared truth and patient clusters."""
    reference, out = None, {}
    for key, run_id in run_ids.items():
        ids, patients, y, p = test_predictions(run_id)
        if reference is None:
            reference = (ids, patients, y)
        elif ids != reference[0] or not np.array_equal(y, reference[2]):
            raise ValueError(f"{run_id}: test order or truth differs")
        out[key] = p
    groups = {}
    for i, patient in enumerate(reference[1]):
        groups.setdefault(patient, []).append(i)
    return reference[2], out, [np.asarray(v) for v in groups.values()]


def macro(y, p):
    return ext.macro_auroc(y, p, CLASS_NAMES)


def hyp(y, p):
    return ext.auroc(y[:, 4], p[:, 4])


def with_holm(rows):
    for row, adjusted in zip(rows, ext.holm([r["p_bootstrap"] for r in rows])):
        row["p_holm"] = adjusted
        row["significant_at_0.05"] = adjusted < 0.05
    return rows


def fmt(r):
    return (f"{r['difference']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] | {r['p_holm']:.4f}")


# ---------------------------------------------------------------- part A: 40 epochs

def run_summary(run_id):
    run = read_json(run_dir(run_id) / "run.json")
    training = run["training"]
    return {"run": run_id, "selected_epoch": training["best_epoch"], "epochs_run": training["epochs_run"],
            "max_epochs": training["config"]["max_epochs"], "stopped_early": training["stopped_early"],
            "validation_macro_auroc": run["metrics"]["validation"]["macro_roc_auc"],
            "test_macro_auroc": run["metrics"]["test"]["macro_roc_auc"], "device": training["device"]}


def analyze_long40(lines):
    out = OUT / "long40"
    runs = []
    for head in HEADS:
        for seed in SEEDS:
            for budget, run_id in ((40, f"ptbxl_v2_long40_{head}_seed{seed}"), (20, twenty_epoch_run(head, seed))):
                if any((ROOT / "results" / f / run_id / "run.json").exists() for f in FAMILIES):
                    runs.append({"head": head, "seed": seed, "budget": budget, **run_summary(run_id)})
    write_csv(out / "per_run.csv", runs)
    missing = [f"{h}:{s}" for h in HEADS for s in SEEDS
               if not any(r["head"] == h and r["seed"] == s and r["budget"] == 40 for r in runs)]
    lines += ["## A. 40-epoch sensitivity (seeds 42-44, adaptive gain)", ""]
    if missing:
        lines += [f"Incomplete: 40-epoch runs missing for {', '.join(missing)}. Analysis postponed.", ""]
        return
    summary = []
    for head in HEADS:
        for budget in (20, 40):
            rows = [r for r in runs if r["head"] == head and r["budget"] == budget]
            gpu = [r for r in rows if r["device"].startswith("cuda")]
            summary.append({"head": head, "budget": budget,
                            "test_mean": np.mean([r["test_macro_auroc"] for r in rows]),
                            "test_sd": np.std([r["test_macro_auroc"] for r in rows], ddof=1),
                            "validation_mean": np.mean([r["validation_macro_auroc"] for r in rows]),
                            "test_mean_gpu_only": np.mean([r["test_macro_auroc"] for r in gpu]) if gpu else float("nan"),
                            "gpu_seeds": len(gpu),
                            "selected_epochs": " ".join(str(r["selected_epoch"]) for r in rows),
                            "epochs_run": " ".join(str(r["epochs_run"]) for r in rows),
                            "selected_in_last_3_epochs": sum(r["selected_epoch"] > r["max_epochs"] - 3 for r in rows)})
    write_csv(out / "summary.csv", summary)
    lines += ["| Head | Epochs | Test macro AUROC (mean ± SD) | Validation | GPU-only test (n) | Selected epochs | Epochs run |",
              "|---|---:|---:|---:|---:|---|---|"]
    lines += [f"| {s['head']} | {s['budget']} | {s['test_mean']:.4f} ± {s['test_sd']:.4f} | {s['validation_mean']:.4f} | "
              f"{s['test_mean_gpu_only']:.4f} ({s['gpu_seeds']}) | {s['selected_epochs']} | {s['epochs_run']} |" for s in summary]
    keys = {(h, b, s): (f"ptbxl_v2_long40_{h}_seed{s}" if b == 40 else twenty_epoch_run(h, s))
            for h in HEADS for b in (20, 40) for s in SEEDS}
    y, probs, clusters = collect(keys)
    table = lambda head, budget: {s: probs[(head, budget, s)] for s in SEEDS}
    budget_rows = with_holm([{"family": "40 - 20 epochs", "comparison": head, "metric": "test macro AUROC",
                              **ext.hierarchical_difference(y, table(head, 20), table(head, 40), macro, list(SEEDS),
                                                            clusters=clusters)} for head in HEADS])
    pair_rows = with_holm([{"family": "head pairs, 40 epochs", "comparison": f"{b} - {a}", "metric": "test macro AUROC",
                            **ext.hierarchical_difference(y, table(a, 40), table(b, 40), macro, list(SEEDS), clusters=clusters)}
                           for a, b in itertools.combinations(HEADS, 2)])
    pair20 = with_holm([{"family": "head pairs, 20 epochs (same seeds)", "comparison": f"{b} - {a}",
                         "metric": "test macro AUROC",
                         **ext.hierarchical_difference(y, table(a, 20), table(b, 20), macro, list(SEEDS), clusters=clusters)}
                        for a, b in itertools.combinations(HEADS, 2)])
    write_csv(out / "hierarchical_test.csv", budget_rows + pair_rows + pair20)
    lines += ["", "Hierarchical seed x patient bootstrap on the PTB-XL test fold (2,000 resamples), Holm within family.", "",
              "| Family | Comparison | Difference [95% CI] | Holm p |", "|---|---|---|---:|"]
    lines += [f"| {r['family']} | {r['comparison']} | {fmt(r)} |" for r in budget_rows + pair_rows + pair20]
    lines.append("")


# ---------------------------------------------------------------- fixed gain at 40 epochs (evening amendment)

def analyze_fixedgain_long40(lines):
    """docs/protocol_amendment_2026-11-11_fixedgain_long40.md: fixed - adaptive at 40 epochs, and 40 - 20 for fixed gain."""
    out = OUT / "fixedgain_long40"
    keys = {}
    for head in FG_HEADS:
        for seed in SEEDS:
            keys[("fixed", 40, head, seed)] = f"ptbxl_v2_long40_fg_{head}_seed{seed}"
            keys[("adaptive", 40, head, seed)] = f"ptbxl_v2_long40_{head}_seed{seed}"
            keys[("fixed", 20, head, seed)] = f"ptbxl_v2_fg_{head}_seed{seed}"
            keys[("adaptive", 20, head, seed)] = twenty_epoch_run(head, seed)
    lines += ["## C. Fixed-gain training with 40 epochs (seeds 42-44)", ""]
    missing = [run for run in keys.values() if not any((ROOT / "results" / f / run / "run.json").exists() for f in FAMILIES)]
    if missing:
        lines += [f"Incomplete: {len(missing)} runs missing ({', '.join(missing[:6])}...). Analysis postponed.", ""]
        return
    runs = [{"pages": pages, "budget": budget, "head": head, "seed": seed, **run_summary(run)}
            for (pages, budget, head, seed), run in keys.items() if pages == "fixed" and budget == 40]
    write_csv(out / "per_run.csv", runs)
    y, probs, clusters = collect(keys)
    table = lambda pages, budget, head: {s: probs[(pages, budget, head, s)] for s in SEEDS}
    primary = with_holm([{"family": "fixed - adaptive, 40 epochs", "comparison": head, "metric": name,
                          **ext.hierarchical_difference(y, table("adaptive", 40, head), table("fixed", 40, head), metric,
                                                        list(SEEDS), clusters=clusters)}
                         for head in FG_HEADS for name, metric in (("macro AUROC", macro), ("HYP AUROC", hyp))])
    secondary = with_holm([{"family": "fixed gain, 40 - 20 epochs", "comparison": head, "metric": "macro AUROC",
                            **ext.hierarchical_difference(y, table("fixed", 20, head), table("fixed", 40, head), macro,
                                                          list(SEEDS), clusters=clusters)} for head in FG_HEADS])
    descriptive = with_holm([{"family": "fixed - adaptive, 20 epochs (same seeds)", "comparison": head, "metric": name,
                              **ext.hierarchical_difference(y, table("adaptive", 20, head), table("fixed", 20, head), metric,
                                                            list(SEEDS), clusters=clusters)}
                             for head in FG_HEADS for name, metric in (("macro AUROC", macro), ("HYP AUROC", hyp))])
    write_csv(out / "hierarchical_test.csv", primary + secondary + descriptive)
    lines += ["| Head | Seed | Selected epoch | Epochs run | Test macro AUROC |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['head']} | {r['seed']} | {r['selected_epoch']} | {r['epochs_run']} | {r['test_macro_auroc']:.4f} |" for r in runs]
    lines += ["", "Hierarchical seed x patient bootstrap on the PTB-XL test fold (2,000 resamples), Holm within family.", "",
              "| Family | Head | Metric | Difference [95% CI] | Holm p |", "|---|---|---|---|---:|"]
    lines += [f"| {r['family']} | {r['comparison']} | {r['metric']} | {fmt(r)} |" for r in primary + secondary + descriptive]
    lines.append("")


# ---------------------------------------------------------------- part B: ResNet-18

def resnet_runs():
    runs = []
    for page, pattern in RESNET.items():
        for seed in SEEDS:
            run_id = pattern.format(seed)
            directory = run_dir(run_id)
            runs.append((run_id, directory.parent.name, "network", page, "resnet18", seed))
    return runs


def load_resnets(runs, device):
    import torch
    from src.models.ptbxl_resnet18 import LeadImageResNet18
    models = {}
    for run_id, family, *_ in runs:
        checkpoint = torch.load(ROOT / "models" / family / run_id / "best.pt", map_location="cpu", weights_only=True)
        model = LeadImageResNet18(pretrained=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[run_id] = (model.to(device).eval(), checkpoint["normalization"])
    return models


def external_infer(dataset, device="cuda", tolerance=1e-3):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    ext.check_freeze()
    cache = ext.cache_dir(dataset)
    metadata = read_json(cache / "metadata.json")
    if sha256_file(cache / "manifest.csv") != metadata["manifest_sha256"]:
        raise ValueError("External cache changed since it was built")
    out = OUT / "external" / dataset
    runs = resnet_runs()
    done = [r[0] for r in runs if (out / r[0] / "inference_state.json").exists()]
    if done:
        raise FileExistsError(f"External inference already started for {done}; never repeated")
    models = load_resnets(runs, torch.device(device))
    check = ext.reproduction_check(runs, models, torch.device(device))
    write_json(out / "reproduction_check.json", {"checked_utc": utc_now(), "tolerance": tolerance, "amendment": AMENDMENT,
                                                  "max_abs_difference": check})
    failed = {k: v for k, v in check.items() if v > tolerance}
    if failed:
        raise ValueError(f"Reloaded models do not reproduce PTB-XL validation probabilities: {failed}")
    with (cache / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for run_id, *_ in runs:
        (out / run_id).mkdir(parents=True)
        write_json(out / run_id / "inference_state.json", {"status": "started", "started_utc": utc_now(),
                                                          "dataset": dataset, "inference_calls": 1, "amendment": AMENDMENT})
    parts = {run_id: [] for run_id, *_ in runs}
    for start in range(0, len(rows), 64):
        batch = rows[start:start + 64]
        names = [r["record"].replace("/", "__") for r in batch]
        stacks = {page: np.stack([ext._decode(cache / page / f"{n}.png", r[f"{page}_pixel_sha256"])
                                  for n, r in zip(names, batch)]) for page in ("adaptive", "fixed")}
        for run_id, p in ext.predict(models, runs, stacks, None, torch.device(device)).items():
            parts[run_id].append(p)
        if start % 1280 == 0:
            print(f"{utc_now()} {dataset} inference {start}/{len(rows)}", flush=True)
    for run_id, family, kind, page, head, seed in runs:
        p = np.concatenate(parts[run_id])
        write_csv(out / run_id / "predictions.csv", [{"record": r["record"], **{f"truth_{c}": r[f"label_{c}"] for c in CLASS_NAMES},
                  **{f"probability_{c}": float(p[i, j]) for j, c in enumerate(CLASS_NAMES)}} for i, r in enumerate(rows)])
        write_json(out / run_id / "inference_state.json", {"status": "completed", "completed_utc": utc_now(), "dataset": dataset,
                   "inference_calls": 1, "records": len(rows), "family": family, "kind": kind, "page": page, "head": head,
                   "seed": seed, "amendment": AMENDMENT, "predictions_sha256": sha256_file(out / run_id / "predictions.csv")})
    print(f"{utc_now()} {dataset}: ResNet-18 external inference complete ({len(runs)} runs x {len(rows)} records)")


def external_predictions(dataset):
    out, truth, probs = OUT / "external" / dataset, None, {}
    for run_id, _, _, page, _, seed in resnet_runs():
        state = read_json(out / run_id / "inference_state.json")
        if state["status"] != "completed":
            raise ValueError(f"{run_id}: external inference incomplete")
        with (out / run_id / "predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        y = np.asarray([[int(r[f"truth_{c}"]) for c in CLASS_NAMES] for r in rows])
        if truth is not None and not np.array_equal(truth, y):
            raise ValueError("External truth differs between runs")
        truth = y
        probs[(page, seed)] = np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in rows])
    return truth, probs


def analyze_resnet(lines):
    out = OUT / "resnet18"
    keys = {(page, seed): pattern.format(seed) for page, pattern in RESNET.items() for seed in SEEDS}
    y, probs, clusters = collect(keys)
    per_run = []
    for (page, seed), run_id in keys.items():
        summary = run_summary(run_id)
        p = probs[(page, seed)]
        per_run.append({"cohort": "PTB-XL test", "page": page, "seed": seed, "macro_auroc": macro(y, p),
                        **{f"auroc_{c}": ext.auroc(y[:, j], p[:, j]) for j, c in enumerate(CLASS_NAMES)},
                        "selected_epoch": summary["selected_epoch"], "epochs_run": summary["epochs_run"]})
    table = lambda page, pr: {s: pr[(page, s)] for s in SEEDS}
    ptbxl = with_holm([{"cohort": "PTB-XL test", "comparison": "fixed - adaptive", "metric": name,
                        **ext.hierarchical_difference(y, table("adaptive", probs), table("fixed", probs), metric,
                                                      list(SEEDS), clusters=clusters)}
                       for name, metric in (("macro AUROC", macro), ("HYP AUROC", hyp))])
    external, available = [], []
    for dataset in ext.DATASETS:
        if not (OUT / "external" / dataset).exists():
            continue
        available.append(dataset)
        labels = read_json(ext.experiment_dir(dataset) / "inventory.json")["eligible_labels"]
        ye, pe = external_predictions(dataset)
        cohort_macro = lambda yy, pp, labels=labels: ext.macro_auroc(yy, pp, labels)
        for (page, seed), p in sorted(pe.items()):
            run_id = keys[(page, seed)]
            same_labels = ext.ptbxl_test_macro(run_dir(run_id).parent.name, run_id, labels)
            per_run.append({"cohort": dataset, "page": page, "seed": seed, "macro_auroc": cohort_macro(ye, p),
                            **{f"auroc_{c}": ext.auroc(ye[:, j], p[:, j]) for j, c in enumerate(CLASS_NAMES)},
                            "ptbxl_test_macro_auroc_same_labels": same_labels,
                            "external_minus_ptbxl": cohort_macro(ye, p) - same_labels})
        external += [{"cohort": dataset, "comparison": "fixed - adaptive", "metric": name,
                      **ext.hierarchical_difference(ye, table("adaptive", pe), table("fixed", pe), metric, list(SEEDS))}
                     for name, metric in (("macro AUROC (eligible labels)", cohort_macro), ("HYP AUROC", hyp))]
    external = with_holm(external) if external else []
    columns = list(dict.fromkeys(k for r in per_run for k in r))
    write_csv(out / "per_run.csv", [{k: r.get(k, "") for k in columns} for r in per_run])
    write_csv(out / "fixed_minus_adaptive.csv", ptbxl + external)
    lines += ["## B. ImageNet-pretrained ResNet-18 (seeds 42-44)", "",
              "| Cohort | Pages | Macro AUROC (mean ± SD) | HYP AUROC (mean) | Selected epochs |", "|---|---|---:|---:|---|"]
    for cohort in ["PTB-XL test", *available]:
        for page in RESNET:
            rows = [r for r in per_run if r["cohort"] == cohort and r["page"] == page]
            lines.append(f"| {cohort} | {page} | {np.mean([r['macro_auroc'] for r in rows]):.4f} ± "
                         f"{np.std([r['macro_auroc'] for r in rows], ddof=1):.4f} | {np.mean([r['auroc_HYP'] for r in rows]):.4f} | "
                         f"{' '.join(str(r.get('selected_epoch', '')) for r in rows)} |")
    lines += ["", "Fixed - adaptive gain. PTB-XL: seed x patient bootstrap, Holm over 2. External: seed x record bootstrap, "
              "Holm over the cohort x metric comparisons available.", "",
              "| Cohort | Metric | Difference [95% CI] | Holm p |", "|---|---|---|---:|"]
    lines += [f"| {r['cohort']} | {r['metric']} | {fmt(r)} |" for r in ptbxl + external]
    lines.append("")


def analyze():
    lines = [f"# Amendment of 11 Nov 2026: results ({utc_now()[:16]} UTC)", "",
             f"Protocol: {AMENDMENT}. PTB-XL fold 10 was observed before: exploratory sensitivity analyses.", ""]
    analyze_resnet(lines)
    analyze_long40(lines)
    analyze_fixedgain_long40(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["external-infer", "analyze"])
    parser.add_argument("--dataset", choices=list(ext.DATASETS))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.step == "external-infer":
        external_infer(arguments.dataset, arguments.device)
    else:
        analyze()
