"""Additional analyses for the supervisor's review of 15 November 2026.

Usage:
    python -X utf8 -m src.review_20261115 clipping      # two clipping definitions, PTB-XL and both external cohorts
    python -X utf8 -m src.review_20261115 wide          # capacity-matched head on fixed-gain pages
    python -X utf8 -m src.review_20261115 descriptive   # PCA + LDA fixed - adaptive; image - signal gaps
    python -X utf8 -m src.review_20261115 composition   # which SNOMED CT codes the excluded/included records carry
    python -X utf8 -m src.review_20261115 mapping       # label-mapping sensitivity variants V0-V4
    python -X utf8 -m src.review_20261115 stats         # old versus revised bootstrap p-values, every comparison

Protocols (all frozen before the corresponding results existed):
    docs/protocol_amendment_2026-11-15_fixedgain_wide.md
    docs/protocol_amendment_2026-11-15_bootstrap_inference.md
    docs/protocol_amendment_2026-11-15_label_mapping_sensitivity.md
Reads saved manifests, run records and predictions only; writes results/review_20261115/. Nothing is trained and
no inference is run here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import itertools
import multiprocessing
import os

import numpy as np

from src import compare_heads_hierarchical as heads
from src import external_test as ext
from src.common import ROOT, read_json, utc_now, write_csv, write_json

OUT = ROOT / "results/review_20261115"
CLASS_NAMES = ext.CLASS_NAMES
SEEDS = (42, 43, 44, 45, 46)
HEADS = ext.HEADS
COHORTS = ("georgia", "china")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def with_holm(rows, key="p_holm"):
    for row, adjusted in zip(rows, ext.holm([r["p_bootstrap"] for r in rows])):
        row[key] = adjusted
        row["at_resolution_floor"] = row["p_bootstrap"] <= ext.P_FLOOR * (1 + 1e-9)
        row["family_size"] = len(rows)
        row["supported_at_0.05"] = adjusted < 0.05
    return rows


def macro_of(labels):
    return lambda y, p: ext.macro_auroc(y, p, labels)


def hyp(y, p):
    return ext.auroc(y[:, 4], p[:, 4])


WORKERS = int(os.environ.get("REVIEW_WORKERS", "4"))  # processes for the bootstraps (results do not depend on it)


def _job(task):
    y, a, b, spec, seeds, clusters = task
    metric = hyp if spec == "HYP" else macro_of(list(spec))
    return ext.hierarchical_difference(y, a, b, metric, seeds, clusters=clusters)


def bootstrap_all(tasks):
    """tasks: [(row fields, y, a, b, metric spec, seeds, clusters)] -> [{**fields, **result}] in the same order.

    Metric spec: "HYP" or a tuple of label names (macro AUROC over them). Each comparison is the unchanged
    ext.hierarchical_difference with its own fixed random seed, so the result is the same with any number of workers."""
    payload = [task[1:] for task in tasks]
    if WORKERS > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_job, payload))
    else:
        results = [_job(p) for p in payload]
    return [{**task[0], **result} for task, result in zip(tasks, results)]


def metric_specs(labels):
    return (("macro AUROC", tuple(labels)), ("HYP AUROC", "HYP"))


# ---------------------------------------------------------------- clipping definitions (review point 4)

def clipping():
    """Two different boundaries were quoted in the paper; count both, for every cohort.

    band  : a page has a sample beyond the +-12-mm band of the original layout at 2.5 mm/mV. This is the criterion
            that selected 2.5 mm/mV (protocol: >= 98% of training pages must fit). Equivalent to the adaptive
            renderer choosing a gain below 2.5 mm/mV for that ECG.
    row   : a page has a sample beyond its 30-mm row (+-15 mm) at 2.5 mm/mV. Only these samples are actually clipped
            by the fixed-gain renderer (clipped_point_count > 0)."""
    rows = []

    def add(cohort, part, items, gain_key, clip_key, hyp_key, true):
        size = len(items)
        band = [r for r in items if float(r[gain_key]) < 2.5]
        row = [r for r in items if int(r[clip_key]) > 0]
        positives = [r for r in items if r[hyp_key] == true]
        if any(float(r[gain_key]) >= 2.5 for r in row):
            raise ValueError("A clipped page fits the band: the two definitions are not nested")
        rows.append({"cohort": cohort, "partition": part, "pages": size,
                     "outside_12mm_band_at_2p5": len(band), "outside_12mm_band_percent": 100 * len(band) / size,
                     "clipped_at_30mm_row": len(row), "clipped_at_30mm_row_percent": 100 * len(row) / size,
                     "hyp_positive_pages": len(positives),
                     "hyp_positive_outside_band": sum(float(r[gain_key]) < 2.5 for r in positives),
                     "hyp_positive_clipped": sum(int(r[clip_key]) > 0 for r in positives),
                     "clipped_samples_median_among_clipped": float(np.median([int(r[clip_key]) for r in row])) if row else 0.0,
                     "clipped_samples_max": max([int(r[clip_key]) for r in row], default=0)})

    ptbxl = read_rows(ROOT / "data/processed/ptbxl_lead_images_fixedgain2p5_v1/cache_manifest.csv")
    for part in ("train", "validation", "test"):
        add("PTB-XL", part, [r for r in ptbxl if r["split"] == part], "v1_gain_mm_per_mv", "clipped_point_count",
            "label_HYP", "True")
    add("PTB-XL", "all", ptbxl, "v1_gain_mm_per_mv", "clipped_point_count", "label_HYP", "True")
    for cohort in COHORTS:
        frozen = read_rows(ext.cache_dir(cohort) / "manifest.csv")
        add(cohort, "included (confirmatory cohort)", frozen, "adaptive_gain_mm_per_mv", "fixed_clipped_points", "label_HYP", "1")
        unmapped = ROOT / "data/external/cache_unmapped" / cohort / "manifest.csv"
        if unmapped.exists():
            extra = read_rows(unmapped)
            add(cohort, "no mapped superclass (sensitivity only)", extra, "adaptive_gain_mm_per_mv", "fixed_clipped_points",
                "label_HYP", "1")
            add(cohort, "all rendered records", frozen + extra, "adaptive_gain_mm_per_mv", "fixed_clipped_points",
                "label_HYP", "1")
    out = OUT / "clipping"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "clipping_definitions.csv", rows)
    write_json(out / "summary.json", {"created_utc": utc_now(), "definitions": clipping.__doc__, "rows": rows})
    for r in rows:
        print(f"{r['cohort']:8s} {r['partition']:40s} n={r['pages']:6d} band {r['outside_12mm_band_at_2p5']:5d} "
              f"({r['outside_12mm_band_percent']:.2f}%) clipped {r['clipped_at_30mm_row']:5d} "
              f"({r['clipped_at_30mm_row_percent']:.2f}%)")


# ---------------------------------------------------------------- capacity-matched head on fixed gain (point 2)

def fixed_path(head, seed):
    return ROOT / "results/ptbxl_v2_fixedgain" / f"ptbxl_v2_fg_{head}_seed{seed}"


def ptbxl_tables():
    """{(head, page): {seed: test probabilities}}, truth and patient clusters on the PTB-XL test fold."""
    tables, reference = {}, None
    for head in HEADS:
        for page in ("adaptive", "fixed"):
            tables[(head, page)] = {}
            for seed in SEEDS:
                path = heads.run_path(head, seed) if page == "adaptive" else fixed_path(head, seed)
                ids, patients, y, p = heads.load(path, "test")
                if reference is None:
                    reference = (ids, patients, y)
                elif ids != reference[0] or not np.array_equal(y, reference[2]):
                    raise ValueError(f"{head} {page} seed {seed}: test records not aligned")
                tables[(head, page)][seed] = p
    groups = {}
    for i, patient in enumerate(reference[1]):
        groups.setdefault(patient, []).append(i)
    return tables, reference[2], [np.asarray(v) for v in groups.values()]


def external_tables(cohort):
    y, by_run = ext.load_predictions(cohort)
    tables = {}
    for _, (state, p) in by_run.items():
        if state["kind"] == "network" and state["head"] in HEADS:
            tables.setdefault((state["head"], state["page"]), {})[int(state["seed"])] = p
    return tables, y


def wide():
    out = OUT / "fixedgain_wide"
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        path = fixed_path("cnn_pool_wide", seed)
        run = read_json(path / "run.json")
        test, val = run["metrics"]["test"], run["metrics"]["validation"]
        runs.append({"run": path.name, "seed": seed, "device": run["training"].get("device"),
                     "started_utc": run["started_utc"], "completed_utc": run["completed_utc"],
                     "selected_epoch": run["training"].get("best_epoch"), "epochs_run": run["training"].get("epochs_run"),
                     "validation_macro_auroc": val["macro_roc_auc"], "test_macro_auroc": test["macro_roc_auc"],
                     "test_macro_ap": test["macro_average_precision"], "test_macro_f1": test["macro_f1"],
                     **{f"test_auroc_{c}": test["per_label"][c]["roc_auc"] for c in CLASS_NAMES},
                     "test_inference_calls": run["test_inference_calls"], "checkpoint_sha256": run["checkpoint_sha256"]})
    write_csv(out / "per_run.csv", runs)
    rows = []
    cohorts = {"PTB-XL test": (*ptbxl_tables(), list(CLASS_NAMES))}
    for cohort in COHORTS:
        tables, y = external_tables(cohort)
        if all(len(tables.get((h, "fixed"), {})) == len(SEEDS) for h in HEADS):
            cohorts[cohort] = (tables, y, None, read_json(ext.experiment_dir(cohort) / "inventory.json")["eligible_labels"])
        else:
            print(f"{cohort}: external predictions of the fixed-gain capacity-matched runs not found; skipped")
    for cohort, (tables, y, clusters, labels) in cohorts.items():
        primary = [({"cohort": cohort, "family": "capacity-matched: fixed - adaptive", "comparison": "cnn_pool_wide",
                     "metric": name}, y, tables[("cnn_pool_wide", "adaptive")], tables[("cnn_pool_wide", "fixed")], spec,
                    list(SEEDS), clusters) for name, spec in metric_specs(labels)]
        all_heads = [({"cohort": cohort, "family": "five heads: fixed - adaptive (sensitivity, family of 10)",
                       "comparison": head, "metric": name}, y, tables[(head, "adaptive")], tables[(head, "fixed")], spec,
                      list(SEEDS), clusters) for head in HEADS for name, spec in metric_specs(labels)]
        pairs = [({"cohort": cohort, "family": "head pairs on fixed-gain pages", "comparison": f"{b} - {a}",
                   "metric": "macro AUROC"}, y, tables[(a, "fixed")], tables[(b, "fixed")], tuple(labels), list(SEEDS),
                  clusters) for a, b in itertools.combinations(HEADS, 2)]
        done = bootstrap_all(primary + all_heads + pairs)
        primary = with_holm(done[:len(primary)])
        all_heads = with_holm(done[len(primary):len(primary) + len(all_heads)])
        pairs = with_holm(done[len(primary) + len(all_heads):])
        rows += primary + all_heads + pairs
        for r in primary + pairs:
            print(f"{cohort} | {r['family']} | {r['comparison']} | {r['metric']}: {r['difference']:+.4f} "
                  f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] Holm p {r['p_holm']:.4f}", flush=True)
    write_csv(out / "hierarchical.csv", [{k: v for k, v in r.items() if k != "seeds"} for r in rows])
    means = []
    for cohort, (tables, y, clusters, labels) in cohorts.items():
        macro = macro_of(labels)
        for (head, page), by_seed in tables.items():
            values = [macro(y, by_seed[s]) for s in SEEDS if s in by_seed]
            if len(values) == len(SEEDS):
                means.append({"cohort": cohort, "head": head, "page": page, "macro_auroc_mean": float(np.mean(values)),
                              "macro_auroc_sd": float(np.std(values, ddof=1)),
                              **{f"auroc_{c}": float(np.mean([ext.auroc(y[:, j], by_seed[s][:, j]) for s in SEEDS]))
                                 for j, c in enumerate(CLASS_NAMES) if c in labels}})
    write_csv(out / "means.csv", means)
    write_json(out / "metadata.json", {"created_utc": utc_now(), "label": "Additional analysis",
               "amendment": "docs/protocol_amendment_2026-11-15_fixedgain_wide.md",
               "config": "experiments/configs/ptbxl_v2_fg_cnn_pool_wide.json", "dataset": "ptbxl_images_fixedgain2p5_v1",
               "seeds": list(SEEDS), "runs": [r["run"] for r in runs], "bootstrap_resamples": ext.RESAMPLES,
               "bootstrap_seed": ext.BOOT_SEED, "cohorts": list(cohorts)})


# ---------------------------------------------------------------- descriptive additions (points 6 and 8)

def all_groups():
    """Per cohort: ({(model, page): {seed: probabilities}}, truth, patient clusters or None, eligible labels) for the
    five heads, ResNet-18, the PCA + LDA fits and the signal reference."""
    from src import amendment_20261111 as amend
    tables, y, clusters = ptbxl_tables()
    extra = {("resnet18", page): {s: ROOT / "results/ptbxl_v2_resnet18" / pattern.format(s) for s in amend.SEEDS}
             for page, pattern in amend.RESNET.items()}
    extra[("resnet1d_wang", "signal")] = {s: ROOT / f"results/ptbxl_signal/ptbxl_v2_signal_resnet1d_seed{s}" for s in SEEDS}
    extra[("pca_lda", "adaptive")] = {42: ROOT / "results/ptbxl_v2/ptbxl_v2_lda_wide_seed42"}
    extra[("pca_lda", "fixed")] = {42: ROOT / "results/ptbxl_v2_fixedgain/ptbxl_v2_fg_lda_seed42"}
    reference = heads.load(heads.run_path("cnn_pool", 43), "test")
    for group, paths in extra.items():
        tables[group] = {}
        for seed, path in paths.items():
            ids, _, truth, p = heads.load(path, "test")
            if ids != reference[0] or not np.array_equal(truth, reference[2]):
                raise ValueError(f"{path.name}: PTB-XL test records not aligned")
            tables[group][seed] = p
    cohorts = {"PTB-XL test": (tables, y, clusters, list(CLASS_NAMES))}
    for cohort in COHORTS:
        truth, by_run = ext.load_predictions(cohort)
        groups = {}
        for run_id, (state, p) in by_run.items():
            if run_id == "ptbxl_v2_lda_seed42":
                continue  # narrow-grid LDA; the paper reports the wide-grid fit
            groups.setdefault((state["head"], state["page"]), {})[int(state["seed"])] = p
        resnet_truth, resnet = amend.external_predictions(cohort)
        if not np.array_equal(resnet_truth, truth):
            raise ValueError(f"{cohort}: ResNet-18 truth differs")
        for (page, seed), p in resnet.items():
            groups.setdefault(("resnet18", page), {})[int(seed)] = p
        cohorts[cohort] = (groups, truth, None, read_json(ext.experiment_dir(cohort) / "inventory.json")["eligible_labels"])
    return cohorts


def descriptive():
    out = OUT / "descriptive"
    out.mkdir(parents=True, exist_ok=True)
    lda_rows, gap_rows = [], []
    for cohort, (tables, y, clusters, labels) in all_groups().items():
        macro = macro_of(labels)
        lda = [({"cohort": cohort, "family": "PCA + LDA: fixed - adaptive", "comparison": "pca_lda", "metric": name},
                y, tables[("pca_lda", "adaptive")], tables[("pca_lda", "fixed")], spec, [42], clusters)
               for name, spec in metric_specs(labels)]
        family = []
        signal = tables[("resnet1d_wang", "signal")]
        for model in (*HEADS, "resnet18"):
            for page in ("adaptive", "fixed"):
                image = tables.get((model, page), {})
                seeds = sorted(set(image) & set(signal))
                if not seeds:
                    continue
                family.append(({"cohort": cohort, "family": "image model - signal reference",
                                "comparison": f"{model} ({page})", "metric": "macro AUROC",
                                "image_mean": float(np.mean([macro(y, image[s]) for s in seeds])),
                                "signal_mean_same_seeds": float(np.mean([macro(y, signal[s]) for s in seeds]))},
                               y, signal, image, tuple(labels), seeds, clusters))
        done = bootstrap_all(lda + family)
        lda, family = with_holm(done[:len(lda)]), with_holm(done[len(lda):])
        lda_rows += lda
        gap_rows += family
        for r in lda + family:
            print(f"{cohort} | {r['comparison']} | {r['metric']}: {r['difference']:+.4f} [{r['ci_lower']:+.4f}, "
                  f"{r['ci_upper']:+.4f}] Holm p {r['p_holm']:.4f}", flush=True)
    write_csv(out / "lda_fixed_minus_adaptive.csv", [{k: v for k, v in r.items() if k != "seeds"} for r in lda_rows])
    write_csv(out / "image_minus_signal.csv", [{k: (len(v) if k == "seeds" else v) for k, v in r.items()} for r in gap_rows])
    write_json(out / "metadata.json", {"created_utc": utc_now(), "label": "Additional analysis",
               "protocol": "docs/protocol_amendment_2026-11-15_descriptive_additions.md",
               "bootstrap_resamples": ext.RESAMPLES, "bootstrap_seed": ext.BOOT_SEED})


# ---------------------------------------------------------------- SNOMED CT composition (point 3)

def code_names():
    names = {}
    for name in ("snomed_code_names.csv", "snomed_code_names_all.csv"):
        path = ROOT / "experiments/external" / name
        if path.exists():
            for r in read_rows(path):
                names.setdefault(r["snomed_ct_code"], r)
    return names


# Codes that the Challenge organisers score as the same diagnosis (Notes column of dx_mapping_scored.csv) but that
# never occur in their PTB-XL encoding, so the data-driven rule could not see them: RBBB = CRBBB, CLBBB = LBBB.
EQUIVALENT = {"59118001": "713427006", "733534002": "164909002"}


def mapping_at(threshold, equivalents=False):
    """{code: set of superclasses} for the share rule at a threshold (NORM never reaches it)."""
    table = {}
    for r in read_rows(ROOT / "experiments/external/label_mapping.csv"):
        assigned = {c for c in CLASS_NAMES if float(r[f"share_{c}"]) >= threshold}
        if assigned:
            table[r["snomed_code"]] = assigned
    if equivalents:
        for code, same_as in EQUIVALENT.items():
            if same_as in table:
                table[code] = set(table[same_as])
    return table


def composition():
    out = OUT / "mapping"
    out.mkdir(parents=True, exist_ok=True)
    names, frozen_map = code_names(), mapping_at(0.9)
    table = {r["snomed_code"]: r for r in read_rows(ROOT / "experiments/external/label_mapping.csv")}
    code_rows, summary = [], {}
    for cohort in COHORTS:
        rows = read_rows(ext.experiment_dir(cohort) / "inclusion.csv")
        groups = {"included": [r for r in rows if r["status"] == "included"],
                  "no_mapped_superclass": [r for r in rows if r["status"] == "no_mapped_superclass"]}
        counts = {g: {} for g in groups}
        for g, items in groups.items():
            for r in items:
                for code in set(r["dx"].split("|")):
                    counts[g][code] = counts[g].get(code, 0) + 1
        for code in sorted(set(counts["included"]) | set(counts["no_mapped_superclass"]),
                           key=lambda c: -(counts["included"].get(c, 0) + counts["no_mapped_superclass"].get(c, 0))):
            info = table.get(code)
            code_rows.append({"cohort": cohort, "snomed_ct_code": code,
                              "clinical_name": (names.get(code) or {}).get("challenge_dx_name", ""),
                              "abbreviation": (names.get(code) or {}).get("challenge_abbreviation", ""),
                              "in_ptbxl_table": info is not None,
                              "assigned_superclasses": "|".join(sorted(frozen_map.get(code, ()))),
                              "largest_share": max((float(info[f"share_{c}"]) for c in CLASS_NAMES[1:]), default="") if info else "",
                              "largest_share_superclass": max(CLASS_NAMES[1:], key=lambda c: float(info[f"share_{c}"])) if info else "",
                              "norm_share": float(info["share_NORM"]) if info else "",
                              "included_records": counts["included"].get(code, 0),
                              "included_percent": 100 * counts["included"].get(code, 0) / len(groups["included"]),
                              "excluded_records": counts["no_mapped_superclass"].get(code, 0),
                              "excluded_percent": 100 * counts["no_mapped_superclass"].get(code, 0) / len(groups["no_mapped_superclass"])})
        excluded = groups["no_mapped_superclass"]
        rhythm = {"426783006", "426177001", "427084000", "427393009"}  # sinus rhythm, bradycardia, tachycardia, arrhythmia
        relaxed = mapping_at(0.8)
        summary[cohort] = {
            "records": len(rows), "included": len(groups["included"]), "excluded_no_mapped_superclass": len(excluded),
            "excluded_percent_of_all": 100 * len(excluded) / len(rows),
            "distinct_codes": len(set(counts["included"]) | set(counts["no_mapped_superclass"])),
            "distinct_codes_in_ptbxl_table": sum(c in table for c in set(counts["included"]) | set(counts["no_mapped_superclass"])),
            "excluded_only_sinus_rhythm_code": sum(set(r["dx"].split("|")) == {"426783006"} for r in excluded),
            "excluded_only_sinus_rate_or_rhythm_codes": sum(set(r["dx"].split("|")) <= rhythm for r in excluded),
            "excluded_with_no_code_in_ptbxl_table": sum(all(c not in table for c in r["dx"].split("|")) for r in excluded),
            "excluded_with_a_near_miss_code_share_0p80_to_0p90": sum(any(c in relaxed for c in r["dx"].split("|")) for r in excluded),
            "codes_per_record_mean": {g: float(np.mean([len(r["dx"].split("|")) for r in items])) for g, items in groups.items()},
            "positives_by_assigned_code": {
                label: {code: sum(code in r["dx"].split("|") for r in groups["included"])
                        for code, assigned in frozen_map.items() if label in assigned
                        and any(code in r["dx"].split("|") for r in groups["included"])} for label in CLASS_NAMES[1:]}}
    write_csv(out / "codes_by_cohort.csv", code_rows)
    write_json(out / "composition_summary.json", {"created_utc": utc_now(), **summary})
    for cohort, s in summary.items():
        print(cohort, {k: v for k, v in s.items() if k != "positives_by_assigned_code"})


# ---------------------------------------------------------------- mapping sensitivity (point 3)

VARIANTS = (("V0", "published: frozen cohort, 90% rule", 0.9, False, False),
            ("V1", "strict: frozen cohort, share >= 0.99", 0.99, False, False),
            ("V2", "relaxed: frozen cohort, share >= 0.80", 0.8, False, False),
            ("V3", "all records: unmapped as negatives, 90% rule", 0.9, True, False),
            ("V4", "all records, share >= 0.80", 0.8, True, False),
            ("V5", "all records, 90% rule plus organiser-declared equivalent codes", 0.9, True, True))


def cohort_predictions(cohort):
    """Records, SNOMED codes and {(head, page): {seed: probabilities}} for frozen and unmapped records together."""
    from src import amendment_20261111 as amend
    from src import external_allrecords as unmapped
    inclusion = {r["record"]: r for r in read_rows(ext.experiment_dir(cohort) / "inclusion.csv")}
    frozen_records, extra_records, tables = None, None, {}
    sources = [(p.parent, read_json(p)) for p in sorted(ext.result_dir(cohort).glob("*/inference_state.json"))]
    sources += [(p.parent, read_json(p)) for p in sorted((amend.OUT / "external" / cohort).glob("*/inference_state.json"))]
    for folder, state in sources:
        if state["status"] != "completed" or folder.name == "ptbxl_v2_lda_seed42":
            continue
        rows = read_rows(folder / "predictions.csv")
        records = [r["record"] for r in rows]
        if frozen_records is None:
            frozen_records = records
        elif records != frozen_records:
            raise ValueError(f"{folder.name}: frozen record order differs")
        p = np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in rows])
        extra_folder = unmapped.result_dir(cohort) / folder.name
        if (extra_folder / "predictions.csv").exists():
            more = read_rows(extra_folder / "predictions.csv")
            names = [r["record"] for r in more]
            if extra_records is None:
                extra_records = names
            elif names != extra_records:
                raise ValueError(f"{folder.name}: unmapped record order differs")
            q = np.asarray([[float(r[f"probability_{c}"]) for c in CLASS_NAMES] for r in more])
        else:
            q = None
        tables.setdefault((state["head"], state["page"]), {})[int(state["seed"])] = (p, q)
    return inclusion, frozen_records, extra_records or [], tables


def labels_for(records, inclusion, mapping):
    y = np.zeros((len(records), len(CLASS_NAMES)), dtype=np.int64)
    for i, record in enumerate(records):
        for code in inclusion[record]["dx"].split("|"):
            for label in mapping.get(code, ()):
                y[i, CLASS_NAMES.index(label)] = 1
    return y


def mapping():
    out = OUT / "mapping"
    out.mkdir(parents=True, exist_ok=True)
    prevalence, performance, hypotheses = [], [], []
    with (ROOT / "results/ptbxl_v2/ptbxl_v2_cnn_pool_seed43/test_predictions.csv").open(newline="") as handle:
        ptbxl_truth = np.asarray([[int(r[f"truth_{c}"]) for c in CLASS_NAMES] for r in csv.DictReader(handle)])
    prevalence.append({"cohort": "PTB-XL test", "variant": "-", "description": "PTB-XL test fold", "records": len(ptbxl_truth),
                       "eligible_labels": "|".join(CLASS_NAMES),
                       **{f"positives_{c}": int(ptbxl_truth[:, j].sum()) for j, c in enumerate(CLASS_NAMES)},
                       **{f"prevalence_{c}": float(ptbxl_truth[:, j].mean()) for j, c in enumerate(CLASS_NAMES)}})
    for cohort in COHORTS:
        inclusion, frozen, extra, tables = cohort_predictions(cohort)
        frozen_eligible = read_json(ext.experiment_dir(cohort) / "inventory.json")["eligible_labels"]
        saved = np.asarray([[int(inclusion[r][f"label_{c}"]) for c in CLASS_NAMES] for r in frozen])
        if not np.array_equal(labels_for(frozen, inclusion, mapping_at(0.9)), saved):
            raise ValueError(f"{cohort}: relabelling with the 90% rule does not reproduce the frozen labels")
        for key, description, threshold, everything, equivalents in VARIANTS:
            if everything and not extra:
                print(f"{cohort} {key}: predictions for unmapped records not available; skipped")
                continue
            records = frozen + (extra if everything else [])
            y = labels_for(records, inclusion, mapping_at(threshold, equivalents))
            keep = np.ones(len(records), dtype=bool) if everything else y.sum(1) > 0
            y = y[keep]
            eligible = [c for j, c in enumerate(CLASS_NAMES)
                        if y[:, j].sum() >= ext.MIN_POSITIVES and len(y) - y[:, j].sum() >= ext.MIN_NEGATIVES]
            common = [c for c in frozen_eligible if c in eligible]
            prevalence.append({"cohort": cohort, "variant": key, "description": description, "records": int(len(y)),
                               "eligible_labels": "|".join(eligible),
                               **{f"positives_{c}": int(y[:, j].sum()) for j, c in enumerate(CLASS_NAMES)},
                               **{f"prevalence_{c}": float(y[:, j].mean()) for j, c in enumerate(CLASS_NAMES)}})
            probabilities = {}
            for group, by_seed in tables.items():
                if everything and any(q is None for _, q in by_seed.values()):
                    continue
                probabilities[group] = {s: (np.concatenate([p, q]) if everything else p)[keep] for s, (p, q) in by_seed.items()}
            macro = macro_of(common)
            for (head, page), by_seed in probabilities.items():
                seeds = sorted(by_seed)
                performance.append({"cohort": cohort, "variant": key, "head": head, "page": page, "seeds": len(seeds),
                                    "macro_labels": "|".join(common),
                                    "macro_auroc": float(np.mean([macro(y, by_seed[s]) for s in seeds])),
                                    "macro_auroc_variant_eligible": float(np.mean([macro_of(eligible)(y, by_seed[s]) for s in seeds])),
                                    **{f"auroc_{c}": (float(np.mean([ext.auroc(y[:, j], by_seed[s][:, j]) for s in seeds]))
                                                      if c in eligible else "") for j, c in enumerate(CLASS_NAMES)}})
            bootstrap = key == "V3"
            tasks = []
            for a, b in itertools.combinations(HEADS, 2):
                ta, tb = probabilities[(a, "adaptive")], probabilities[(b, "adaptive")]
                tasks.append(({"cohort": cohort, "variant": key, "family": "H1", "comparison": f"{b} - {a}",
                               "metric": "macro AUROC"}, y, ta, tb, tuple(common), sorted(set(ta) & set(tb)), None))
            for head in ("cnn_pool", "cnn_lstm", "temporal_conv", "hybrid_vit"):
                ta, tb = probabilities[(head, "adaptive")], probabilities[(head, "fixed")]
                for name, spec in metric_specs(common):
                    tasks.append(({"cohort": cohort, "variant": key, "family": "H2", "comparison": f"{head}: fixed - adaptive",
                                   "metric": name}, y, ta, tb, spec, sorted(set(ta) & set(tb)), None))
            if bootstrap:
                done = [{k: v for k, v in r.items() if k != "seeds"} for r in bootstrap_all(tasks)]
                h1 = with_holm([r for r in done if r["family"] == "H1"])
                h2 = with_holm([r for r in done if r["family"] == "H2"])
            else:
                done = []
                for fields, yy, ta, tb, spec, seeds, _ in tasks:
                    metric = hyp if spec == "HYP" else macro_of(list(spec))
                    done.append({**fields, "difference": float(np.mean([metric(yy, tb[s]) for s in seeds])
                                                               - np.mean([metric(yy, ta[s]) for s in seeds]))})
                h1 = [r for r in done if r["family"] == "H1"]
                h2 = [r for r in done if r["family"] == "H2"]
            hypotheses += h1 + h2
            print(f"{utc_now()} {cohort} {key}: {len(y)} records, eligible {eligible}", flush=True)
    columns = list(dict.fromkeys(k for r in hypotheses for k in r))
    write_csv(out / "prevalence_by_variant.csv", prevalence)
    write_csv(out / "performance_by_variant.csv", performance)
    write_csv(out / "hypotheses_by_variant.csv", [{k: r.get(k, "") for k in columns} for r in hypotheses])
    write_json(out / "metadata.json", {"created_utc": utc_now(), "label": "Additional analysis",
               "amendment": "docs/protocol_amendment_2026-11-15_label_mapping_sensitivity.md",
               "variants": [{"key": k, "description": d, "share_threshold": t, "includes_unmapped_records": e,
                             "organiser_equivalent_codes": q} for k, d, t, e, q in VARIANTS],
               "equivalent_codes": EQUIVALENT, "bootstrap_resamples": ext.RESAMPLES})


# ---------------------------------------------------------------- old versus revised p-values (point 5)

LEGACY = OUT.parent / "stats_legacy_2000resamples_20261115/results"
STAT_FILES = (("comparisons/ptbxl_heads_hierarchical/pairwise.csv", ("partition", "seed_set"), "comparison", None),
              ("fixed_gain/retrained_comparison/hierarchical_test.csv", ("seed_set",), "architecture", "metric"),
              ("external/georgia/hypotheses.csv", ("family",), "comparison", "metric"),
              ("external/china/hypotheses.csv", ("family",), "comparison", "metric"),
              ("amendment_20261111/resnet18/fixed_minus_adaptive.csv", None, "comparison", "metric"),
              ("amendment_20261111/long40/hierarchical_test.csv", ("family",), "comparison", "metric"),
              ("amendment_20261111/fixedgain_long40/hierarchical_test.csv", ("family",), "comparison", "metric"))


def stats():
    """Every comparison: old (2,000 resamples, p may be 0) next to revised (10,000 resamples, add-one p)."""
    out = OUT / "stats_revision"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative, family_keys, comparison_key, metric_key in STAT_FILES:
        new, old = read_rows(ROOT / "results" / relative), read_rows(LEGACY / relative)
        if "resnet18" in relative:  # Holm families: 2 on PTB-XL, 4 on the external cohorts together
            family_of = lambda r: "PTB-XL test" if r["cohort"] == "PTB-XL test" else "external cohorts"
            identity = lambda r: (r["cohort"], r["metric"])
        else:
            family_of = lambda r: " | ".join(r[k] for k in family_keys)
            identity = lambda r: (family_of(r), r[comparison_key], r[metric_key] if metric_key else "")
        legacy = {identity(r): r for r in old}
        families = {}
        for r in new:
            families.setdefault(family_of(r), []).append(r)
        for family, members in families.items():
            holm_new = ext.holm([float(r["p_bootstrap"]) for r in members])
            holm_old = ext.holm([float(legacy[identity(r)]["p_bootstrap"]) for r in members])
            for r, adjusted, adjusted_old in zip(members, holm_new, holm_old):
                o = legacy[identity(r)]
                if abs(float(r["p_legacy_2000"]) - float(o["p_bootstrap"])) > 1e-12 or \
                        abs(float(r["ci_lower_legacy_2000"]) - float(o["ci_lower"])) > 1e-9 or \
                        abs(float(r["difference"]) - float(o["difference"])) > 1e-12:
                    raise ValueError(f"{relative} {identity(r)}: the first 2,000 draws do not reproduce the archived result")
                floor = len(members) * ext.P_FLOOR
                rows.append({"file": relative, "family": family, "family_size": len(members),
                             "comparison": " | ".join(x for x in identity(r)[1:] if x) if "resnet18" not in relative
                             else f"{r['cohort']} | {r['metric']}",
                             "difference": float(r["difference"]),
                             "ci_lower_old": float(o["ci_lower"]), "ci_upper_old": float(o["ci_upper"]),
                             "ci_lower_revised": float(r["ci_lower"]), "ci_upper_revised": float(r["ci_upper"]),
                             "p_raw_old": float(o["p_bootstrap"]), "p_holm_old": adjusted_old,
                             "p_raw_revised": float(r["p_bootstrap"]), "p_holm_revised": adjusted,
                             "replicates_crossing_zero": min(int(r["replicates_le_0"]), int(r["replicates_ge_0"])),
                             "holm_floor_revised": floor, "at_resolution_floor": float(r["p_bootstrap"]) <= ext.P_FLOOR * (1 + 1e-9),
                             "p_centred_raw": float(r["p_centred"]),
                             "old_printed_as_below_0p001": adjusted_old < 0.001,
                             "supported_old": adjusted_old < 0.05, "supported_revised": adjusted < 0.05,
                             "support_changed": (adjusted_old < 0.05) != (adjusted < 0.05)})
    write_csv(out / "old_vs_revised.csv", rows)
    changed = [r for r in rows if r["support_changed"]]
    summary = {"created_utc": utc_now(), "comparisons": len(rows), "resamples_old": ext.LEGACY_RESAMPLES,
               "resamples_revised": ext.RESAMPLES, "raw_p_floor_revised": ext.P_FLOOR,
               "old_adjusted_p_exactly_zero": sum(r["p_holm_old"] == 0 for r in rows),
               "old_printed_as_below_0p001": sum(r["old_printed_as_below_0p001"] for r in rows),
               "revised_at_resolution_floor": sum(r["at_resolution_floor"] for r in rows),
               "support_at_0p05_changed": len(changed),
               "changed": [{k: r[k] for k in ("file", "family", "comparison", "p_holm_old", "p_holm_revised")} for r in changed],
               "max_abs_ci_bound_shift": max(max(abs(r["ci_lower_old"] - r["ci_lower_revised"]),
                                                 abs(r["ci_upper_old"] - r["ci_upper_revised"])) for r in rows),
               "first_2000_draws_reproduce_archived_results": True}
    write_json(out / "summary.json", summary)
    print({k: v for k, v in summary.items() if k != "changed"})
    for r in changed:
        print("CHANGED:", r["file"], "|", r["family"], "|", r["comparison"], f"{r['p_holm_old']:.4f} -> {r['p_holm_revised']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["clipping", "wide", "descriptive", "composition", "mapping", "stats"])
    step = parser.parse_args().step
    {"clipping": clipping, "wide": wide, "descriptive": descriptive, "composition": composition, "mapping": mapping,
     "stats": stats}[step]()
