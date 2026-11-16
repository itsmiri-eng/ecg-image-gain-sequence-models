"""Supplementary Information for the Scientific Reports manuscript, generated from saved results only.

Usage:
    python -X utf8 -m src.make_supplementary

Writes paper/ieee_manuscript/supplementary_scirep.tex, a standalone pdfLaTeX document (article class, no BibTeX
run needed). Every number is read from saved run records and analysis outputs; nothing is transcribed by hand.
The model cards come from src/make_model_cards.py and the protocol amendments are converted from their dated
documents. Run first: src.make_paper_figures, src.analyze_calibration_thresholds, src.external_test analyze
(both cohorts) and src.count_gain_clipping. Table numbers follow the order of appearance (S1, S2, ...); the
mapping to labels is printed at the end so the main text can cite them.
"""
from __future__ import annotations

import csv
import json
import re

import numpy as np

from src.common import ROOT
from src.external_test import P_FLOOR, RESAMPLES, holm
from src.make_model_cards import CARDS, COMMON, get
from src.make_paper_tables import HEADS, LABELS, NAMES, SEEDS, fmt_diff, fmt_p, run_path

OUT = ROOT / "paper/ieee_manuscript/supplementary_scirep.tex"
# Additional analyses of the supervisor's review, 15 Nov 2026 (src/review_20261115.py)
REVIEW = ROOT / "results/review_20261115"
ADDED = "$^\\P$"  # marks rows from the Additional analysis of 15 Nov 2026
ADDED_NOTE = ("$^\\P$Additional analysis (amendment of 15 November 2026, Section~\\ref{sec:s_review}): the "
              "capacity-matched head trained on fixed-gain pages, added after all other results were known. ")
BOOT = f"{RESAMPLES:,}".replace(",", "{,}") + " resamples"
FLOOR_NOTE = ("$p$ printed as ``$\\le$\'\' is at the resolution limit of the bootstrap: no replicate crossed zero, and "
              "no smaller value is claimed (Section~\\ref{sec:s_review}). ")
BIB = ROOT / "paper/ieee_manuscript/references.bib"
TITLE = ("Rendering gain and sequence modeling in 12-lead ECG-image classification: "
         "a controlled comparison with external validation")
REFERENCE_RUNS = (("Adaptive", "PCA + LDA (grid $\\le$256)", "ptbxl_v2/ptbxl_v2_lda_seed42"),
                  ("Adaptive", "PCA + LDA (grid 256--1024)", "ptbxl_v2/ptbxl_v2_lda_wide_seed42"),
                  ("Adaptive", "PCA + linear SVM", "ptbxl_v2/ptbxl_v2_svm_seed42"),
                  ("Fixed", "PCA + LDA (grid 256--1024)", "ptbxl_v2_fixedgain/ptbxl_v2_fg_lda_seed42"))
SIGNAL = "1D ResNet (raw signals)"
UNICODE = {"→": "$\\rightarrow$", "≥": "$\\geq$", "≤": "$\\leq$", "±": "$\\pm$", "×": "$\\times$", "Δ": "$\\Delta$",
           "α": "$\\alpha$", "·": "$\\cdot$", "≈": "$\\approx$", "§": "\\S{}", "–": "--", "—": "---", "’": "'",
           "‘": "`", "“": "``", "”": "''", "µ": "$\\mu$", "°": "$^\\circ$", "−": "$-$",
           "∆": "$\\hat{\\Delta}$", "≡": "$\\equiv$"}  # "∆" (U+2206) stands for a Delta with a hat, see markdown()
SPECIAL = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_", "{": "\\{",
           "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}

# Gain-policy survey (docs/gain_policy_survey_2026-11-11.md lists the exact URL and file/line or section per claim).
SURVEY = (
    ("ECG-Image-Kit \\cite{shivashankara2024ecgimagekit}", "fixed, 10 mm/mV; 1-mV pulse", "no",
     "no per-row clipping in code; traces may overlap adjacent rows", "no (only pixel resolution)"),
    ("PhysioNet Challenge 2024 images \\cite{reyna2024challenge}", "generated with ECG-Image-Kit", "not stated",
     "not stated", "not stated"),
    ("ECG-Image-Database \\cite{reyna2024imagedatabase}", "ECG-Image-Kit, printed; 0.5 mV per 5-mm square",
     "not stated", "not stated", "not stated"),
    ("Sangha et al.\\ 2022 \\cite{sangha2022images}", "ecg-plot 0.2.8: fixed, $\\approx$10.16 mm/mV at nominal size",
     "no", "no per-row clipping in code; traces may overlap adjacent rows", "not stated"),
    ("Khan et al.\\ 2021 (real printouts) \\cite{khan2021ecgimages}", "not stated", "not stated", "not stated",
     "not applicable"),
)


def esc(text):
    return "".join(SPECIAL.get(c) or UNICODE.get(c) or c for c in str(text))


def inline(text):
    """Markdown inline (`code`, **bold**, [text](url)) to LaTeX; everything else escaped."""
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            code = part[1:-1]  # \nolinkurl breaks long paths at "/" but drops spaces, so only for plain paths
            plain = " " not in code and not set(code) & set("%#\\")
            out.append("\\nolinkurl{" + code + "}" if plain else "\\texttt{" + esc(code) + "}")
        elif part.startswith("**") and part.endswith("**") and len(part) > 3:
            out.append("\\textbf{" + inline(part[2:-2]) + "}")
        elif re.fullmatch(r"\[[^\]]+\]\([^)]+\)", part):
            out.append(esc(re.match(r"\[([^\]]+)\]", part).group(1)))
        else:
            out.append(esc(part))
    return "".join(out)


def markdown(text, skip_title=True):
    """The small Markdown subset used in the project's protocol documents (headings, nested lists, paragraphs)."""
    out, stack = [], []

    def close(to_indent=-1):
        while stack and stack[-1][0] > to_indent:
            out.append(f"\\end{{{stack.pop()[1]}}}")

    in_paragraph = False
    table = []

    def flush_table():
        """A Markdown pipe table (header, separator, rows) as a small tabular with wrapped columns."""
        if not table:
            return
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.fullmatch(r"[\s|:-]+", r)]
        width = max(len(r) for r in rows)
        out.append("\\begin{center}\\footnotesize\\begin{tabular}{@{}" + f"p{{{0.92 / width:.2f}\\textwidth}}" * width + "@{}}")
        out.append("\\toprule")
        for k, row in enumerate(rows):
            out.append(" & ".join(inline(c) for c in row + [""] * (width - len(row))) + " \\\\")
            if k == 0:
                out.append("\\midrule")
        out.append("\\bottomrule\\end{tabular}\\end{center}")
        table.clear()

    for line in text.replace("Δ̂", "∆").splitlines():  # Delta + combining circumflex -> one placeholder character
        if line.lstrip().startswith("|"):
            close()
            table.append(line)
            in_paragraph = False
            continue
        flush_table()
        if not line.strip():
            in_paragraph = False
            continue
        heading = re.match(r"^(#+)\s+(.*)", line)
        item = re.match(r"^(\s*)(-|\d+\.)\s+(.*)", line)
        if heading:
            close()
            if len(heading.group(1)) == 1 and skip_title:
                continue
            out.append(f"\\paragraph{{{inline(heading.group(2))}}}")
        elif item:
            indent, kind = len(item.group(1)), "itemize" if item.group(2) == "-" else "enumerate"
            close(indent)
            if stack and stack[-1][0] == indent and stack[-1][1] != kind:
                close(indent - 1)
            if not stack or stack[-1][0] < indent:
                stack.append((indent, kind))
                out.append(f"\\begin{{{kind}}}")
            out.append("\\item " + inline(item.group(3)))
        elif stack and line.startswith(" "):
            out.append(inline(line.strip()))
        else:
            close()
            if not in_paragraph:
                out.append("")
            out.append(inline(line.strip()))
            in_paragraph = True
            continue
        in_paragraph = False
    flush_table()
    close()
    return "\n".join(out)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def n(x):
    return f"{int(x):,}".replace(",", "{,}")


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def clinical_name(row):
    """Challenge diagnosis name with its abbreviation, e.g. 'left ventricular hypertrophy (LVH)'."""
    name, short = row.get("challenge_dx_name", ""), row.get("challenge_abbreviation", "")
    return f"{name} ({short})" if name and short else name or "name not in the Challenge tables"


def p_cell(row, adjusted=None):
    """Holm-adjusted p of a result row, printed as an upper bound at the bootstrap's resolution floor."""
    return fmt_p(float(row["p_holm"] if adjusted is None else adjusted), row["p_bootstrap"])


def longtable(caption, label, spec, header, rows, note="", size="\\footnotesize", colsep="3pt"):
    lines = ["\\begingroup" + size, "\\setlength{\\tabcolsep}{" + colsep + "}", f"\\begin{{longtable}}{{{spec}}}",
             f"\\caption{{{caption}}}\\label{{{label}}}\\\\", "\\toprule", header + " \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", header + " \\\\", "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
    for row in rows:
        lines.append(row if row == "\\midrule" else row + " \\\\")
    lines += ["\\end{longtable}", "\\endgroup"]
    if note:
        lines += ["{\\footnotesize " + note + "\\par}", "\\medskip"]
    return "\n".join(lines)


# ---------------------------------------------------------------- runs

def all_runs():
    """(pages, model name, head key, seed, path) for every PTB-XL run reported in the paper."""
    runs = []
    for fixed in (False, True):
        for head in HEADS:
            name = NAMES[head] + (ADDED if fixed and head == "cnn_pool_wide" else "")
            runs += [("Fixed" if fixed else "Adaptive", name, head, s, run_path(head, s, fixed)) for s in SEEDS]
    runs += [(pages, name, "reference", None, ROOT / "results" / rel) for pages, name, rel in REFERENCE_RUNS]
    runs += [("Signal", SIGNAL, "resnet1d_wang", s, ROOT / f"results/ptbxl_signal/ptbxl_v2_signal_resnet1d_seed{s}")
             for s in SEEDS]
    return runs


def run_info(path):
    run = read_json(path / "run.json")
    training = run.get("training") or {}
    test, val = run["metrics"]["test"], run["metrics"]["validation"]
    return {"device": get(training, "device") or "cpu", "epochs": get(training, "epochs_run"),
            "best": get(training, "best_epoch"), "val": val["macro_roc_auc"], "test": test["macro_roc_auc"],
            "ap": test["macro_average_precision"], "f1": test["macro_f1"],
            "labels": [test["per_label"][l]["roc_auc"] for l in LABELS]}


# ---------------------------------------------------------------- tables

def table_cohort():
    rows = read_csv(ROOT / "data/processed/ptbxl_lead_images_v1/cache_manifest.csv")
    summary = read_json(ROOT / "data/derived/ptbxl_images_v1/generation_summary.json")["source_manifest_summary"]
    body = []
    for split in ("train", "validation", "test", None):
        part = [r for r in rows if split is None or r["split"] == split]
        name = {"train": "Training (folds 1--8)", "validation": "Validation (fold 9)", "test": "Test (fold 10)",
                None: "Total"}[split]
        cells = [name, n(len(part)), n(len({r["patient_id"] for r in part}))]
        cells += [n(sum(r[f"label_{l}"] == "True" for r in part)) for l in LABELS]
        body += (["\\midrule"] if split is None else []) + [" & ".join(cells)]
    note = (f"PTB-XL v1.0.3 contains {n(summary['all_source_records'])} records from {n(summary['all_source_patients'])} "
            f"patients; {n(summary['classification_ineligible_records'])} records without any diagnostic superclass were "
            "excluded, and every included record was rendered and used. Labels overlap, so positives do not sum to "
            "the number of records.")
    return longtable("PTB-XL cohort by official fold: records, patients and positives per superclass.", "tab:s_cohort",
                     "lrrrrrrr", "Partition & Records & Patients & " + " & ".join(LABELS), body, note)


def table_external_flow():
    inv = {d: read_json(ROOT / f"experiments/external/{d}/inventory.json") for d in ("georgia", "china")}
    steps = ["records", "signal_file_failed_checksum", "not_12_standard_leads", "not_500_hz", "not_millivolt",
             "shorter_than_10_s", "no_mapped_superclass", "included"]
    names = {"records": "Records in the Challenge 2021 training set", "signal_file_failed_checksum":
             "Excluded: signal file failed PhysioNet SHA-256 (amendment)", "not_12_standard_leads":
             "Excluded: not 12 standard leads", "not_500_hz": "Excluded: not 500 Hz", "not_millivolt":
             "Excluded: not in millivolts", "shorter_than_10_s": "Excluded: shorter than 10 s",
             "no_mapped_superclass": "Excluded: no mapped superclass", "included": "Included"}
    body = [" & ".join([names[s]] + [n(inv[d]["counts"].get(s, 0)) for d in inv]) for s in steps]
    body.append("\\midrule")
    for label in LABELS:
        cells = [f"Positives, {label}"]
        for d in inv:
            mark = "" if label in inv[d]["eligible_labels"] else "$^*$"
            cells.append(n(inv[d]["positives"][label]) + mark)
        body.append(" & ".join(cells))
    note = ("$^*$Not evaluated in that cohort: a label needs at least 50 positives and 50 negatives (prespecified). "
            "Steps are applied in the order shown; each record is counted at its first failed step. China: "
            "Chapman-Shaoxing and Ningbo databases. Source: experiments/external/$<$cohort$>$/inventory.json.")
    return longtable("External cohorts: inclusion flow and positives per superclass.", "tab:s_flow", "lrr",
                     "Step & Georgia, USA & China", body, note)


def table_mapping():
    meta = read_json(ROOT / "experiments/external/label_mapping.json")
    names = {r["snomed_ct_code"]: r for r in read_csv(ROOT / "experiments/external/snomed_code_names.csv")}
    body = []
    for r in read_csv(ROOT / "experiments/external/label_mapping.csv"):
        shares = [float(r[f"share_{l}"]) for l in LABELS]
        name = names[r["snomed_code"]]
        if not name["challenge_dx_name"]:
            raise ValueError(f"No clinical name for SNOMED CT code {r['snomed_code']}")
        cells = [r["snomed_code"], esc(clinical_name(name)), n(r["ptbxl_records"])]
        cells += [("\\textbf{%.2f}" % s) if s >= 0.9 else f"{s:.2f}" for s in shares]
        cells.append(esc(r["assigned_superclasses"].replace("|", ", ")) or "--")
        body.append(" & ".join(cells))
    checked = sum(r["names_agree"] == "yes" for r in names.values())
    differ = sum(r["names_agree"] == "no" for r in names.values())
    note = (f"Rule fixed before the mapping was computed: {esc(meta['rule'])}. Shares are the fraction of PTB-XL records "
            "carrying the SNOMED CT code (Challenge organizers' encoding of PTB-XL) that have each superclass; bold "
            f"shares meet the 90\\% rule. {meta['codes_assigned']} of {meta['codes_seen']} codes were assigned. "
            f"Headers checked: {n(meta['headers_checked'])}; age/sex mismatches: {meta['age_sex_mismatches']}. "
            "Clinical names (added 15 November 2026): the diagnosis name and abbreviation that the PhysioNet/CinC "
            "Challenge 2021 organizers give for each code (files dx\\_mapping\\_scored.csv and "
            "dx\\_mapping\\_unscored.csv of the repository physionetchallenges/evaluation-2021), lower-cased as in the "
            f"source. The name was compared with the SNOMED CT concept name from an independent terminology server for "
            f"{checked + differ} of the {len(names)} codes ({checked} agree in meaning, {differ} differ; the rest could "
            "not be checked); sources and notes per code are in experiments/external/snomed\\_code\\_names.csv. The "
            "mapping itself is data-driven and was not reviewed by a cardiologist (Section~\\ref{sec:s_review}).")
    return longtable("SNOMED CT code, clinical name and PTB-XL superclass mapping used for the external cohorts.",
                     "tab:s_mapping", "lp{4.6cm}rrrrrrl", "SNOMED CT & Clinical name (abbreviation) & PTB-XL records & "
                     + " & ".join(LABELS) + " & Assigned", body, note, size="\\scriptsize")


def table_per_run(runs):
    body, body_labels, previous = [], [], None
    for pages, name, head, seed, path in runs:
        info = run_info(path)
        if previous is not None and pages != previous:
            body.append("\\midrule")
            body_labels.append("\\midrule")
        epochs = "--" if info["epochs"] is None else f"{info['best']} of {info['epochs']}"
        seed_text = "--" if seed is None else str(seed)
        body.append(" & ".join([pages, name, seed_text, info["device"].upper() if info["device"] == "cpu" else "GPU",
                                epochs, f3(info["val"]), f3(info["test"]), f3(info["ap"]), f3(info["f1"])]))
        body_labels.append(" & ".join([pages, name, seed_text] + [f3(v) for v in info["labels"]]))
        previous = pages
    per_run = longtable("Every PTB-XL run: hardware, selected epoch, validation and test (fold 10, exploratory) results.",
                        "tab:s_runs", "lllllrrrr", "Pages & Model & Seed & Device & Selected epoch & Val. AUROC & "
                        "Test AUROC & Test AP & Test F1", body,
                        "Macro averages over the five labels; F1 at the fixed threshold 0.5. Selected epoch: epoch of the "
                        "checkpoint chosen on validation, of the epochs run (maximum 20; fewer means early stopping). "
                        "CPU: laptop CPU (Mac, seed 42 of the four original adaptive-gain networks); GPU: NVIDIA RTX "
                        "5070 Ti. PCA references are single fits. Seed 42 of CNN-LSTM, temporal convolution and ViT with "
                        "adaptive gain are the original v1 runs. " + ADDED_NOTE, size="\\scriptsize")
    labels = longtable("Every PTB-XL run: test AUROC per superclass.", "tab:s_runs_labels", "lllrrrrr",
                       "Pages & Model & Seed & " + " & ".join(LABELS), body_labels, ADDED_NOTE, size="\\scriptsize")
    return per_run, labels


def table_epochs(runs):
    groups = {}
    for pages, name, head, seed, path in runs:
        if head == "reference":
            continue
        info = run_info(path)
        groups.setdefault((pages, name), []).append(info)
    body, totals = [], {"Adaptive": [0, 0], "Fixed": [0, 0]}
    for (pages, name), infos in groups.items():
        late = sum(i["best"] >= 18 for i in infos)
        early = sum(i["epochs"] < 20 for i in infos)
        if pages in totals:
            totals[pages][0] += late
            totals[pages][1] += len(infos)
        body.append(" & ".join([pages, name, ", ".join(str(i["best"]) for i in infos), f"{late} of {len(infos)}",
                                f"{early} of {len(infos)}"]))
    note = (f"Seeds 42--46 in order. Image networks selecting epoch 18--20 of 20: {totals['Adaptive'][0]} of "
            f"{totals['Adaptive'][1]} adaptive-gain runs and {totals['Fixed'][0]} of {totals['Fixed'][1]} fixed-gain runs. "
            "The epoch budget was prespecified and not tuned; a cosine schedule that ends at epoch 20 favours late "
            "selection, but late selection may also mean that some models were still improving. " + ADDED_NOTE +
            "The fixed-gain count includes its five runs.")
    return longtable("Selected (validation-best) epoch per run, by model.", "tab:s_epochs", "llccc",
                     "Pages & Model & Selected epoch per seed & Selected $\\ge$18 & Stopped early", body, note), totals


def table_gpu_pairs():
    rows = read_csv(ROOT / "results/comparisons/ptbxl_heads_hierarchical/pairwise.csv")
    pick = lambda part, seeds: {r["comparison"]: r for r in rows if r["partition"] == part and r["seed_set"] == seeds}
    body = []
    for part, title in (("test", "Test fold (exploratory)"), ("validation", "Validation fold")):
        primary, gpu = pick(part, "42-46 (primary)"), pick(part, "43-46 GPU only")
        if body:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}")
        for comparison, r in primary.items():
            cells = [esc(comparison.replace(" minus ", " $-$ ")).replace("\\$", "$")]
            for row in (r, gpu[comparison]):
                cells += [fmt_diff(float(row["difference"]), float(row["ci_lower"]), float(row["ci_upper"])), p_cell(row)]
            body.append(" & ".join(cells))
    return longtable("PTB-XL head comparisons (adaptive gain) with all seeds and with GPU-trained seeds only: difference "
                     "in mean-over-seeds macro AUROC [95\\% CI] and Holm-adjusted $p$.", "tab:s_gpu_pairs",
                     "lcccc", "Comparison & \\multicolumn{2}{c}{Seeds 42--46} & \\multicolumn{2}{c}{Seeds 43--46, GPU only}",
                     body, "Hierarchical bootstrap (seeds, then patients; " + BOOT + "), Holm over the 10 pairs within "
                     "each fold and seed set. " + FLOOR_NOTE +
                     "Seeds 43--46 exclude the CPU-trained seed 42 of the original networks.",
                     size="\\scriptsize")


def table_gpu_fixed():
    rows = read_csv(ROOT / "results/fixed_gain/retrained_comparison/hierarchical_test.csv")
    by_set = {}
    for seeds in ("42-46", "43-46 GPU only"):
        part = [r for r in rows if r["seed_set"] == seeds]
        for r, adjusted in zip(part, holm([float(r["p_bootstrap"]) for r in part])):
            by_set[(seeds, r["architecture"], r["metric"])] = (r, adjusted)
    body = []
    for head in ("cnn_pool", "temporal_conv", "cnn_lstm", "hybrid_vit"):
        for metric in ("macro AUROC", "HYP AUROC"):
            cells = [NAMES[head] if metric == "macro AUROC" else "", metric.replace(" AUROC", "")]
            for seeds in ("42-46", "43-46 GPU only"):
                r, adjusted = by_set[(seeds, head, metric)]
                cells += [fmt_diff(float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"])),
                          fmt_p(adjusted, r["p_bootstrap"])]
            body.append(" & ".join(cells))
    return longtable("PTB-XL test: fixed-gain minus adaptive-gain training with all seeds and with GPU-trained seeds "
                     "only [95\\% CI], Holm-adjusted $p$.", "tab:s_gpu_fixed", "llcccc",
                     "Architecture & AUROC & \\multicolumn{2}{c}{Seeds 42--46} & \\multicolumn{2}{c}{Seeds 43--46, GPU only}",
                     body, "Hierarchical bootstrap (seeds, then patients; " + BOOT + "); Holm over the 8 comparisons within each "
                     "seed set. " + FLOOR_NOTE +
                     "With seeds 42--46, the seed-42 pair combines a CPU-trained adaptive model with a GPU-trained "
                     "fixed-gain model; the GPU-only set removes that hardware difference.")


def group_of(runs):
    return {path.name: (pages, name) for pages, name, head, seed, path in runs}


def table_calibration(runs):
    groups, order = {}, []
    lookup = group_of(runs)
    for r in read_csv(ROOT / "results/secondary/calibration_thresholds/summary.csv"):
        key = lookup.get(r["run_id"])
        if key is None:
            continue  # runs not reported in the paper's tables
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(r)
    order.sort(key=lambda k: [x[:2] for x in runs].index(k))
    cols = ("validation_macro_ece", "test_macro_ece", "test_macro_brier", "test_macro_f1_at_0_5",
            "test_macro_f1_at_validation_thresholds")
    body, previous = [], None
    for key in order:
        rows = groups[key]
        if previous is not None and key[0] != previous:
            body.append("\\midrule")
        cells = [key[0], key[1], str(len(rows))]
        for c in cols:
            v = np.asarray([float(r[c]) for r in rows])
            cells.append(f"{v.mean():.3f}" + (f" $\\pm$ {v.std(ddof=1):.3f}" if len(v) > 1 else ""))
        body.append(" & ".join(cells))
        previous = key[0]
    return longtable("Calibration and validation-chosen thresholds (mean $\\pm$ SD over runs).", "tab:s_calibration",
                     "lllccccc", "Pages & Model & Runs & Val. ECE & Test ECE & Test Brier & Test F1 (0.5) & "
                     "Test F1 (val.\\ thr.)", body,
                     "Macro averages over the five labels. ECE: expected calibration error with 10 equal-width bins. "
                     "Validation thresholds: per label, the threshold in 0.01--0.99 that maximizes validation F1, "
                     "applied unchanged to test. The fixed 0.5 rule remains primary.", size="\\scriptsize")


def table_thresholds(runs):
    lookup = group_of(runs)
    groups = {}
    for r in read_csv(ROOT / "results/secondary/calibration_thresholds/thresholds_per_label.csv"):
        key = lookup.get(r["run_id"])
        if key is not None:
            groups.setdefault((key, r["label"]), []).append(r)
    cols = ("validation_threshold", "test_sensitivity_at_0.5", "test_specificity_at_0.5",
            "test_sensitivity_at_validation_threshold", "test_specificity_at_validation_threshold", "test_f1_at_0.5",
            "test_f1_at_validation_threshold")
    body, previous = [], None
    for key in dict.fromkeys(x[:2] for x in runs):
        if (key, LABELS[0]) not in groups:
            continue
        if previous is not None:
            body.append("\\midrule")
        for label in LABELS:
            rows = groups[(key, label)]
            cells = [key[0] if label == LABELS[0] else "", key[1] if label == LABELS[0] else "", label]
            cells += [f"{np.mean([float(r[c]) for r in rows]):.2f}" for c in cols]
            body.append(" & ".join(cells))
        previous = key
    return longtable("Validation-chosen thresholds per label and their effect on test (mean over runs).",
                     "tab:s_thresholds", "lllrrrrrrr", "Pages & Model & Label & Thr. & Sens.\\ 0.5 & Spec.\\ 0.5 & "
                     "Sens.\\ thr. & Spec.\\ thr. & F1 0.5 & F1 thr.", body,
                     "Thr.: mean validation-chosen threshold. Sens./Spec./F1 on the test fold at 0.5 and at the "
                     "validation-chosen threshold.", size="\\scriptsize")


def table_external_labels():
    eligible = {d: read_json(ROOT / f"experiments/external/{d}/inventory.json")["eligible_labels"]
                for d in ("georgia", "china")}
    data = {}
    for d in ("georgia", "china"):
        for r in read_csv(ROOT / f"results/external/{d}/per_run.csv"):
            if r["run"] == "ptbxl_v2_lda_seed42":
                continue  # narrow-grid LDA; the paper reports the wide-grid fit
            data.setdefault((r["head"], r["page"]), {}).setdefault(d, []).append(r)
    order = [(h, "adaptive") for h in HEADS] + [(h, "fixed") for h in HEADS]
    order += [("pca_lda", "adaptive"), ("pca_svm", "adaptive"), ("pca_lda", "fixed"), ("resnet1d_wang", "signal")]
    names = {**NAMES, "pca_lda": "PCA + LDA", "pca_svm": "PCA + linear SVM", "resnet1d_wang": SIGNAL}
    body, previous = [], None
    for head, page in order:
        if previous is not None and page != previous:
            body.append("\\midrule")
        added = ADDED if (head, page) == ("cnn_pool_wide", "fixed") else ""
        cells = [page.capitalize(), names[head] + added, str(len(data[(head, page)]["georgia"]))]
        for d in ("georgia", "china"):
            rows = data[(head, page)][d]
            cells += [f"{np.mean([float(r[f'auroc_{l}']) for r in rows]):.3f}" for l in eligible[d]]
            cells.append(f"{np.mean([float(r['external_macro_auroc']) for r in rows]):.3f}")
        body.append(" & ".join(cells))
        previous = page
    header = ("Pages & Model & Runs & " + " & ".join(eligible["georgia"]) + " & Macro & "
              + " & ".join(eligible["china"]) + " & Macro")
    spec = "lll" + "r" * (len(eligible["georgia"]) + len(eligible["china"]) + 2)
    return longtable("External AUROC per eligible superclass (mean over seeds). Left: Georgia, USA; right: China.",
                     "tab:s_external_labels", spec, header, body,
                     "Each run performed exactly one inference per cohort. NORM is not evaluable externally; MI is not "
                     "evaluable in Georgia (7 positives). Signal seeds 45 and 46 were added under the 11 November 2026 "
                     "amendment (Section~\\ref{sec:s_amendments}) as a descriptive addition. " + ADDED_NOTE, size="\\scriptsize")


def table_survey():
    body = [" & ".join(row) for row in SURVEY]
    return longtable("How public ECG-image generators and datasets set the page gain (checked 11 November 2026).",
                     "tab:s_survey", "p{3.6cm}p{3.6cm}p{1.6cm}p{3.6cm}p{2.2cm}",
                     "Resource & Gain when drawing & Amplitude-dependent? & Large deflections & Gain randomized?", body,
                     "Primary sources only (papers, code at a fixed commit, documentation); ``not stated'' means not "
                     "found in those sources. ECG-Image-Kit: repository alphanumericslab/ecg-image-kit, commit 27b90f5, "
                     "codes/ecg-image-generator/ecg\\_plot.py (0.5 mV per 5-mm grid, page label 10 mm/mV). ecg-plot "
                     "0.2.8: the release current when Sangha et al.\\ accessed it (repository dy1901/ecg\\_plot, commit "
                     "3855e94). The project file docs/gain\\_policy\\_survey\\_2026-11-11.md gives the URL and line or "
                     "section for every entry.", size="\\scriptsize")


def table_clipping():
    summary = read_json(ROOT / "results/fixed_gain/clipping_by_gain/summary.json")
    body = []
    for gain, row in summary["by_gain"].items():
        body.append(" & ".join([gain, n(row["pages_with_any_clipping"]), f"{row['percent_of_pages']:.1f}",
                                *[n(row["by_split"][s]) for s in ("train", "validation", "test")],
                                n(row["hyp_positive_pages_with_clipping"])]))
    return longtable("PTB-XL pages with at least one clipped sample if every page were drawn at one fixed gain "
                     "(clipping at the edge of the 30-mm lead row).", "tab:s_clipping",
                     "rrrrrrr", "Gain (mm/mV) & Pages & \\% of " + n(summary["ecgs"]) +
                     " & Training & Validation & Test & HYP-positive", body,
                     "Definition used in this table: a page is counted when at least one sample of at least one lead "
                     "falls outside that lead's 30-mm row ($\\pm$15 mm around the row centre); the fixed-gain renderer "
                     "clips such samples to the row edge. Numerator: pages with $\\geq$1 clipped sample; denominator: all "
                     "pages of the column (21{,}388 in total; 17{,}084 training, 2{,}146 validation, 2{,}158 test; 2{,}649 "
                     "HYP-positive). Stage: page rendering, before any model sees the page; no page was excluded. The 2.5 "
                     "mm/mV counts reproduce the clipping recorded in the fixed-gain dataset for every ECG. The 121 "
                     "training pages of the first row are a subset of the 274 training pages that exceed the narrower "
                     "$\\pm$12-mm band used to choose the gain; Supplementary Table~\\ref{tab:s_clipping_defs} gives both "
                     "counts side by side, for PTB-XL and for both external cohorts. Computed by "
                     "src/count\\_gain\\_clipping.py with the unchanged renderer.")


# ---------------------------------------------------------------- amendment of 11 Nov 2026 (results/amendment_20261111)

AMENDMENT = ROOT / "results/amendment_20261111"
COHORTS = {"PTB-XL test": "PTB-XL test", "georgia": "Georgia, USA", "china": "China"}
HEAD_NAMES = {**NAMES, "cnn_pool_wide": "CNN-only, capacity-m."}


def table_resnet():
    rows = read_csv(AMENDMENT / "resnet18/per_run.csv")
    body = []
    for cohort, name in COHORTS.items():
        for r in (r for r in rows if r["cohort"] == cohort):
            delta = r.get("external_minus_ptbxl") or ""
            body.append(" & ".join([name, r["page"].capitalize(), r["seed"], f3(float(r["macro_auroc"])),
                                    *[f3(float(r[f"auroc_{l}"])) for l in LABELS],
                                    f"{float(delta):+.3f}".replace("-", "$-$") if delta else "--",
                                    r.get("selected_epoch") or "--"]))
        body.append("\\midrule")
    first = longtable("ImageNet-pretrained ResNet-18 (sensitivity analysis, amendment of 11 November 2026): every run.",
                      "tab:s_resnet_runs", "lllrrrrrrrr",
                      "Cohort & Pages & Seed & Macro & NORM & MI & STTC & CD & HYP & $\\Delta$ & Epoch", body[:-1],
                      "Macro AUROC over all five labels on PTB-XL and over the eligible labels externally (Georgia: STTC, "
                      "CD, HYP; China: MI, STTC, CD, HYP). Per-label columns are AUROC (labels not eligible externally are "
                      "shown for completeness). $\\Delta$: external minus PTB-XL test macro AUROC on the same labels. "
                      "Epoch: selected epoch (of at most 20). Every reloaded model reproduced its saved PTB-XL validation "
                      "probabilities exactly before its single external inference.", size="\\scriptsize")
    fixed = read_csv(AMENDMENT / "resnet18/fixed_minus_adaptive.csv")
    body = [" & ".join([COHORTS[r["cohort"]], r["metric"].replace(" (eligible labels)", ""),
                        fmt_diff(float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"])),
                        p_cell(r)]) for r in fixed]
    second = longtable("ImageNet-pretrained ResNet-18: fixed-gain minus adaptive-gain AUROC [95\\% CI] and Holm-adjusted "
                       "$p$ (seeds 42--44).", "tab:s_resnet_fixed", "llcr", "Cohort & Metric & Difference & $p$", body,
                       "PTB-XL: hierarchical seed $\\times$ patient bootstrap, Holm over 2 comparisons. External cohorts: "
                       "seed $\\times$ record bootstrap, Holm over the 4 cohort $\\times$ metric comparisons. " + BOOT +
                       ", seed 2024. " + FLOOR_NOTE + "Prespecified in the amendment of 11 November 2026, after the frozen external "
                       "plan; not part of the confirmatory hypotheses H1 and H2.")
    return first + "\n\n" + second


def table_long40():
    if not (AMENDMENT / "long40/summary.csv").exists():
        return ("The 40-epoch sensitivity runs had not all completed when this document was generated; "
                "results will be added when they finish.")
    summary = read_csv(AMENDMENT / "long40/summary.csv")
    body = [" & ".join([HEAD_NAMES[r["head"]], r["budget"], f"{float(r['test_mean']):.3f} $\\pm$ {float(r['test_sd']):.3f}",
                        f3(float(r["validation_mean"])), r["selected_epochs"], r["epochs_run"]]) for r in summary]
    first = longtable("Training-budget sensitivity (amendment of 11 November 2026): 20 versus at most 40 epochs, "
                      "adaptive-gain pages, seeds 42--44.", "tab:s_long40", "lrcrll",
                      "Model & Max. epochs & Test macro AUROC & Validation & Selected epochs & Epochs run", body,
                      "Identical settings except the epoch budget (the cosine schedule spans the budget; early-stopping "
                      "patience 4 in both). Seed 42 of the 20-epoch CNN-only, CNN-LSTM, temporal-convolution and ViT runs "
                      "was trained on a CPU; all 40-epoch runs on the GPU.", size="\\scriptsize")
    rows = read_csv(AMENDMENT / "long40/hierarchical_test.csv")
    body, family = [], None
    for r in rows:
        if family is not None and r["family"] != family:
            body.append("\\midrule")
        family = r["family"]
        names = [HEAD_NAMES[h] for h in r["comparison"].split(" - ")]
        body.append(" & ".join([esc(r["family"]), " $-$ ".join(names),
                                fmt_diff(float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"])),
                                p_cell(r)]))
    second = longtable("Training-budget sensitivity: 40 minus 20 epochs per model, and the ten head pairs with each budget "
                       "(PTB-XL test, seeds 42--44).", "tab:s_long40_pairs", "llcr",
                       "Family & Comparison & Difference [95\\% CI] & $p$", body,
                       "Difference in mean-over-seeds test macro AUROC; hierarchical seed $\\times$ patient bootstrap "
                       "(" + BOOT + "), Holm within each family (5 or 10 comparisons). " + FLOOR_NOTE, size="\\scriptsize")
    return first + "\n\n" + second


def table_fixedgain_long40():
    if not (AMENDMENT / "fixedgain_long40/hierarchical_test.csv").exists():
        return ("The fixed-gain 40-epoch runs had not all completed when this document was generated; "
                "results will be added when they finish.")
    runs = read_csv(AMENDMENT / "fixedgain_long40/per_run.csv")
    epochs = {h: (" ".join(r["selected_epoch"] for r in runs if r["head"] == h),
                  " ".join(r["epochs_run"] for r in runs if r["head"] == h)) for h in dict.fromkeys(r["head"] for r in runs)}
    body, family = [], None
    for r in read_csv(AMENDMENT / "fixedgain_long40/hierarchical_test.csv"):
        if family is not None and r["family"] != family:
            body.append("\\midrule")
        family = r["family"]
        body.append(" & ".join([esc(r["family"]), HEAD_NAMES[r["comparison"]], r["metric"].replace(" AUROC", ""),
                                fmt_diff(float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"])),
                                p_cell(r)]))
    note = ("Fixed-gain amendment of 11 November 2026 (evening): the four fixed-gain networks retrained with at most "
            "40 epochs, otherwise identical, seeds 42--44, all on the GPU. Selected epochs (epochs run): " +
            "; ".join(f"{HEAD_NAMES[h]} {e[0]} ({e[1]})" for h, e in epochs.items()) + ". Hierarchical seed "
            "$\\times$ patient bootstrap on the PTB-XL test fold (" + BOOT + "), Holm within each family "
            "(8, 4 and 8 comparisons). " + FLOOR_NOTE + "No external inference was run for these models. The statement "
            "in the amendment text below that the capacity-matched head had no fixed-gain runs was true on 11 November; "
            "those runs were added on 15 November (Section~\\ref{sec:s_review}).")
    return longtable("Fixed-gain minus adaptive-gain training with a 40-epoch budget, fixed-gain 40 minus 20 epochs, "
                     "and the 20-epoch fixed-minus-adaptive differences for the same seeds (PTB-XL test, seeds 42--44).",
                     "tab:s_fg_long40", "lllcr", "Family & Model & AUROC & Difference [95\\% CI] & $p$", body, note,
                     size="\\scriptsize")


# ---------------------------------------------------------------- Additional analyses of 15 Nov 2026 (review)

MODEL_NAMES = {**NAMES, "resnet18": "ResNet-18, ImageNet", "pca_lda": "PCA + LDA", "pca_svm": "PCA + linear SVM",
               "resnet1d_wang": SIGNAL}
SHORT = {"cnn_pool": "CNN-only", "cnn_pool_wide": "Capacity-m.", "cnn_lstm": "CNN-LSTM", "temporal_conv": "Temporal conv.",
         "hybrid_vit": "ViT", "resnet18": "ResNet-18"}


def diff_cell(r):
    return fmt_diff(float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"]))


def pair_name(comparison, names=SHORT):
    b, a = comparison.split(" - ")
    return f"{names[b]} $-$ {names[a]}"


def review_intro():
    return ("\\label{sec:s_review}Everything in this section was added on 15 November 2026 in response to a review of the "
            "manuscript by the senior author, after all earlier results were known. Each part follows a dated protocol "
            "that was written and committed before the corresponding numbers existed (reproduced in "
            "Section~\\ref{sec:s_amendments}). The analyses are labelled \\textbf{Additional analysis} here and in the "
            "main text and are kept apart from the prospectively specified analyses; none of them changes a "
            "completed run, a saved prediction or the frozen external plan. Source: results/review\\_20261115/, "
            "generated by src/review\\_20261115.py.")


def table_wide():
    runs = read_csv(REVIEW / "fixedgain_wide/per_run.csv")
    body = [" & ".join([r["seed"], "GPU" if r["device"] == "cuda" else esc(r["device"]).upper(),
                        f"{r['selected_epoch']} of {r['epochs_run']}", f3(float(r["validation_macro_auroc"])),
                        f3(float(r["test_macro_auroc"])), f3(float(r["test_macro_ap"])), f3(float(r["test_macro_f1"])),
                        *[f3(float(r[f"test_auroc_{l}"])) for l in LABELS]]) for r in runs]
    values = np.asarray([[float(r[k]) for k in ("validation_macro_auroc", "test_macro_auroc", "test_macro_ap",
                                                "test_macro_f1", *[f"test_auroc_{l}" for l in LABELS])] for r in runs])
    body += ["\\midrule", " & ".join(["Mean", "", "", *[f3(v) for v in values.mean(0)]]),
             " & ".join(["SD", "", "", *[f3(v) for v in values.std(0, ddof=1)]])]
    meta = read_json(REVIEW / "fixedgain_wide/metadata.json")
    first = longtable("Additional analysis: capacity-matched CNN-only head trained on fixed-gain (2.5 mm/mV) pages, every "
                      "run (PTB-XL; test fold exploratory).", "tab:s_wide_runs", "lllrrrrrrrrr",
                      "Seed & Device & Selected epoch & Val. & Test & AP & F1 & " + " & ".join(LABELS), body,
                      "Protocol: amendment of 15 November 2026 (capacity-matched head on fixed-gain pages), frozen before "
                      "the first run. Config experiments/configs/ptbxl\\_v2\\_fg\\_cnn\\_pool\\_wide.json: the head's "
                      "adaptive-gain config with only the dataset fields changed (dataset " + esc(meta["dataset"]) + "); "
                      "identical training block, the same five seeds, the same validation-based selection, threshold 0.5 "
                      "and one test inference per run. Val./Test: macro AUROC; AP: macro average precision; F1: macro F1 "
                      "at 0.5; label columns: test AUROC. Run identifiers: ptbxl\\_v2\\_fg\\_cnn\\_pool\\_wide\\_seed42 to "
                      "seed46 (results/ptbxl\\_v2\\_fixedgain/), run on " + esc(runs[0]["started_utc"][:10]) + " (UTC).",
                      size="\\scriptsize")
    rows = read_csv(REVIEW / "fixedgain_wide/hierarchical.csv")
    own = [r for r in rows if r["family"] == "capacity-matched: fixed - adaptive"]
    ten = {(r["cohort"], r["metric"]): r for r in rows if r["family"].startswith("five heads") and r["comparison"] == "cnn_pool_wide"}
    body = [" & ".join([COHORTS[r["cohort"]], r["metric"].replace(" AUROC", ""), diff_cell(r), p_cell(r),
                        p_cell(ten[(r["cohort"], r["metric"])])]) for r in own]
    second = longtable("Additional analysis: capacity-matched head, fixed-gain minus adaptive-gain training: difference in "
                       "mean-over-seeds AUROC [95\\% CI] and Holm-adjusted $p$ (seeds 42--46).", "tab:s_wide_fixed", "llccc",
                       "Cohort & AUROC & Difference & $p$ (family of 2) & $p$ (family of 10)", body,
                       "Macro AUROC over the eligible labels of each cohort; HYP: hypertrophy AUROC. Hierarchical bootstrap "
                       "(seeds, then patients on PTB-XL or records externally; " + BOOT + "). Family of 2: the two "
                       "metrics of this head within the cohort (prespecified primary). Family of 10: this head together "
                       "with the four heads of the main fixed-gain analysis (five heads $\\times$ two metrics), as a "
                       "sensitivity analysis. All ten runs of this head were GPU-trained. External cohorts: one inference "
                       "per run, outside the confirmatory hypotheses H1 and H2. " + FLOOR_NOTE)
    pairs = [r for r in rows if r["family"] == "head pairs on fixed-gain pages"]
    by = {(r["cohort"], r["comparison"]): r for r in pairs}
    body = []
    for comparison in dict.fromkeys(r["comparison"] for r in pairs):
        cells = [pair_name(comparison)]
        for cohort in COHORTS:
            r = by.get((cohort, comparison))
            cells += [diff_cell(r), p_cell(r)] if r else ["--", "--"]
        body.append(" & ".join(cells))
    third = longtable("Additional analysis: head pairs on fixed-gain pages: difference in mean-over-seeds macro AUROC "
                      "[95\\% CI] and Holm-adjusted $p$ (seeds 42--46).", "tab:s_fixed_pairs", "lcccccc",
                      "Comparison & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in COHORTS.values()), body,
                      "The same ten pairs as Table~2 of the main text, with every model trained and tested on fixed-gain "
                      "pages. Capacity-m.: capacity-matched CNN-only head; Temporal conv.: CNN + temporal convolution; ViT: CNN + "
                      "small ViT. Holm over the 10 pairs within each cohort; " + BOOT + ". " + FLOOR_NOTE, size="\\scriptsize",
                      colsep="2pt")
    return first + "\n\n" + second + "\n\n" + third


def table_descriptive():
    lda = read_csv(REVIEW / "descriptive/lda_fixed_minus_adaptive.csv")
    body = [" & ".join([COHORTS[r["cohort"]], r["metric"].replace(" AUROC", ""), diff_cell(r), p_cell(r)]) for r in lda]
    first = longtable("Additional analysis: PCA + LDA, fixed-gain minus adaptive-gain pages: difference in AUROC [95\\% CI] "
                      "and Holm-adjusted $p$.", "tab:s_lda_fixed", "llcc", "Cohort & AUROC & Difference & $p$", body,
                      "The two single fits reported in the main text (no training seed, so the interval reflects test "
                      "sampling only: patients resampled on PTB-XL, records externally; " + BOOT + "). Holm over the 2 "
                      "metrics within each cohort. Post hoc. " + FLOOR_NOTE)
    gap = read_csv(REVIEW / "descriptive/image_minus_signal.csv")
    by = {(r["cohort"], r["comparison"]): r for r in gap}
    body, previous = [], None
    for comparison in dict.fromkeys(r["comparison"] for r in gap):
        model, page = comparison[:-1].split(" (")
        if previous is not None and (model == "resnet18") != (previous == "resnet18"):
            body.append("\\midrule")
        cells = [SHORT[model], page.capitalize(), by[("PTB-XL test", comparison)]["seeds"]]
        for cohort in COHORTS:
            r = by[(cohort, comparison)]
            cells += [diff_cell(r), p_cell(r)]
        body.append(" & ".join(cells))
        previous = model
    second = longtable("Additional analysis: image model minus signal reference: difference in mean-over-seeds macro AUROC "
                       "[95\\% CI] and Holm-adjusted $p$.", "tab:s_signal_gap", "lllcccccc",
                       "Model & Pages & Seeds & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in COHORTS.values()),
                       body, "Negative values: the image model is below the 1D ResNet on raw signals. Capacity-m.: capacity-matched "
                       "CNN-only head; Temporal conv.: CNN + temporal convolution; ViT: CNN + small ViT; ResNet-18: "
                       "ImageNet-pretrained. Compact heads: seeds "
                       "42--46; ResNet-18: seeds 42--44 (the signal reference restricted to the same seeds). Hierarchical "
                       "bootstrap, " + BOOT + ", Holm over the 12 comparisons within each cohort. Post hoc; seeds of "
                       "different architectures are paired by number only. An interval that includes zero means the two "
                       "models were not distinguishable in this sample; it does not show that they are equivalent. "
                       + FLOOR_NOTE, size="\\scriptsize", colsep="2pt")
    return first + "\n\n" + second


def table_clipping_defs():
    rows = read_csv(REVIEW / "clipping/clipping_definitions.csv")
    names = {"PTB-XL": "PTB-XL", "georgia": "Georgia, USA", "china": "China"}
    parts = {"train": "training (folds 1--8)", "validation": "validation (fold 9)", "test": "test (fold 10)", "all": "all",
             "included (confirmatory cohort)": "included (confirmatory)",
             "no mapped superclass (sensitivity only)": "unmapped (sensitivity only)", "all rendered records": "all rendered"}
    body, previous = [], None
    for r in rows:
        if previous is not None and r["cohort"] != previous:
            body.append("\\midrule")
        previous = r["cohort"]
        body.append(" & ".join([names[r["cohort"]], esc(parts.get(r["partition"], r["partition"])), n(r["pages"]),
                                n(r["outside_12mm_band_at_2p5"]), f"{float(r['outside_12mm_band_percent']):.2f}",
                                n(r["clipped_at_30mm_row"]), f"{float(r['clipped_at_30mm_row_percent']):.2f}",
                                n(r["hyp_positive_pages"]), n(r["hyp_positive_outside_band"]), n(r["hyp_positive_clipped"])]))
    return longtable("Additional analysis: the two amplitude boundaries at 2.5 mm/mV, for PTB-XL and both external cohorts.",
                     "tab:s_clipping_defs", "llrrrrrrrr",
                     "Cohort & Records & Pages & Outside band & \\% & Clipped & \\% & HYP+ & HYP+ outside & HYP+ clipped",
                     body, "\\emph{Outside band}: at 2.5 mm/mV at least one sample lies beyond the $\\pm$12-mm band that the "
                     "original layout allows around a row centre. This is the criterion by which 2.5 mm/mV was chosen (at "
                     "least 98\\% of training pages must fit: 16{,}810 of 17{,}084 fit, 274 do not) and it is identical to "
                     "the set of ECGs that the adaptive renderer drew below 2.5 mm/mV. Nothing is clipped at this boundary. "
                     "\\emph{Clipped}: at least one sample lies beyond the edge of the 30-mm row ($\\pm$15 mm) and was "
                     "clipped to the row edge by the fixed-gain renderer. Every clipped page is also outside the band "
                     "(checked), so the 121 clipped training pages are a subset of the 274. Numerators: pages meeting the "
                     "definition; denominators: the pages of the row (column Pages). Stage: page rendering; no page was "
                     "excluded for clipping in any cohort. HYP+: pages with the hypertrophy label (external cohorts: the "
                     "mapped label); HYP+ outside: HYP+ pages outside the band. ``Unmapped'' rows are the external records without a "
                     "mapped superclass, used only in the "
                     "label-mapping sensitivity analysis. Source: the fixed-gain cache manifests.", size="\\scriptsize",
                     colsep="2pt")


def table_codes(top=22):
    rows = read_csv(REVIEW / "mapping/codes_by_cohort.csv")
    summary = read_json(REVIEW / "mapping/composition_summary.json")
    parts = []
    for cohort, name in (("georgia", "Georgia, USA"), ("china", "China")):
        s = summary[cohort]
        body = []
        for r in [r for r in rows if r["cohort"] == cohort][:top]:
            share = (f"{float(r['largest_share']):.2f} {r['largest_share_superclass']}" if r["largest_share"] != "" else
                     "not in PTB-XL")
            body.append(" & ".join([r["snomed_ct_code"], esc(clinical_name({"challenge_dx_name": r["clinical_name"],
                                    "challenge_abbreviation": r["abbreviation"]})),
                                    esc(r["assigned_superclasses"].replace("|", ", ")) or "--", share,
                                    n(r["included_records"]), f"{float(r['included_percent']):.1f}",
                                    n(r["excluded_records"]), f"{float(r['excluded_percent']):.1f}"]))
        note = (f"The {top} most frequent of {s['distinct_codes']} distinct codes in this cohort "
                f"({s['distinct_codes_in_ptbxl_table']} of them occur in the organizers' PTB-XL encoding and could "
                f"therefore be mapped). Included: {n(s['included'])} records; excluded for having no mapped superclass: "
                f"{n(s['excluded_no_mapped_superclass'])} ({s['excluded_percent_of_all']:.1f}\\% of all records). A record "
                "can carry several codes, so percentages do not sum to 100. Largest share: the largest share of a "
                "non-NORM superclass among PTB-XL records with that code. Of the excluded records, "
                f"{n(s['excluded_only_sinus_rate_or_rhythm_codes'])} carry only sinus rhythm or sinus-rate statements "
                f"(sinus rhythm, bradycardia, tachycardia, arrhythmia; {n(s['excluded_only_sinus_rhythm_code'])} only sinus "
                f"rhythm), {n(s['excluded_with_a_near_miss_code_share_0p80_to_0p90'])} carry a code with a share between "
                f"0.80 and 0.90, and {n(s['excluded_with_no_code_in_ptbxl_table'])} carry no code that occurs in PTB-XL. "
                f"Mean codes per record: {s['codes_per_record_mean']['included']:.2f} (included), "
                f"{s['codes_per_record_mean']['no_mapped_superclass']:.2f} (excluded).")
        parts.append(longtable(f"Additional analysis: SNOMED CT codes of included and excluded records, {name}.",
                               f"tab:s_codes_{cohort}", "lp{4.8cm}llrrrr",
                               "SNOMED CT & Clinical name (abbreviation) & Assigned & Largest share & Included & \\% & "
                               "Excluded & \\%", body, note, size="\\scriptsize"))
    return "\n\n".join(parts)


VARIANT_TEXT = ("V0: published cohort and 90\\% rule. V1: share $\\geq$0.99 (records left without a label dropped). V2: share "
                "$\\geq$0.80 on the published cohort. V3: published cohort plus every record excluded only for having no "
                "mapped superclass, counted as negative for all labels. V4: as V3 with share $\\geq$0.80. V5: as V3, and "
                "the codes that the organizers score as equivalent to an assigned code (right bundle branch block "
                "59118001 = complete right bundle branch block; complete left bundle branch block 733534002 = left bundle "
                "branch block) inherit its assignment. ")


def table_prevalence():
    rows = read_csv(REVIEW / "mapping/prevalence_by_variant.csv")
    names = {"PTB-XL test": "PTB-XL test", "georgia": "Georgia, USA", "china": "China"}
    body, previous = [], None
    for r in rows:
        if previous is not None and r["cohort"] != previous:
            body.append("\\midrule")
        previous = r["cohort"]
        eligible = r["eligible_labels"].split("|")
        cells = [names[r["cohort"]], r["variant"], n(r["records"])]
        for label in LABELS[1:]:
            mark = "" if label in eligible else "$^*$"
            cells.append(f"{n(r[f'positives_{label}'])} ({100 * float(r[f'prevalence_{label}']):.1f}){mark}")
        body.append(" & ".join(cells))
    return longtable("Additional analysis: records and positives (prevalence, \\%) per superclass under each label-mapping "
                     "variant.", "tab:s_prevalence", "llrrrrr", "Cohort & Variant & Records & " + " & ".join(LABELS[1:]),
                     body, VARIANT_TEXT + "$^*$Not eligible in that variant (fewer than 50 positives or 50 negatives). "
                     "NORM has no code in any variant.", size="\\scriptsize")


def table_mapping_perf():
    rows = read_csv(REVIEW / "mapping/performance_by_variant.csv")
    order = [(h, "adaptive") for h in (*HEADS, "resnet18")] + [(h, "fixed") for h in (*HEADS, "resnet18")]
    order += [("pca_lda", "adaptive"), ("pca_lda", "fixed"), ("resnet1d_wang", "signal")]
    parts = []
    for cohort, name in (("georgia", "Georgia, USA"), ("china", "China")):
        data = {(r["head"], r["page"], r["variant"]): r for r in rows if r["cohort"] == cohort}
        variants = list(dict.fromkeys(r["variant"] for r in rows if r["cohort"] == cohort))
        labels = {v: next(r["macro_labels"] for r in rows if r["cohort"] == cohort and r["variant"] == v).replace("|", ", ")
                  for v in variants}
        body, previous = [], None
        for head, page in order:
            if previous is not None and page != previous:
                body.append("\\midrule")
            previous = page
            cells = [page.capitalize(), MODEL_NAMES[head]]
            cells += [f3(float(data[(head, page, v)]["macro_auroc"])) if (head, page, v) in data else "--" for v in variants]
            body.append(" & ".join(cells))
        parts.append(longtable(f"Additional analysis: external macro AUROC (mean over seeds) under each label-mapping "
                               f"variant, {name}.", f"tab:s_mapping_perf_{cohort}", "ll" + "r" * len(variants),
                               "Pages & Model & " + " & ".join(variants), body,
                               VARIANT_TEXT + "Macro AUROC over the labels of the published analysis that remain eligible "
                               "in the variant (" + "; ".join(f"{v}: {labels[v]}" for v in variants) + "). V0 reproduces "
                               "Table~4 of the main text. Predictions of the published cohort are the saved ones; the "
                               "added records received one inference per run.", size="\\scriptsize"))
    return "\n\n".join(parts)


def table_mapping_hyp():
    rows = read_csv(REVIEW / "mapping/hypotheses_by_variant.csv")
    parts = []
    for cohort, name in (("georgia", "Georgia, USA"), ("china", "China")):
        data = {(r["family"], r["comparison"], r["metric"], r["variant"]): r for r in rows if r["cohort"] == cohort}
        variants = list(dict.fromkeys(r["variant"] for r in rows if r["cohort"] == cohort))
        body, previous = [], None
        for family, comparison, metric in dict.fromkeys((r["family"], r["comparison"], r["metric"]) for r in rows):
            if previous is not None and family != previous:
                body.append("\\midrule")
            previous = family
            if family == "H1":
                label = pair_name(comparison)
            else:
                label = f"{SHORT[comparison.split(':')[0]]}, {metric.replace(' AUROC', '')}"
            cells = [family, label]
            for v in variants:
                r = data.get((family, comparison, metric, v))
                if r is None:
                    cells.append("--")
                elif v == "V3":
                    cells += [diff_cell(r), p_cell(r)]
                else:
                    cells.append(f"{float(r['difference']):+.3f}".replace("-", "$-$"))
            body.append(" & ".join(cells))
        header = "Family & Comparison & " + " & ".join("V3 [95\\% CI] & V3 $p$" if v == "V3" else v for v in variants)
        parts.append(longtable(f"Additional analysis: the external hypotheses H1 and H2 under each label-mapping variant, "
                               f"{name}: difference in mean-over-seeds AUROC.", f"tab:s_mapping_hyp_{cohort}",
                               "ll" + "".join("cc" if v == "V3" else "r" for v in variants), header, body,
                               VARIANT_TEXT + "H1: head pairs on adaptive-gain pages (macro AUROC). H2: fixed-gain minus "
                               "adaptive-gain training (macro and HYP AUROC). V0 repeats the published point estimates. V3: "
                               "hierarchical seed $\\times$ record bootstrap (" + BOOT + "), Holm within H1 (10) and H2 (8); "
                               "the other variants are point estimates. " + FLOOR_NOTE, size="\\scriptsize"))
    return "\n\n".join(parts)


def table_stats():
    rows = read_csv(REVIEW / "stats_revision/old_vs_revised.csv")
    summary = read_json(REVIEW / "stats_revision/summary.json")
    titles = {"comparisons/ptbxl_heads_hierarchical/pairwise.csv": "PTB-XL head pairs (S7)",
              "fixed_gain/retrained_comparison/hierarchical_test.csv": "PTB-XL fixed $-$ adaptive (Table 3, S8)",
              "external/georgia/hypotheses.csv": "Georgia H1, H2 (Tables 2, 3)",
              "external/china/hypotheses.csv": "China H1, H2 (Tables 2, 3)",
              "amendment_20261111/resnet18/fixed_minus_adaptive.csv": "ResNet-18 fixed $-$ adaptive (S15)",
              "amendment_20261111/long40/hierarchical_test.csv": "40-epoch sensitivity (S17)",
              "amendment_20261111/fixedgain_long40/hierarchical_test.csv": "Fixed gain, 40 epochs (S18)"}
    groups = {}
    for r in rows:
        groups.setdefault((r["file"], r["family"]), []).append(r)
    true = lambda r, key: r[key] == "True"
    body = []
    for (file, family), members in groups.items():
        size = int(members[0]["family_size"])
        floor = float(members[0]["holm_floor_revised"])
        body.append(" & ".join([titles[file], esc(family), str(size),
                                str(sum(float(r["p_holm_old"]) == 0 for r in members)),
                                str(sum(true(r, "at_resolution_floor") for r in members)),
                                "$\\le$" + f"{np.ceil(floor * 1000 - 1e-9) / 1000:.3f}",
                                str(sum(true(r, "supported_old") for r in members)),
                                str(sum(true(r, "supported_revised") for r in members))]))
    first = longtable("Additional analysis: bootstrap $p$-values before and after the statistical revision of 15 November "
                      "2026, by comparison family.", "tab:s_stats_families", "p{3.9cm}p{4.0cm}rrrlrr",
                      "Analysis & Family & $m$ & Old $p_\\mathrm{adj}=0$ & At floor & Printed as & Old $<$0.05 & Revised $<$0.05",
                      body, f"Old: {n(summary['resamples_old'])} resamples, $p=2\\min\\{{P(\\Delta^*\\le0),P(\\Delta^*\\ge0)\\}}$, "
                      "which is 0 when no replicate crosses zero and stays 0 after Holm correction; such values were printed "
                      f"as ``$<$0.001''. Revised: {n(summary['resamples_revised'])} resamples and the add-one estimator "
                      "$p=2\\min\\{(1+\\#[\\Delta^*\\le0])/(B+1),(1+\\#[\\Delta^*\\ge0])/(B+1)\\}$, whose smallest value is "
                      f"{summary['raw_p_floor_revised']:.4f}; after Holm correction in a family of $m$ comparisons the smallest "
                      "attainable value is $m$ times that (column ``Printed as''). $m$: comparisons in the Holm family. "
                      "Old $p_\\mathrm{adj}=0$: comparisons whose old adjusted $p$ was exactly zero. At floor: comparisons "
                      "with no replicate across zero in the revised bootstrap. Last two columns: comparisons with adjusted "
                      f"$p<0.05$. In total {summary['comparisons']} comparisons; {summary['old_adjusted_p_exactly_zero']} had "
                      f"an old adjusted $p$ of zero; {summary['support_at_0p05_changed']} changed side of 0.05. The first "
                      f"{n(summary['resamples_old'])} replicates of the revised bootstrap are the old random draws and "
                      "reproduce every archived old value exactly (checked). The old outputs are archived in "
                      "results/stats\\_legacy\\_2000resamples\\_20261115/; every comparison is listed in "
                      "results/review\\_20261115/stats\\_revision/old\\_vs\\_revised.csv.", size="\\scriptsize")
    changed = [r for r in rows if true(r, "support_changed")]
    near = [r for r in rows if not true(r, "support_changed")
            and min(float(r["p_holm_old"]), float(r["p_holm_revised"])) < 0.1
            and max(float(r["p_holm_old"]), float(r["p_holm_revised"])) >= 0.01
            and abs(float(r["p_holm_old"]) - float(r["p_holm_revised"])) >= 0.005]
    body = []
    for label, group in (("Changed side of 0.05", changed), ("Other changes of at least 0.005 near 0.05", near)):
        if not group:
            body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}: none}}}}")
            continue
        body.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{label}}}}}")
        for r in group:
            body.append(" & ".join([titles[r["file"]], esc(r["family"]), esc(r["comparison"]),
                                    f"{float(r['difference']):+.3f}".replace("-", "$-$"), f"{float(r['p_holm_old']):.3f}",
                                    f"{float(r['p_holm_revised']):.3f}"]))
    second = longtable("Additional analysis: comparisons whose Holm-adjusted $p$ moved across or near 0.05 with the revised "
                       "bootstrap.", "tab:s_stats_changed", "p{3.6cm}p{3.6cm}p{4.2cm}rrr",
                       "Analysis & Family & Comparison & Difference & Old $p$ & Revised $p$", body,
                       "Changes away from the resolution floor come from Monte Carlo error only (2{,}000 versus 10{,}000 "
                       "resamples) and from the add-one term.", size="\\scriptsize")
    return first + "\n\n" + second


def mapping_validation_text():
    return ("\\paragraph{Validation of the label mapping without an expert.} No cardiologist reviewed the mapping; this "
            "remains a limitation. Three checks were possible. (i) \\emph{Names.} Every code has a clinical name in the "
            "Challenge organizers' code tables (Supplementary Table~\\ref{tab:s_mapping}), and the name agrees in meaning "
            "with the SNOMED CT concept for all assigned codes except anterior myocardial infarction (54329005), whose "
            "SNOMED CT concept is \\emph{acute} myocardial infarction of the anterior wall. (ii) \\emph{Clinical meaning "
            "against the PTB-XL superclass definitions.} The assignments of the MI, STTC and CD codes agree with PTB-XL's own "
            "statement table: ischemic and nonspecific ST-T changes and a prolonged QT interval belong to STTC, and "
            "bundle-branch, fascicular and atrioventricular blocks and the Wolff-Parkinson-White pattern to CD. For HYP, "
            "left atrial enlargement and right atrial hypertrophy are assigned because PTB-XL places atrial "
            "overload/enlargement in its hypertrophy superclass. One assignment has no clinical basis: right ventricular "
            "hypertrophy (89792004) is assigned to CD as well as HYP only because 94\\% of the 126 PTB-XL records with it "
            "also have a conduction disturbance (co-occurrence); PTB-XL itself classifies right ventricular hypertrophy as "
            "HYP only. Variant V1 removes this assignment. (iii) \\emph{What the rule missed.} T-wave abnormality "
            "(164934002; STTC share 0.88) is the most frequent near miss; abnormal QRS and Q-wave abnormality reach 0.80--0.81 "
            "for MI, but PTB-XL treats the corresponding statements as descriptions of form, not as diagnoses, so V2 and "
            "V4, which add all three, are deliberately liberal. Codes that never occur in the organizers' PTB-XL encoding "
            "cannot be assigned by a data-driven rule at all: right bundle branch block (59118001) and complete left bundle "
            "branch block (733534002), which the organizers score as equivalent to assigned codes (variant V5 adds them), "
            "and left ventricular high voltage (55827005; China only), which stays unmapped in every variant because "
            "assigning it to HYP would be a clinical judgement that we are not qualified to make.")


# ---------------------------------------------------------------- text sections

def model_cards():
    # \sloppy from here to the end of the document: the model cards and the reproduced amendments contain long
    # file paths and hashes that otherwise run into the margin
    lines = ["\\sloppy", inline(COMMON), "", "\\begin{itemize}"]
    lines += [f"\\item \\textbf{{{esc(name)}.}} {inline(text)}" for name, text in CARDS.items()]
    lines += ["\\end{itemize}", "Every completed run, with hardware, selected epoch and results, is listed in "
              "Supplementary Table~\\ref{tab:s_runs}. Generated from src/make\\_model\\_cards.py."]
    return "\n".join(lines)


def amendments():
    parts = []
    doc = (ROOT / "docs/protocol_amendment_2026-11-03.md").read_text(encoding="utf-8")
    parts += ["\\subsection{Amendment of 3 November 2026: seeds 45/46 and a capacity-matched head}", markdown(doc)]
    parts += ["\\subsection{External-test amendment of 6 November 2026}",
              "Recorded in experiments/external/amendments.json; the hash chain is checked before every external step."]
    for a in read_json(ROOT / "experiments/external/amendments.json"):
        parts.append(f"\\begin{{itemize}}\\item Time (UTC): {esc(a['utc'])}; file: {esc(a['file'])}.\\item "
                     f"{esc(a['reason'])}\\end{{itemize}}")
    doc = (ROOT / "docs/protocol_amendment_2026-11-11_signal_seeds.md").read_text(encoding="utf-8")
    parts += ["\\subsection{Amendment of 11 November 2026: signal-reference seeds 45 and 46}", markdown(doc)]
    doc = (ROOT / "docs/protocol_amendment_2026-11-11.md").read_text(encoding="utf-8")
    parts += ["\\subsection{Amendment of 11 November 2026: 40-epoch sensitivity and a pretrained ResNet-18}", markdown(doc)]
    doc = (ROOT / "docs/protocol_amendment_2026-11-11_fixedgain_long40.md").read_text(encoding="utf-8")
    parts += ["\\subsection{Amendment of 11 November 2026 (evening): fixed-gain training with a 40-epoch budget}",
              markdown(doc)]
    for name, title in (("fixedgain_wide", "Amendment of 15 November 2026: capacity-matched head on fixed-gain pages"),
                        ("bootstrap_inference", "Statistical revision of 15 November 2026: finite-resample bootstrap "
                         "$p$-values"),
                        ("label_mapping_sensitivity", "Amendment of 15 November 2026: label-mapping documentation and "
                         "sensitivity"),
                        ("descriptive_additions", "Protocol note of 15 November 2026: two descriptive additions")):
        doc = (ROOT / f"docs/protocol_amendment_2026-11-15_{name}.md").read_text(encoding="utf-8")
        parts += [f"\\subsection{{{title}}}", markdown(doc)]
    return "\\label{sec:s_amendments}" + "\n\n".join(parts)


# ---------------------------------------------------------------- bibliography

def bibliography(body):
    keys = list(dict.fromkeys(k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for k in group.split(",")))
    entries = {}
    for kind, key, fields in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", BIB.read_text(encoding="utf-8"), re.S):
        entries[key] = {m.group(1).lower(): m.group(2).replace("{", "").replace("}", "")
                        for m in re.finditer(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", fields, re.M)}
    lines = ["\\begin{thebibliography}{99}"]
    for key in keys:
        e = entries[key]
        authors = [a.strip() for a in e["author"].split(" and ")]
        names = ", ".join("et al." if a == "others" else a for a in authors)
        venue = e.get("journal") or e.get("booktitle") or e.get("howpublished") or e.get("note", "")
        detail = ", ".join(x for x in (e.get("volume"), e.get("pages")) if x)
        lines.append(f"\\bibitem{{{key}}} {names.rstrip('.')}. {e['title']}. \\emph{{{venue}}}"
                     f"{' ' + detail if detail else ''} ({e['year']}). doi:\\nolinkurl{{{e['doi']}}}")
    lines.append("\\end{thebibliography}")
    return "\n".join(lines)


def main():
    runs = all_runs()
    per_run, per_label = table_per_run(runs)
    epochs, totals = table_epochs(runs)
    sections = [
        ("Cohorts and labels", [table_cohort(), table_external_flow(), table_mapping()]),
        ("Every run: results and selected epochs", [per_run, per_label, epochs]),
        ("Sensitivity analyses with GPU-trained seeds only", [table_gpu_pairs(), table_gpu_fixed()]),
        ("Calibration and validation-chosen thresholds", [table_calibration(runs), table_thresholds(runs)]),
        ("External per-label results", [table_external_labels()]),
        ("Gain policy: public resources and clipping at fixed gain", [table_survey(), table_clipping()]),
        ("Sensitivity analyses of 11 November 2026: a pretrained ResNet-18 and a longer training budget",
         [table_resnet(), table_long40(), table_fixedgain_long40()]),
        ("Additional analyses of 15 November 2026 (review by the senior author)",
         [review_intro(), "\\subsection{Capacity-matched head on fixed-gain pages}", table_wide(),
          "\\subsection{PCA + LDA under fixed gain, and the gap to the signal reference}", table_descriptive(),
          "\\subsection{Clipping: two boundaries and the external cohorts}", table_clipping_defs(),
          "\\subsection{Label mapping: clinical names, excluded records and sensitivity}", mapping_validation_text(),
          table_codes(), table_prevalence(), table_mapping_perf(), table_mapping_hyp(),
          "\\subsection{Bootstrap $p$-values and Holm correction}", table_stats()]),
        ("Model cards", [model_cards()]),
        ("Protocol amendments", [amendments()]),
    ]
    body = []
    for title, parts in sections:
        body.append(f"\\section{{{title}}}")
        body += parts
    body = "\n\n".join(body)
    preamble = r"""% Generated by src/make_supplementary.py from saved results; do not edit by hand.
\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\makeatletter\g@addto@macro\UrlBreaks{\do\_\do\-}\makeatother
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\setcounter{secnumdepth}{2}
\makeatletter % room for the two-digit section number S10 in the table of contents
\patchcmd{\l@section}{1.5em}{2.1em}{}{\PackageWarningNoLine{supplement}{table-of-contents patch failed}}
\renewcommand*\l@subsection{\@dottedtocline{2}{2.1em}{2.9em}}\makeatother
\title{Supplementary Information\\[4pt] \large """ + TITLE + r"""}
\author{Miriam Istanboli and Mohamed Elgendi}
\date{}
\begin{document}
\maketitle
\noindent Every number in this document was generated from the saved run records and analysis outputs by
\nolinkurl{src/make_supplementary.py}. PTB-XL test-fold (fold 10) results are exploratory, because that fold had been inspected
before the models were specified; the external cohorts are the confirmatory test. A completed TRIPOD+AI checklist
\cite{collins2024tripodai} is provided as a separate file.
\tableofcontents
"""
    text = preamble + "\n" + body + "\n\n" + bibliography(preamble + body) + "\n\\end{document}\n"
    OUT.write_text(text, encoding="utf-8")
    numbers = {label: f"S{i + 1}" for i, label in enumerate(re.findall(r"\\label\{(tab:s_[^}]+)\}", text))}
    print(f"wrote {OUT} ({len(text):,} characters)")
    print("table numbers:", json.dumps(numbers))
    print("late selection (epoch >= 18):", totals)


if __name__ == "__main__":
    main()
