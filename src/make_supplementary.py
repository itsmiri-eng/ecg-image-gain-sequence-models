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
from src.external_test import holm
from src.make_model_cards import CARDS, COMMON, get
from src.make_paper_tables import HEADS, LABELS, NAMES, SEEDS, fmt_diff, fmt_p, run_path

OUT = ROOT / "paper/ieee_manuscript/supplementary_scirep.tex"
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
           "‘": "`", "“": "``", "”": "''", "µ": "$\\mu$", "°": "$^\\circ$", "−": "$-$"}
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
    for line in text.splitlines():
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


def longtable(caption, label, spec, header, rows, note="", size="\\footnotesize"):
    lines = ["\\begingroup" + size, "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{longtable}}{{{spec}}}",
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
            if fixed and head == "cnn_pool_wide":
                continue
            runs += [("Fixed" if fixed else "Adaptive", NAMES[head], head, s, run_path(head, s, fixed)) for s in SEEDS]
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
    body = []
    for r in read_csv(ROOT / "experiments/external/label_mapping.csv"):
        shares = [float(r[f"share_{l}"]) for l in LABELS]
        cells = [r["snomed_code"], n(r["ptbxl_records"])]
        cells += [("\\textbf{%.2f}" % s) if s >= 0.9 else f"{s:.2f}" for s in shares]
        cells.append(esc(r["assigned_superclasses"].replace("|", ", ")) or "--")
        body.append(" & ".join(cells))
    note = (f"Rule fixed before the mapping was computed: {esc(meta['rule'])}. Shares are the fraction of PTB-XL records "
            "carrying the SNOMED CT code (Challenge organizers' encoding of PTB-XL) that have each superclass; bold "
            f"shares meet the 90\\% rule. {meta['codes_assigned']} of {meta['codes_seen']} codes were assigned. "
            f"Headers checked: {n(meta['headers_checked'])}; age/sex mismatches: {meta['age_sex_mismatches']}.")
    return longtable("SNOMED CT code to PTB-XL superclass mapping used for the external cohorts.", "tab:s_mapping",
                     "lrrrrrrl", "SNOMED CT & PTB-XL records & " + " & ".join(LABELS) + " & Assigned", body, note,
                     size="\\scriptsize")


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
                        "adaptive gain are the original v1 runs.", size="\\scriptsize")
    labels = longtable("Every PTB-XL run: test AUROC per superclass.", "tab:s_runs_labels", "lllrrrrr",
                       "Pages & Model & Seed & " + " & ".join(LABELS), body_labels, size="\\scriptsize")
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
            "selection, but late selection may also mean that some models were still improving.")
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
                cells += [fmt_diff(float(row["difference"]), float(row["ci_lower"]), float(row["ci_upper"])),
                          fmt_p(float(row["p_holm"]))]
            body.append(" & ".join(cells))
    return longtable("PTB-XL head comparisons (adaptive gain) with all seeds and with GPU-trained seeds only: difference "
                     "in mean-over-seeds macro AUROC [95\\% CI] and Holm-adjusted $p$.", "tab:s_gpu_pairs",
                     "lcccc", "Comparison & \\multicolumn{2}{c}{Seeds 42--46} & \\multicolumn{2}{c}{Seeds 43--46, GPU only}",
                     body, "Hierarchical bootstrap (seeds, then patients; 2{,}000 resamples), Holm over the 10 pairs within "
                     "each fold and seed set. Seeds 43--46 exclude the CPU-trained seed 42 of the original networks.",
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
                cells += [fmt_diff(float(r["difference"]), float(r["ci_lower"]), float(r["ci_upper"])), fmt_p(adjusted)]
            body.append(" & ".join(cells))
    return longtable("PTB-XL test: fixed-gain minus adaptive-gain training with all seeds and with GPU-trained seeds "
                     "only [95\\% CI], Holm-adjusted $p$.", "tab:s_gpu_fixed", "llcccc",
                     "Architecture & AUROC & \\multicolumn{2}{c}{Seeds 42--46} & \\multicolumn{2}{c}{Seeds 43--46, GPU only}",
                     body, "Hierarchical bootstrap (seeds, then patients); Holm over the 8 comparisons within each seed set. "
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
    order = [(h, "adaptive") for h in HEADS] + [(h, "fixed") for h in ("cnn_pool", "cnn_lstm", "temporal_conv", "hybrid_vit")]
    order += [("pca_lda", "adaptive"), ("pca_svm", "adaptive"), ("pca_lda", "fixed"), ("resnet1d_wang", "signal")]
    names = {**NAMES, "pca_lda": "PCA + LDA", "pca_svm": "PCA + linear SVM", "resnet1d_wang": SIGNAL}
    body, previous = [], None
    for head, page in order:
        if previous is not None and page != previous:
            body.append("\\midrule")
        cells = [page.capitalize(), names[head], str(len(data[(head, page)]["georgia"]))]
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
                     "amendment (Section S8) as a descriptive addition.", size="\\scriptsize")


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
    return longtable("Pages that would need clipping if every PTB-XL page were drawn at one fixed gain.", "tab:s_clipping",
                     "rrrrrrr", "Gain (mm/mV) & Pages & \\% of " + n(summary["ecgs"]) +
                     " & Training & Validation & Test & HYP-positive", body,
                     "A page needs clipping when any sample of any lead falls outside its 30-mm row, the rule of the "
                     "fixed-gain renderer. The 2.5 mm/mV counts reproduce the clipping recorded in the fixed-gain "
                     "dataset for every ECG. Computed by src/count\\_gain\\_clipping.py with the unchanged renderer.")


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
                        fmt_p(float(r["p_holm"]))]) for r in fixed]
    second = longtable("ImageNet-pretrained ResNet-18: fixed-gain minus adaptive-gain AUROC [95\\% CI] and Holm-adjusted "
                       "$p$ (seeds 42--44).", "tab:s_resnet_fixed", "llcr", "Cohort & Metric & Difference & $p$", body,
                       "PTB-XL: hierarchical seed $\\times$ patient bootstrap, Holm over 2 comparisons. External cohorts: "
                       "seed $\\times$ record bootstrap, Holm over the 4 cohort $\\times$ metric comparisons. 2{,}000 "
                       "resamples, seed 2024. Prespecified in the amendment of 11 November 2026, after the frozen external "
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
                                fmt_p(float(r["p_holm"]))]))
    second = longtable("Training-budget sensitivity: 40 minus 20 epochs per model, and the ten head pairs with each budget "
                       "(PTB-XL test, seeds 42--44).", "tab:s_long40_pairs", "llcr",
                       "Family & Comparison & Difference [95\\% CI] & $p$", body,
                       "Difference in mean-over-seeds test macro AUROC; hierarchical seed $\\times$ patient bootstrap "
                       "(2{,}000 resamples), Holm within each family (5 or 10 comparisons).", size="\\scriptsize")
    return first + "\n\n" + second

# ---------------------------------------------------------------- text sections

def model_cards():
    lines = [inline(COMMON), "", "\\begin{itemize}"]
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
    return "\n\n".join(parts)


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
         [table_resnet(), table_long40()]),
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
