"""Consistency audit: every "difference [lower, upper]" triple printed in the manuscript must exist in a result file.

Usage:
    python -X utf8 -m src.audit_manuscript_numbers

Reads paper/ieee_manuscript/main_source_scirep.tex and every analysis output that holds bootstrap differences
(head pairs, fixed versus adaptive, external hypotheses, the dated sensitivity analyses and the Additional
analyses of 15 Nov 2026). A triple matches when its three numbers, rounded to three decimals, equal a row's
(difference, ci_lower, ci_upper), or that row with the sign reversed (the text sometimes says "below by ...").
Triples that match nothing are listed; so are Supplementary Table numbers that the supplement does not define.
Writes nothing; exits with code 1 if anything is unmatched.
"""
from __future__ import annotations

import csv
import re
import sys

from src.common import ROOT

PAPER = ROOT / "paper/ieee_manuscript"
SOURCES = ("results/comparisons/ptbxl_heads_hierarchical/pairwise.csv",
           "results/fixed_gain/retrained_comparison/hierarchical_test.csv",
           "results/external/georgia/hypotheses.csv", "results/external/china/hypotheses.csv",
           "results/amendment_20261111/resnet18/fixed_minus_adaptive.csv",
           "results/amendment_20261111/long40/hierarchical_test.csv",
           "results/amendment_20261111/fixedgain_long40/hierarchical_test.csv",
           "results/review_20261115/fixedgain_wide/hierarchical.csv",
           "results/review_20261115/descriptive/lda_fixed_minus_adaptive.csv",
           "results/review_20261115/descriptive/image_minus_signal.csv",
           "results/review_20261115/mapping/hypotheses_by_variant.csv")
EXTRA = {(-0.043, -0.054, -0.033): "results/fixed_gain/transfer_test/summary.md (PCA + LDA transfer)"}


def rounded(x):
    return round(float(x) + 0.0, 3) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def known_triples():
    known = dict(EXTRA)
    for relative in SOURCES:
        path = ROOT / relative
        if not path.exists():
            print(f"missing source (skipped): {relative}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r.get("ci_lower") in (None, ""):
                    continue
                known[(rounded(r["difference"]), rounded(r["ci_lower"]), rounded(r["ci_upper"]))] = relative
    return known


def main():
    text = (PAPER / "main_source_scirep.tex").read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%.*", "", text)
    number = r"\$?([+-]|\$-\$)?\$?(\d\.\d{3})\$?"
    pattern = re.compile(number + r"\s*\[" + number + r",\s*" + number + r"\]")

    def value(sign, digits):
        return rounded(("-" if sign and "-" in sign else "") + digits)

    known, missing, found = known_triples(), [], 0
    for m in pattern.finditer(text):
        triple = (value(m.group(1), m.group(2)), value(m.group(3), m.group(4)), value(m.group(5), m.group(6)))
        mirror = (rounded(-triple[0]), rounded(-triple[2]), rounded(-triple[1]))
        if triple in known or mirror in known:
            found += 1
        else:
            line = text.count("\n", 0, m.start()) + 1
            missing.append((line, m.group(0), text[max(0, m.start() - 70):m.start()].replace("\n", " ")))
    print(f"difference [CI] triples in the manuscript: {found + len(missing)}; matched to a result file: {found}")
    for line, shown, context in missing:
        print(f"  UNMATCHED line {line}: {shown}   ...{context}")
    supplement = (PAPER / "supplementary_scirep.tex").read_text(encoding="utf-8")
    defined = len(re.findall(r"\\label\{tab:s_[^}]+\}", supplement))
    cited = set()
    for group in re.findall(r"Supplementary Tables?~((?:S\d+(?:--|, | and )?)+)", text):
        numbers = [int(x) for x in re.findall(r"S(\d+)", group)]
        cited.update(range(numbers[0], numbers[-1] + 1) if "--" in group else numbers)
    undefined = sorted(n for n in cited if n > defined)
    uncited = sorted(set(range(1, defined + 1)) - cited)
    print(f"supplementary tables defined: S1-S{defined}; cited in the manuscript: {len(cited)}; "
          f"cited but not defined: {undefined}; defined but not cited: {uncited}")
    placeholders = sorted(set(re.findall(r"@@[A-Z_0-9]+@@", text)))
    if placeholders:
        print("unfilled placeholders:", placeholders)
    if missing or undefined or placeholders:
        sys.exit(1)


if __name__ == "__main__":
    main()
