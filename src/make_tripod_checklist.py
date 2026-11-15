"""TRIPOD+AI checklist for the Scientific Reports manuscript (Markdown and Word copies from one item list).

Usage:
    python -X utf8 -m src.make_tripod_checklist

Item numbers, sections and applicability follow the official TRIPOD+AI checklist (Collins et al., BMJ
2024;385:e078378, Table 2, and the TRIPOD+AI for Abstracts checklist), read on 11 Nov 2026 from the open-access
copy https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/. Item topics are abbreviated here; the official form has the
full wording. Locations refer to main_source_scirep.tex (sections) and supplementary_scirep.tex (Tables S1-S13,
Sections S1-S10; revised 15 Nov 2026). Writes paper/ieee_manuscript/tripod_ai_checklist.md and .docx (the .docx is plain Office Open XML
written with the standard library, so no extra package is needed).
"""
from __future__ import annotations

from xml.sax.saxutils import escape
import zipfile

from src.common import ROOT

OUT = ROOT / "paper/ieee_manuscript/tripod_ai_checklist"
DONE, PART, NO, NA = "Addressed", "Partly addressed", "Not addressed", "Not applicable"

# (section, item, abbreviated topic, where in the manuscript, status, note)
MAIN = [
    ("Title", "1", "Identify development/evaluation of a prediction model, target population and outcome",
     "Title", PART, "The title names ECG-image classification and external validation; the five diagnostic "
     "superclasses (outcome) and the PTB-XL population are given in the Abstract."),
    ("Abstract", "2", "See TRIPOD+AI for Abstracts", "Abstract", PART, "See the abstract items A1-A13 below."),
    ("Background", "3a", "Healthcare context, rationale, references to existing models",
     "Introduction (paragraphs 1-4)", DONE, ""),
    ("Background", "3b", "Target population, intended purpose and intended users",
     "Discussion (Limitations: methodological study without clinical evaluation); Supplementary S9 (model cards: "
     "research use only)", PART, "The models are research tools for comparing image heads and rendering policies; "
     "no clinical use or user group is intended."),
    ("Background", "3c", "Known health inequalities between sociodemographic groups", "--", NO,
     "Not examined; the study compares architectures and rendering policies, not subgroups."),
    ("Objectives", "4", "Study objectives; development, validation or both",
     "Introduction (final paragraph and list of contributions)", DONE, ""),
    ("Data", "5a", "Data sources for development and evaluation, rationale, representativeness",
     "Methods (Data and labels; External confirmatory test, Cohorts); Discussion (Generalization; Limitations)",
     DONE, ""),
    ("Data", "5b", "Dates of the collected data", "--", NO,
     "Not stated in the manuscript. PTB-XL was recorded October 1989 to June 1996 (Wagner et al. 2020); dates of the Challenge 2021 cohorts were not checked. The authors may add a sentence."),
    ("Participants", "6a", "Study setting, number and location of centres",
     "Methods (Data and labels; External confirmatory test, Cohorts)", PART,
     "Countries and source institutions are named; the number of PTB-XL recording sites is not stated."),
    ("Participants", "6b", "Eligibility criteria",
     "Methods (Data and labels; External confirmatory test, Plan); Supplementary Tables S1-S2", DONE, ""),
    ("Participants", "6c", "Treatments received and their handling", "--", NA,
     "Diagnostic classification of single ECGs; no treatment information is used."),
    ("Data preparation", "7", "Pre-processing and quality checks; similarity across groups",
     "Methods (Page rendering, including both clipping definitions; Training; External confirmatory test, Procedure); "
     "Supplementary Tables S2, S13, S24", PART,
     "Pre-processing and checks (checksums, reproduction check, clipping rates in every cohort) are reported; "
     "similarity across sociodemographic groups was not examined."),
    ("Outcome", "8a", "Outcome definition, time horizon, assessment, rationale, consistency across groups",
     "Methods (Data and labels; External confirmatory test, Label mapping; Clinical names and validation of the "
     "mapping); Results (External validation, sensitivity to the label mapping); Supplementary Tables S3, S25-S31",
     PART, "Diagnostic (same-ECG) outcome. External labels come from a data-driven mapping of SNOMED CT codes with "
     "clinical names; its sensitivity is reported. Consistency across sociodemographic groups not examined."),
    ("Outcome", "8b", "Qualifications and demographics of outcome assessors",
     "Methods (Data and labels: PTB-XL annotated by cardiologists; Clinical names and validation of the mapping); "
     "Discussion (Limitations)", PART,
     "Assessor details for the external cohorts are not stated in their source publications checked. The manuscript "
     "states that no cardiologist reviewed the code-to-superclass mapping."),
    ("Outcome", "8c", "Blinding of outcome assessment", "--", NA,
     "Labels pre-existed in the public datasets and were assigned without knowledge of the models."),
    ("Predictors", "9a", "Choice of initial predictors and any pre-selection",
     "Methods (Page rendering; Models)", DONE, "Inputs are the whole rendered page (all twelve leads) or the raw "
     "signals; no predictor selection."),
    ("Predictors", "9b", "Definition of all predictors, how and when measured",
     "Methods (Page rendering; Models, signal reference)", DONE, ""),
    ("Predictors", "9c", "Qualifications of predictor assessors", "--", NA,
     "Inputs are produced automatically by a deterministic renderer; no human interpretation."),
    ("Sample size", "10", "How the study size was arrived at and its sufficiency",
     "Methods (Data and labels; External confirmatory test, Label mapping: 50-positive rule); Supplementary "
     "Tables S1-S2", PART, "All eligible records were used; no formal sample-size calculation was made."),
    ("Missing data", "11", "Handling of missing data; reasons for omitting data",
     "Methods (Data and labels; External confirmatory test); Supplementary Tables S1-S2", DONE,
     "No input values are missing; records were excluded only by the stated criteria."),
    ("Analytical methods", "12a", "How the data were used and partitioned",
     "Methods (Data and labels: official folds; Training: selection on validation)", DONE, ""),
    ("Analytical methods", "12b", "Handling of predictors (rescaling, transformation, standardisation)",
     "Methods (Page rendering; Training; Models)", DONE, ""),
    ("Analytical methods", "12c", "Model type, rationale, building steps, hyperparameter tuning, internal validation",
     "Introduction; Methods (Models; Training)", DONE,
     "Hyperparameters prespecified, not tuned; PCA components chosen on validation."),
    ("Analytical methods", "12d", "Heterogeneity across clusters (e.g. countries)",
     "Methods (External confirmatory test); Results (External validation); Supplementary Tables S11, S28-S31", PART,
     "Each external cohort is analysed separately; no formal heterogeneity model."),
    ("Analytical methods", "12e", "Performance measures and plots; methods to compare models",
     "Methods (Evaluation and statistics, including the revised bootstrap p-values); Supplementary Tables S32-S33",
     DONE, "Hierarchical bootstrap with 10,000 resamples, finite-resample p-values, Holm correction; resolution "
     "limit stated."),
    ("Analytical methods", "12f", "Model updating (e.g. recalibration) after evaluation", "Methods (External "
     "confirmatory test, Procedure: no recalibration)", NA, "No model was updated."),
    ("Analytical methods", "12g", "How predictions were calculated for evaluation",
     "Methods (External confirmatory test, Procedure); Code availability", DONE, ""),
    ("Class imbalance", "13", "Class-imbalance methods and any recalibration", "Methods (Training: unweighted "
     "binary cross-entropy)", DONE, "No class-imbalance method was used."),
    ("Fairness", "14", "Approaches to address model fairness", "--", NO,
     "Not examined; no sociodemographic subgroup analysis was performed."),
    ("Model output", "15", "Model output and thresholds", "Methods (Training: threshold 0.5); Results (Calibration "
     "and thresholds); Supplementary Tables S9-S10", DONE, ""),
    ("Training versus evaluation", "16", "Differences between development and evaluation data",
     "Methods (External confirmatory test); Results (External validation); Discussion (Generalization; "
     "Limitations)", DONE, ""),
    ("Ethical approval", "17", "Ethics committee approval and consent or waiver", "Ethics declarations", PART,
     "Statements for PTB-XL, Chapman-Shaoxing and Ningbo verified from their publications; Georgia approval and the "
     "authors' institutional determination are marked to be confirmed by the authors."),
    ("Funding", "18a", "Source of funding and role of funders", "Acknowledgements", NO,
     "Placeholder, to be completed by the authors."),
    ("Conflicts of interest", "18b", "Conflicts of interest", "Competing interests", NO,
     "Placeholder, to be completed by the authors."),
    ("Protocol", "18c", "Where the protocol can be accessed", "Methods (Training; External confirmatory test, Plan); "
     "Code availability; Supplementary Section S10 (amendments)", DONE,
     "Protocols, the frozen plan and all dated amendments are in the linked repository and reproduced in the "
     "Supplementary Information."),
    ("Registration", "18d", "Registration information or statement of no registration",
     "Methods (External confirmatory test, Plan)", DONE, "Not registered; the frozen plan's hashes were recorded."),
    ("Data sharing", "18e", "Availability of study data", "Data availability", DONE, ""),
    ("Code sharing", "18f", "Availability of analytical code", "Code availability", PART,
     "A publicly readable repository is linked and its contents are listed. Still open for the authors: code "
     "licence, a versioned archive with a DOI, and availability of the model checkpoints."),
    ("Patient and public involvement", "19", "Patient and public involvement", "--", NO,
     "No patient or public involvement (secondary analysis of public data); not stated in the manuscript."),
    ("Participants", "20a", "Flow of participants, numbers with and without the outcome",
     "Results (External validation); Methods (Data and labels); Supplementary Tables S1-S2, S25-S27", DONE,
     "Includes the diagnoses of the records excluded for having no mapped superclass."),
    ("Participants", "20b", "Characteristics overall and per data source (demographics, events, missing data)",
     "Supplementary Tables S1-S2, S25-S27", PART,
     "Label prevalence per fold, cohort and mapping variant is given; age and sex are not reported."),
    ("Participants", "20c", "Comparison of development and evaluation data distributions",
     "Supplementary Tables S1-S2, S24, S27", PART,
     "Outcome prevalence and clipping rates compared; demographics not compared."),
    ("Model development", "21", "Number of participants and events in each analysis",
     "Methods (Data and labels); Supplementary Tables S1-S2", DONE, ""),
    ("Model specification", "22", "Full model specification and access", "Methods (Models; Training); Code "
     "availability; Supplementary Section S9 (model cards)", PART,
     "Architectures fully described and code linked; trained checkpoints are not in the repository (hashes "
     "recorded; availability to be stated by the authors)."),
    ("Model performance", "23a", "Performance with confidence intervals, including key subgroups",
     "Results (Tables 1-4, Figs. 3-4); Supplementary Tables S4-S11, S14-S23", PART,
     "Intervals for all comparisons; no sociodemographic subgroups."),
    ("Model performance", "23b", "Heterogeneity in performance across clusters",
     "Results (External validation, Table 4); Supplementary Tables S11, S28-S31", DONE, ""),
    ("Model updating", "24", "Results of any model updating", "--", NA, "No model was updated."),
    ("Interpretation", "25", "Overall interpretation, including fairness", "Discussion", PART,
     "Fairness is not discussed (not examined)."),
    ("Limitations", "26", "Limitations and their effects", "Discussion (Limitations)", DONE, ""),
    ("Usability", "27a", "Handling poor-quality or unavailable input data in practice", "Discussion (Limitations: "
     "clean renderings only)", NA, "Not intended for implementation."),
    ("Usability", "27b", "User interaction and required expertise", "--", NA, "Not intended for implementation."),
    ("Usability", "27c", "Next steps for future research", "Discussion (Limitations; Conclusion)", DONE, ""),
]
ABSTRACT = [
    ("A1", "Identify the study, population and outcome", PART, "Task and external validation named; outcome partly."),
    ("A2", "Healthcare context and rationale", DONE, "First sentence."),
    ("A3", "Objectives", DONE, ""),
    ("A4", "Sources of data", DONE, "PTB-XL; cohorts from the USA and China."),
    ("A5", "Eligibility criteria and setting", PART, "Labeled 12-lead records; setting not described."),
    ("A6", "Outcome to be predicted", PART, "Hypertrophy named; the five superclasses are not listed."),
    ("A7", "Model type, model-building steps, internal validation", DONE, "Shared encoder, five heads, five seeds."),
    ("A8", "Performance measures", DONE, "Macro and hypertrophy AUROC; Holm-adjusted p."),
    ("A9", "Number of participants and outcome events", PART, "Record counts given; events not given."),
    ("A10", "Predictors in the final model", DONE, "Rendered page images (and a signal reference)."),
    ("A11", "Performance estimates with confidence intervals", PART, "Differences and p-values; no intervals "
     "(200-word limit)."),
    ("A12", "Overall interpretation", DONE, "Final sentence; supported and unsupported differences are distinguished."),
    ("A13", "Registration", NO, "Not stated in the abstract; the study was not registered."),
]
INTRO = ("Item numbers, sections and applicability follow the official TRIPOD+AI checklist (Collins et al., BMJ "
         "2024;385:e078378, Table 2) and the TRIPOD+AI for Abstracts checklist. Topics are abbreviated; use the "
         "official form for the full item wording. Locations refer to the Scientific Reports manuscript (section "
         "names) and its Supplementary Information (Tables S1-S33, Sections S1-S10). Prepared 11 November 2026 and revised 15 "
         "November 2026 for the revised manuscript; statuses are deliberately conservative (an item is 'Addressed' only "
         "if the manuscript reports it). The authors should check it before submission; items 17, 18a, 18b and 18f "
         "depend on text that the authors still have to complete.")


def markdown():
    lines = ["# TRIPOD+AI checklist", "", INTRO, "",
             "| Section/topic | Item | Topic (abbreviated) | Where addressed | Status | Note |", "|---|---|---|---|---|---|"]
    lines += [f"| {s} | {i} | {t} | {w} | {st} | {n} |" for s, i, t, w, st, n in MAIN]
    lines += ["", "## TRIPOD+AI for Abstracts", "", "| Item | Topic (abbreviated) | Status | Note |", "|---|---|---|---|"]
    lines += [f"| {i} | {t} | {st} | {n} |" for i, t, st, n in ABSTRACT]
    counts = {s: sum(r[4] == s for r in MAIN) for s in (DONE, PART, NO, NA)}
    lines += ["", "Main checklist: " + ", ".join(f"{v} {k.lower()}" for k, v in counts.items()) + "."]
    return "\n".join(lines) + "\n"


def docx(path):
    def run(text, bold=False):
        props = "<w:rPr><w:b/><w:sz w:val=\"18\"/></w:rPr>" if bold else "<w:rPr><w:sz w:val=\"18\"/></w:rPr>"
        return f"<w:r>{props}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"

    def para(text, bold=False, style=None):
        p = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
        return f"<w:p>{p}{run(text, bold)}</w:p>"

    def table(header, rows):
        border = "".join(f"<w:{s} w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
                         for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        out = [f"<w:tbl><w:tblPr><w:tblW w:w=\"5000\" w:type=\"pct\"/><w:tblBorders>{border}</w:tblBorders></w:tblPr>"]
        for k, row in enumerate([header] + rows):
            out.append("<w:tr>" + "".join(f"<w:tc>{para(cell, bold=k == 0)}</w:tc>" for cell in row) + "</w:tr>")
        return "".join(out) + "</w:tbl>"

    body = [para("TRIPOD+AI checklist", True, "Title"), para(INTRO),
            table(["Section/topic", "Item", "Topic (abbreviated)", "Where addressed", "Status", "Note"],
                  [list(r) for r in MAIN]),
            para("TRIPOD+AI for Abstracts", True, "Heading1"),
            table(["Item", "Topic (abbreviated)", "Status", "Note"], [list(r) for r in ABSTRACT])]
    document = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
                + "".join(body) + "<w:sectPr><w:pgSz w:w=\"16838\" w:h=\"11906\" w:orient=\"landscape\"/>"
                "<w:pgMar w:top=\"1000\" w:right=\"1000\" w:bottom=\"1000\" w:left=\"1000\"/></w:sectPr></w:body></w:document>")
    files = {
        "[Content_Types].xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://"
            "schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/"
            "vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document.main+xml\"/></Types>",
        "_rels/.rels": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://"
            "schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas."
            "openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
            "</Relationships>",
        "word/document.xml": document,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)


def main():
    OUT.with_suffix(".md").write_text(markdown(), encoding="utf-8")
    docx(OUT.with_suffix(".docx"))
    print(f"wrote {OUT.with_suffix('.md')} and {OUT.with_suffix('.docx')} ({len(MAIN)} items + {len(ABSTRACT)} abstract items)")


if __name__ == "__main__":
    main()
