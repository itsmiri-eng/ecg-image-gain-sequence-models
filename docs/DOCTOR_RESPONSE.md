# Point-by-point response to the review of 15 November 2026

Manuscript: *Rendering gain and sequence modeling in 12-lead ECG-image classification: a controlled comparison
with external validation* (Scientific Reports layout; Overleaf project "ECG Paper - Miriam").

Everything new is labelled **Additional analysis** in the manuscript and in the Supplementary Information, and each
new analysis has a dated protocol that was written and committed before its results existed. "Statistically
supported" below means a Holm-adjusted p < 0.05. Table and section numbers refer to the revised files.
Point 9 is not covered here; Miriam is handling it.

**Three things changed the science, not just the wording** (details under points 2, 5 and 8):

1. Fixed gain also helps the capacity-matched head, and once the gain is fixed the sequence heads are only
   0.004–0.015 above it (not supported in Georgia). The paper's first claim is weaker than before.
2. Every adjusted "p < 0.001" was over-stated; they are now reported as bounds (≤ 0.001 to ≤ 0.002). No conclusion
   was lost, and one borderline comparison moved from 0.054 to 0.044.
3. Fixed gain lowers the macro AUROC of the PCA + LDA baseline on PTB-XL and in Georgia (statistically supported),
   so "never reduced performance" was false and has been removed.

---

## Point 1 — Supplementary Information PDF belongs to the old paper

**Doctor's comment.** Check the supplementary information PDF; it belongs to the old paper.

**Action taken.** Confirmed and corrected.
- The file `Supplementary_Information.pdf` in the Overleaf project is a leftover of the earlier paper that used to
  be in that project ("Domain Shift and Format Sensitivity in ECG Image Classification Pipelines", September 2026).
  It was never part of the new paper. It has been renamed on Overleaf to
  `OLD_PAPER_DomainShift_2026-09_Supplementary_Information.pdf` (not deleted).
- The new paper's supplement existed only as LaTeX source (`supplementary_scirep.tex`); no PDF of it had ever been
  stored. That source did belong to the new paper, but it was missing the analyses requested in this review.
- The supplement was regenerated from the saved results by script (nothing typed by hand) and now has Sections
  S1–S10 and Tables S1–S33.

**Checks made on the regenerated supplement.**
- Tables S1–S18 keep their old numbers, so every existing reference in the main text still points to the right
  table; the new tables are S19–S33 (Section S8). Every supplementary table S1–S33 is now cited in the main text
  (S1, S14 and S15 were not cited before), and each citation was checked against the generated numbering.
- Every table is produced from the current result files; tables that show p-values were regenerated with the
  corrected procedure (point 5), and their notes now say 10,000 resamples.
- Table S3 now gives the clinical name of every SNOMED CT code (point 3); Table S13's definition of "clipped" is
  explicit (point 4); Tables S4–S6, S9–S11 include the new fixed-gain runs of the capacity-matched head, marked ¶.
- The sentence "the capacity-matched head was never trained on fixed gain" is still printed, because it is part of
  the 11 November amendment text, which is reproduced unchanged; a note directly above it (Table S18) says that the
  runs were added on 15 November.

**Files changed.** `src/make_supplementary.py`, `paper/ieee_manuscript/supplementary_scirep.tex`; Overleaf:
`supplementary_scirep.tex`, renamed old PDF.

**Result / what is still open.** The supplement source on Overleaf is now the new paper's, identical to the
generated file (checksum-verified), and it compiles without errors to 28 pages; its full compile log was read and
shows no text or table outside the margins. **One step is still open:** a PDF of
it has to be produced and looked at by a person. This session could compile on Overleaf and read the compile log, but
it could not open the PDF (no LaTeX on the PC, and the PDF link needs an Overleaf login). To get the PDF: open
`supplementary_scirep.tex` in Overleaf, click Recompile, download the PDF.

---

## Point 2 — Run the capacity-matched head on fixed gain, same five seeds, same evaluation settings

**Doctor's comment.** The supplement says this head was never trained on fixed gain; run it with the same five
seeds and the same evaluation settings.

**How "the same five C's" was read.** As the five training seeds 42–46 (the only five-fold repetition in the
protocol). The other possible reading, the five diagnostic classes, is satisfied by the same runs. Please tell us
if something else was meant.

**Analysis performed (Additional analysis).**
- 5 new training runs: capacity-matched head on fixed-gain (2.5 mm/mV) pages, seeds 42, 43, 44, 45, 46, on the
  GPU, 15 Nov 2026 21:21–23:37. Config `experiments/configs/ptbxl_v2_fg_cnn_pool_wide.json` = the head's
  adaptive-gain config with only the dataset fields changed (training block identical, checked by script). Same
  validation-based checkpoint selection, threshold 0.5, one test inference per run.
- One external inference per run and cohort (Georgia, China) after the usual reproduction check (maximum
  difference 1.8e-7).
- Statistics as in the main fixed-gain analysis (hierarchical seed × patient bootstrap, now 10,000 resamples).
- Protocol: `docs/protocol_amendment_2026-11-15_fixedgain_wide.md` (committed 21:20, before the first run).
  Run IDs `ptbxl_v2_fg_cnn_pool_wide_seed42…46`; predictions in `results/ptbxl_v2_fixedgain/` and
  `results/external/<cohort>/`; analysis outputs in `results/review_20261115/fixedgain_wide/`.

**Result.**

| Cohort | Adaptive | Fixed | Fixed − adaptive, macro AUROC | Fixed − adaptive, HYP AUROC |
|---|---:|---:|---|---|
| PTB-XL test | 0.865 | 0.883 ± 0.006 | +0.017 [0.010, 0.024] | +0.043 [0.019, 0.069] |
| Georgia | 0.700 | 0.733 | +0.034 [0.020, 0.048] | +0.095 [0.047, 0.141] |
| China | 0.777 | 0.809 | +0.033 [0.021, 0.043] | +0.103 [0.075, 0.131] |

All six differences are statistically supported (p ≤ 0.001 in the head's own family of two; ≤ 0.002 in a family of
ten with the other heads).

**New finding that follows from it.** With every head trained on fixed gain, the sequence heads are only slightly
above the capacity-matched head:

| Sequence head − capacity-matched, fixed-gain pages | PTB-XL | Georgia | China |
|---|---|---|---|
| CNN-LSTM | +0.006 (p = 0.081) | +0.005 (p = 0.63) | +0.008 (p = 0.11) |
| Temporal conv. | +0.012 (p ≤ 0.002) | +0.004 (p = 0.33) | +0.015 (p = 0.004) |
| ViT | +0.008 (p = 0.081) | +0.004 (p = 0.74) | +0.009 (p = 0.029) |

With adaptive gain the same differences were 0.011–0.020 (PTB-XL) and 0.023–0.039 (LSTM and ViT externally).

**How the interpretation changed.** The paper used to say that sequence modelling adds accuracy beyond capacity.
It now says that this difference is small, halves with longer training and is largely absent on fixed-gain pages,
so part of it probably reflects the ability to use the calibration pulse and page gain. This is stated in the
Discussion and in the Conclusion, and marked as resting on an analysis added after
the other results were known.

**Where.** Results ("Gain-controlled retraining", "External validation"), Discussion, Conclusion, Methods
("Training"); Tables 1, 3, 4 (new rows marked ¶); Fig. 4 (new row marked †); Supplementary Tables S4–S6, S11,
S19, S20, S21; amendment reproduced in Section S10.

---

## Point 3 — SNOMED codes without clinical names; validate the mapping; effect of excluding ~80% of Chinese records

**Doctor's comment.** Add the clinical name of each SNOMED code, validate the mapping (expert review or mapping
analysis), and analyse how it affects the results, since about 80% of the Chinese records were excluded.

**Action taken.**
- **Names.** Every code now has its clinical name and abbreviation (Supplementary Table S3; the assigned codes are
  also named in the Methods). Source: the PhysioNet/CinC Challenge 2021 organisers' own code tables
  (`dx_mapping_scored.csv`, `dx_mapping_unscored.csv`, repository physionetchallenges/evaluation-2021). Each name
  was compared with the SNOMED CT concept name from an independent terminology server (EMBL-EBI OLS): 49 of the 50
  codes agree in meaning; for anterior myocardial infarction (54329005) the SNOMED CT concept says *acute*
  infarction. Table with sources: `experiments/external/snomed_code_names.csv` (all 112 codes that occur in the two
  cohorts: `snomed_code_names_all.csv`).
- **Validation.** **No cardiologist has reviewed the mapping; the paper now says so.** What could be done without
  an expert was done: each assignment was compared with PTB-XL's own definition of its superclasses. All agree
  except one: right ventricular hypertrophy is assigned to CD (as well as HYP) only because it co-occurs with
  conduction disturbances in PTB-XL, not because of its meaning. Two further problems are documented: codes the
  organisers treat as equivalent to assigned codes but that never occur in PTB-XL were left unmapped (right bundle
  branch block 59118001; complete left bundle branch block 733534002), and "left ventricular high voltage"
  (55827005; 5,400 Chinese records) cannot be mapped at all.
- **A request to you.** A short expert check of Table S3 and of this list would close the point properly.

**Analysis performed (Additional analysis; protocol `docs/protocol_amendment_2026-11-15_label_mapping_sensitivity.md`).**
- *Who was excluded* (Tables S25, S26). Georgia: 4,245 of 10,344 (41%) without a mapped label; 2,586 of them carry
  only sinus rhythm / sinus-rate statements, and 447 carry a near-miss code (mostly T-wave abnormality, STTC share
  0.88). China: 36,146 of 45,152 (80%); 19,851 only sinus rhythm / sinus-rate statements (14,478 sinus
  bradycardia), 5,933 with a near-miss code, 3,775 with left ventricular high voltage. The excluded records are
  therefore mostly ECGs without a morphological diagnosis; they are not a random 80%.
- *Class composition* (Table S27). Because only records with at least one mapped diagnosis were kept, every
  "negative" in the published cohorts is an ECG with some other abnormality, and prevalence is high (Georgia: STTC
  67%, HYP 35%; China: STTC 63%, CD 43%), unlike PTB-XL test (24%, 12%).
- *Sensitivity variants* (Tables S27–S31): stricter rule (V1), more liberal rule (V2), all records with unmapped
  ones counted as negatives (V3; needed rendering and scoring the 40,391 excluded records once with every model),
  V3 with the liberal rule (V4), V3 with the organisers' equivalent codes (V5).

**Result.** 
1. **Absolute external performance depends strongly on who is scored.** With all records (unmapped ones as
   negatives, V3) every model's macro AUROC is higher: by 0.03–0.06 in Georgia and 0.04–0.09 in China (signal
   reference 0.787 → 0.818 and 0.855 → 0.891). With the strict rule (V1) it is lower by 0.02–0.04 and 0.06–0.11.
   So the published cohorts are a harder task than the full cohorts, and the "external drop" relative to PTB-XL is
   partly a consequence of the cohort definition. The paper now says this and no longer reads the drop as a pure
   measure of generalisation.
2. **Head ordering.** CNN-only < capacity-matched < {LSTM, ViT} has the same sign in every variant and stays
   supported with all records, in both cohorts. The order *among* the sequence heads is not stable: in Georgia the
   temporal-convolution head is no longer supported as lower than LSTM/ViT (p = 0.070), and in China ViT − temporal
   conv. changes sign (+0.005 → −0.006 [−0.012, −0.001], p = 0.031).
3. **Fixed gain (H2).** Georgia: all eight differences positive in every variant and supported with all records
   (p ≤ 0.015). China: CNN-only and LSTM stay positive and supported; the hypertrophy differences stay positive and
   supported with all records but are smaller (+0.010 to +0.068 instead of +0.014 to +0.112); temporal conv. macro is
   no longer supported (+0.004 [−0.003, 0.011], p = 0.35). **With the strict rule the Chinese hypertrophy result
   largely disappears** (+0.014, +0.035, −0.005, +0.001 for CNN-only, LSTM, temporal conv., ViT): that rule drops
   the left-ventricular-hypertrophy code (share 0.91, the lowest accepted) and leaves 131 positives defined by
   right-ventricular and atrial codes. The fixed-gain hypertrophy benefit in China therefore rests mainly on the
   LVH code.
4. **No causal claim is made.** The analyses show how results vary with the cohort definition, not why performance
   differs between cohorts; the text says so.

**Where.** Results ("External validation": inclusion percentages; paragraph "Sensitivity to the label mapping"),
Methods ("Label mapping"; "Clinical names and validation of the mapping"), Discussion (Generalization;
Limitations, third item); Supplementary Table S3 and Section S8 (text on validation, Tables S25–S31).

---

## Point 4 — 274 training pages "do not fit" versus 121 clipped pages in Table S13; clipping in the external cohorts

**Doctor's comment.** The text implies 274 training pages do not fit, but Table S13 reports 121 clipped. Use
consistent row boundaries/definitions and report clipping for both external cohorts.

**What the two numbers are.** Both are correct; they refer to two different boundaries at 2.5 mm/mV.

| | Boundary | Meaning | Training pages | All PTB-XL pages |
|---|---|---|---:|---:|
| "Does not fit the band" | ±12 mm around the row centre | The criterion that *selected* 2.5 mm/mV (≥ 98% of training pages must fit: 16,810 of 17,084 do). Same pages the adaptive renderer drew below 2.5 mm/mV. Nothing is clipped here. | 274 (1.60%) | 350 (1.64%) |
| "Clipped" | ±15 mm = edge of the 30-mm row | At least one sample was actually clipped by the fixed-gain renderer. | 121 (0.71%) | 154 (0.72%) |

The 121 are a subset of the 274 (checked: every clipped page is outside the band). The old text mixed them by
writing "fit within the row band" next to "clipped at the row edge".

**Action taken.** The Methods ("Fixed gain") now defines both boundaries, gives numerator, denominator and stage
(page rendering, before any model; nothing excluded) for each, and states the subset relation. Table S13's caption
and note say which boundary it uses. New Table S24 shows both counts side by side for every partition and cohort.

**Clipping in the external cohorts (new).**

| Cohort | Pages | Outside ±12-mm band | Clipped at row edge |
|---|---:|---:|---:|
| PTB-XL (all) | 21,388 | 350 (1.64%) | 154 (0.72%) |
| Georgia | 6,047 | 78 (1.29%) | 28 (0.46%) |
| China | 9,004 | 795 (8.83%) | 383 (4.25%) |

Clipping is about six times more frequent in the Chinese cohort than in PTB-XL; this is now a stated limitation.

**Where.** Methods ("Page rendering", Fixed gain); Limitations (ninth item); Supplementary Tables S13 (note
rewritten) and S24 (new). Code: `src/review_20261115.py clipping`.

---

## Point 5 — Bootstrap p-values and Holm correction ("adjusted p < 0.01 because no bootstrap sample crossed zero")

**Doctor's comment.** The finite-bootstrap inference looks too optimistic; audit it.

**What was wrong.** p was computed as 2·min(fraction of replicates ≤ 0, fraction ≥ 0) from 2,000 resamples. When
no replicate crossed zero this is exactly 0, it stays 0 after Holm correction, and it was printed as "p < 0.001".
2,000 resamples cannot support that: the smallest two-sided p they can resolve is about 0.001, and about 0.01
after Holm correction over ten comparisons. 81 of the 143 comparisons in the paper were affected.

**Correction (protocol `docs/protocol_amendment_2026-11-15_bootstrap_inference.md`, committed before recomputing).**
- 10,000 resamples instead of 2,000.
- Finite-resample p-value that can never be zero: p = 2·min((1 + #≤0)/(B + 1), (1 + #≥0)/(B + 1))
  (Davison & Hinkley 1997; Phipson & Smyth 2010). Holm step unchanged, same families.
- The resolution is stated: smallest raw p = 0.0002; smallest adjusted p = family size × 0.0002 (0.002 for ten
  comparisons). Values at that limit are printed as "≤ 0.002" (or ≤ 0.001 in small families) and mean only "no
  replicate crossed zero".
- Check of the implementation: the first 2,000 draws of the new bootstrap are the old random draws, and they
  reproduce every one of the 143 archived old results exactly.

**What changed.**
- All 143 comparisons were recomputed (Tables 2 and 3; Supplementary S7, S8, S15, S17, S18; Fig. 4).
- 81 entries "p < 0.001" became "≤ 0.001" or "≤ 0.002". All remain below 0.05.
- Other p-values moved only by resampling noise. One crossed 0.05: ViT − LSTM on the PTB-XL test fold, 0.054 →
  0.044. It is reported as borderline and not used for a claim (it is 0.20 with GPU-only seeds and 0.50 on
  validation). In the GPU-only sensitivity set the LSTM's fixed-gain macro difference is not supported (p = 0.084);
  the text used to say the GPU-only set "gave the same pattern" and now states this exception.
- Confidence intervals moved by at most 0.0013.
- Old outputs are archived (`results/stats_legacy_2000resamples_20261115/`) and compared with the new ones in
  Supplementary Tables S32 and S33.

**Limitation that remains and is now stated.** The bootstrap resamples only five training seeds (three in the
sensitivity analyses), so intervals are probably somewhat too narrow, and the external cohorts have no patient
identifiers. The p-values are approximate.

**Where.** Methods ("Evaluation and statistics", new paragraph with the formula); Limitations (seventh item);
footnotes of Tables 2 and 3; Supplementary S32, S33. Code: `src/external_test.py`, two new unit tests.

---

## Point 6 — "Never reduced performance", "no architecture was worse", "improved every architecture"

**Doctor's comment.** Distinguish positive point estimates from statistically supported improvements; no implied
non-inferiority without a prespecified margin.

**Action taken.** The whole manuscript was audited, not only these sentences. A wording rule is now stated in the
Introduction and Methods: a difference is called "statistically supported" only if its Holm-adjusted p < 0.05;
otherwise it is given as a point estimate with its interval. No non-inferiority or equivalence claim is made, and
Limitations says so.

| Old wording | New wording (short) |
|---|---|
| Abstract: "Fixed-gain re-rendering never reduced performance and helped local models most" | "gave statistically supported gains for heads without row-wide context …, unsupported differences for the transformer, and lower macro AUROC for a linear baseline" |
| Abstract: "Both findings were confirmed … fixed gain improved hypertrophy AUROC for every architecture" | "Under a frozen plan … fixed gain raised hypertrophy AUROC for all four prespecified heads" |
| Results: "No architecture was worse with fixed gain"; "No difference was negative" | Point estimates positive for all five heads, but supported only for some; for the ViT (and LSTM hypertrophy) on PTB-XL "neither that fixed gain helps nor that it does no harm"; five of eight 40-epoch intervals include zero |
| Results H2: "improved … for every architecture" | Supported differences listed by cohort and metric; ViT macro in China +0.004 [−0.004, 0.012] "supports neither a benefit nor the absence of harm" |
| Contribution 3: "re-rendering that improves every architecture" | "whose effect is estimated for every architecture" |
| Conclusion: "improved every architecture on unseen data" | "associated with higher hypertrophy AUROC for every compact network on both external cohorts …; no benefit was statistically supported for the vision transformer on PTB-XL or for a pretrained ResNet-18 externally, and a linear baseline lost macro AUROC on PTB-XL" |
| "indistinguishable", "tied", "matched", "level with", "remained best", "replicated", "confirmed" | replaced by the estimate, its interval and p; "not distinguishable in this sample is not evidence of equivalence" |
| H2 in the frozen plan: "not worse" | Methods now explains that no margin was specified, that it was evaluated as a two-sided test, and how results are read |

**New evidence against the old claim.** For the PCA + LDA baseline, fixed gain lowers macro AUROC on PTB-XL
(−0.013 [−0.023, −0.003], p = 0.017) and in Georgia (−0.007 [−0.013, −0.002], p = 0.016); it raises it in China
(+0.019). Supplementary Table S22 (Additional analysis).

**Where.** Abstract, Introduction (contributions), all Results subsections, Discussion, Limitations (tenth item),
Conclusion, Methods (statistics; external hypotheses); README of the repository.

---

## Point 7 — Pulse-swap findings

**Doctor's comment.** Frame them as consistent with calibration reading, not as proof that the analysis uniquely
separates calibration reading from a gain prior.

**Action taken.**
- Results: the paragraph now reports the two interventions and their difference, says the pattern "is consistent
  with" whole-row heads using the pulse as a voltage scale, and then states that it "does not prove it, and it does
  not uniquely separate calibration reading from a gain prior", with four reasons (additivity is assumed;
  re-rendering also shrinks the waveform; a pulse-only swap creates an internally inconsistent page; post hoc, one
  seed, validation pages).
- Discussion: "Our analyses separate two consequences" → "are consistent with both without separating them
  conclusively"; "This explains why…" → "This account agrees with three observations… None of these observations
  tests the account directly, and other explanations… are not excluded."
- Abstract: "showed whole-row models reading the pulse" → "were consistent with whole-row models reading the pulse".
- Fig. 5b: legend labels changed from "Gain prior" / "Calibration reading" to "Consistent re-rendering" /
  "Pulse-only swap minus consistent re-rendering"; panel title and caption changed accordingly.
- Subsection title "Adaptive gain is a diagnostic cue" → "Adaptive gain carries diagnostic information".
- Limitations (eighth item): the comparison "does not identify a mechanism".

No number changed; no new analysis.

---

## Point 8 — ResNet-18 (0.929) versus temporal convolution (0.895); compact models versus ResNet-18; PCA + LDA

**Doctor's comment.** Temporal convolution is only the best of the compact models. Separate compact models from
ResNet-18 in the signal-versus-image gap. Acknowledge that fixed gain reduces PCA + LDA macro AUROC on PTB-XL.

**Action taken.**
- "With fixed gain, the temporal-convolution head reached the highest image-model test macro AUROC" → "had the
  highest test macro AUROC of the compact models (0.895 ± 0.001)… not [supported] against the LSTM or the ViT, and
  it was 0.034 below the pretrained ResNet-18 (0.929 ± 0.002), the best image model overall". The term "compact
  models" is defined in the Methods; Table 1's footnote and the abstract say the same.
- Signal-versus-image gap, now separate and with intervals (Additional analysis, Supplementary Table S23):

| Fixed gain, image − signal | PTB-XL | Georgia | China |
|---|---|---|---|
| Compact models (range over five heads) | −0.033 to −0.067 | −0.048 to −0.080 | −0.031 to −0.088 |
| ResNet-18 | +0.001 [−0.004, 0.005], p = 0.80 | −0.010 [−0.016, −0.004], p = 0.002 | −0.003 [−0.010, 0.004], p = 0.38 |

  All compact-model gaps are statistically supported. "ResNet-18 matched the signal reference" was replaced by
  these results (not distinguishable on PTB-XL and in China, lower in Georgia; no equivalence claimed).
- "The signal reference … exceeded the best image models by 0.03–0.04" and "about 0.05 and 0.03 above the best
  image models" now refer explicitly to the compact models.
- Discussion: the gap of the compact models "reflects capacity and pretraining" → "is more plausibly related to
  model capacity and pretraining…; ResNet-18 differs in both, so we cannot say which".
- PCA + LDA: stated in Results (with interval and p, see point 6), in the Discussion's summary of where fixed gain
  did and did not help, in the abstract ("lower macro AUROC for a linear baseline") and in the Conclusion.

**Where.** Abstract; Results ("Controlled head comparison", "Gain-controlled retraining", "External validation",
"A pretrained standard backbone"); Discussion; Conclusion; Table 1 footnote; Supplementary S22, S23.

---

## Point 9 — not addressed here (handled by Miriam)

---

## Point 10 — TRIPOD+AI checklist, Code Availability statement, reviewer-accessible repository

**TRIPOD+AI checklist.** `paper/ieee_manuscript/tripod_ai_checklist.docx` (and `.md`). The item list was checked
again against the official table (27 items, 52 sub-items, plus the 13 abstract items). Every item gives the section
or table where it is reported, or says "not addressed" / "not applicable" with the reason; statuses are deliberately
conservative. It was updated for the revised manuscript (new sections and tables). Four items depend on text the
authors still have to supply: ethics confirmations (17), funding (18a), competing interests (18b), licence/archive
(18f). It is a separate Word file (not in the Overleaf project, which
accepts only text and image files through the connection used here); it is in the repository and in the Desktop
folder `manuscript (Scientific Reports, for Overleaf)`. Totals: 23 items addressed, 16 partly addressed, 6 not
addressed, 7 not applicable.

**Code availability.** A separate "Code availability" section follows "Data availability". It names the repository,
lists what is in it (source code, preprocessing, model definitions, configuration of every run, analysis plans and
all dated amendments, evaluation and analysis scripts, saved predictions of every run on every cohort, experiment
log, environment and instructions), says what is not in it and why (source ECG data: obtained from PhysioNet;
rendered images: regenerated and checked by hash; model checkpoints), and that no account is needed. Three
bracketed items are for the authors to decide: licence, DOI archive, availability of checkpoints.

**Repository.** https://github.com/itsmiri-eng/ecg-image-gain-sequence-models
- Checked on 15 Nov 2026: the link opens without login and shows the repository (public).
- **Not yet true until Miriam pushes:** the public copy is behind the local repository. It lacks the work of
  11–16 November (ResNet-18, 40-epoch analyses, everything in this response). All of it is committed locally and
  goes online with one push from GitHub Desktop. Until then the statement in the manuscript describes the local
  repository, not the public one.
- No sensitive data are in the repository: no raw ECGs, only predictions for public, de-identified records.

**Where.** Manuscript ("Data availability", "Code availability"); `README.md`, `docs/REPRODUCIBILITY.md` (all
commands, including the new analyses).

---

## Where to look first

| If you want to check… | Look at |
|---|---|
| The new fixed-gain result for the capacity-matched head | Table 3 (rows marked ¶), Fig. 4 (row marked †), Supplementary S19–S21 |
| The corrected p-values | Tables 2 and 3; Methods "Evaluation and statistics"; Supplementary S32–S33 |
| Clinical names and the mapping | Supplementary Table S3; Section S8 "Label mapping" |
| The two clipping numbers | Methods "Fixed gain"; Supplementary S13 and S24 |
| The softened claims | Abstract; first paragraph of each Discussion part; Conclusion |
