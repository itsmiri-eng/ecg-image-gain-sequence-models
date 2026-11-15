# Protocol amendment, 15 November 2026: label-mapping documentation and sensitivity (Additional analysis)

Written and frozen on 15 Nov 2026 **before** any excluded external record was rendered or given to a model and
before any sensitivity metric was computed. Known at the time of writing, from the saved inclusion tables only
(no model output): which SNOMED CT codes the excluded records carry. Nothing below changes the frozen
confirmatory analysis (`docs/external_test_plan.md`), its cohorts, its predictions or its mapping.

**Motivation.** The supervisor's review (15 Nov 2026) asked for the clinical name of every SNOMED CT code, a
validation of the code-to-diagnosis mapping, and an analysis of how the mapping affects the results, because
the mapping excluded 41% of the Georgia records (4,245 of 10,344) and 80% of the Chinese records (36,146 of
45,152).

## A. Clinical names and validation of the mapping

- Every code in `experiments/external/label_mapping.csv` gets its clinical name from the Challenge organisers'
  own code tables (`dx_mapping_scored.csv`, `dx_mapping_unscored.csv`, repository
  physionetchallenges/evaluation-2021), cross-checked against the SNOMED CT browser where reachable. The table
  is saved as `experiments/external/snomed_code_names.csv` with the source of every name.
- **No cardiologist reviewed the mapping.** This is stated in the paper. The validation that can be done
  without an expert is: (i) names from two independent sources agree; (ii) each assigned code's clinical meaning
  is compared with the definition of its PTB-XL superclass and every mismatch is listed; (iii) the data-driven
  shares are reported for every code.

## B. Who was excluded (descriptive, from the inclusion tables)

Per cohort: the codes carried by excluded and included records (count and share, with names), the number of
excluded records carrying only rhythm statements, label prevalence in the included cohort next to PTB-XL test,
and the positives contributed by each assigned code.

## C. Sensitivity of the results to the mapping

Variants (per cohort; labels eligible if ≥ 50 positives and ≥ 50 negatives, as in the frozen plan):

| Variant | Records | Mapping |
|---|---|---|
| V0 (published) | frozen cohort | 90% rule |
| V1 strict | frozen cohort; records left without a label are dropped | share ≥ 0.99 |
| V2 relaxed, same records | frozen cohort | share ≥ 0.80 (labels added) |
| V3 all records | frozen cohort + every record excluded only for "no mapped superclass", as negatives for all labels | 90% rule |
| V4 all records, relaxed | as V3 | share ≥ 0.80 |

- V0–V2 use the saved predictions only. V3–V4 need predictions for the unmapped records: these records are
  rendered with the unchanged adaptive and fixed-gain renderers into a **separate** cache
  (`data/external/cache_unmapped/<cohort>/`), every completed run is reloaded and must reproduce its saved PTB-XL
  validation probabilities, and each run gets exactly one inference on them
  (`results/external_unmapped/<cohort>/<run>/`). The frozen predictions of the included records are reused, never
  recomputed.
- Reported per variant: records, positives and prevalence per label, eligible labels, mean external macro AUROC
  and per-label AUROC per model, and the H1 and H2 differences. For V3 the H1 and H2 differences get the
  hierarchical bootstrap of the statistical revision (10,000 resamples, Holm within family); the other variants
  are reported as point estimates.
- **Interpretation rule.** A conclusion is called robust to the mapping if the sign of the difference is the same
  in every variant and its Holm-adjusted p stays below 0.05 in V3. Any change of sign or loss of support is
  reported in the main text. No causal statement about why performance differs between cohorts or variants is
  made from these analyses; they show association with the cohort definition only.
- **Known limits.** In V3/V4 unmapped records are counted as negatives although some carry statements close to a
  superclass (for example T-wave abnormality, STTC share 0.88 in PTB-XL), so V3 has label noise in the
  negatives; V4 reduces it for the three near-miss codes. Normal ECGs still cannot be scored as NORM, because
  the organisers have no code that maps to it. The mapping remains data-driven.

## Execution

Rendering on at most 3 CPU workers at low priority; inference on the GPU after the fixed-gain runs of the same
day have finished. Interrupted steps are restarted; an inference that has started for a run is never repeated.

## Addendum, 15 November 2026 (written before any sensitivity metric or any model output on the unmapped records existed)

While collecting the clinical names (part A), the organisers' code tables showed that they score some code pairs
as the same diagnosis: right bundle branch block 59118001 = complete right bundle branch block 713427006, and
complete left bundle branch block 733534002 = left bundle branch block 164909002. The first code of each pair
never occurs in the organisers' PTB-XL encoding, so the data-driven rule could not assign it, although its
equivalent is assigned to CD. One more variant is therefore added:

| Variant | Records | Mapping |
|---|---|---|
| V5 all records, equivalents | as V3 | 90% rule, and 59118001 and 733534002 inherit the assignment (CD) of their organiser-declared equivalents |

V5 is reported as point estimates, like V1, V2 and V4. No other code is added by judgement: for example
"left ventricular high voltage" (55827005, China only, absent from PTB-XL) stays unmapped in every variant and is
named as a limitation, because assigning it to HYP would be our own clinical decision without expert review.
