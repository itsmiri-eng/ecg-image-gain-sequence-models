# External confirmatory test: analysis plan (written 3 November 2026, FROZEN 5 November 2026)

Status: **frozen on 5 Nov 2026.** Its hash, and the hashes of the label mapping, are in
`experiments/external/plan_freeze.json`. The freeze came after the label-mapping coverage check (§3) and
**before** any Georgia, Chapman-Shaoxing or Ningbo header or signal was parsed, rendered or shown to a model.
From here on, any change is an amendment: dated, justified, and reported.

**Mapping outcome, recorded at freeze.** The rule was applied unchanged; see
`experiments/external/label_mapping.csv`. 25 of the 50 SNOMED codes were assigned.
- **NORM** has no code: the organisers coded PTB-XL NORM as sinus rhythm (426783006, NORM share 0.53). NORM is
  therefore excluded from the external macro AUROC.
- **Near misses.** T-wave abnormal (164934002, STTC share 0.88), T inversion (59931005), ST depression
  (429622005) and abnormal Q wave (164951009, MI 0.81) fall below 90% and give no label. This lowers
  STTC/MI sensitivity of the external labels; positives are conservative.
- **Two-class code.** RVH (89792004) maps to both CD (0.94) and HYP (1.00).

Why: PTB-XL fold 10 was observed before the models were specified, so every PTB-XL test result is
exploratory (`docs/ROADMAP_Q1.md`, weakness #1). An untouched dataset from another country, hospital and
ECG machine gives the confirmatory evidence a Q1 journal expects.

## 1. Data (free, CC BY 4.0, open access on PhysioNet)

| Role | Source | Records | Country | Approx. download |
|---|---|---|---|---|
| **Primary** | PhysioNet/CinC Challenge 2021 v1.0.3, `training/georgia` (G12EC) | 10,344 | USA (Emory, Atlanta) | ≈1.2 GB (≈117 KB/record) |
| Secondary | same, `training/chapman_shaoxing` + `training/ningbo` | 45,152 | China | ≈5.3 GB |
| Label mapping | PTB-XL+ v1.0.1, `labels/mapping/ptbxlToSNOMED.csv` | – | – | 13 KB |

The `ptb`, `ptb-xl`, `cpsc_2018*` and `st_petersburg_incart` folders are **not** used. PTB and PTB-XL overlap
our training data, CPSC records vary from 6 to 144 s, and INCART has 74 thirty-minute records at 257 Hz.
Downloads go to a new folder `data/external/` with SHA256SUMS checks. They are kept apart from
`data/raw`/`data/processed`, and no existing file is changed.

## 2. Inclusion (decided now, applied from headers only)

1. All 12 standard leads present, 500 Hz, finite values, millivolt calibration after the header gain.
2. Length ≥ 10 s. The first 10 s (5,000 samples) are used, the same as PTB-XL. Shorter records are excluded
   and counted.
3. At least one diagnosis maps to one of the five superclasses (§3), the same rule that defined the PTB-XL
   cohort (21,388 of 21,799).
4. No other exclusion. Records with a flat lead or clipping are kept and flagged.

**Amendment, 6 Nov 2026** (after the Georgia inventory, before any Chapman-Shaoxing or Ningbo header was parsed).
Two Ningbo signal files (`training/ningbo/g3/JS13118.mat`, `JS13243.mat`) never match PhysioNet's published
SHA-256. They failed on three separate download attempts, so the served files differ from the listed hashes.
Unverified files are not kept. These 2 records (of 34,905 Ningbo records) are excluded as
`signal_file_failed_checksum`, counted before every other inclusion step. Georgia is unaffected.

## 3. Labels: no hand-made mapping

A record's SNOMED-CT code maps to a PTB-XL SCP statement through the published PTB-XL+ mapping
(`ptbxlToSNOMED.csv`, Strodthoff et al., Sci Data 2023), used **in reverse**. The statement then maps to its
superclass through PTB-XL's own `scp_statements.csv` (`diagnostic_class`).

- **Ambiguous codes.** A SNOMED code that maps to statements in two different superclasses is assigned to
  both. Every such code is listed in the freeze file.
- **Rhythm and form statements** (for example 426783006 sinus rhythm) have no diagnostic class in PTB-XL, so
  they give no label, exactly as in PTB-XL.
- **NORM** comes only from codes that map to the PTB-XL statement NORM. Sinus rhythm alone is **not** NORM.
- **Small labels.** Label counts are computed from the headers before any model is run. A label with fewer
  than 50 positives or 50 negatives in a dataset is excluded from that dataset's macro AUROC, reported
  per label only, and named in the results. The expected case is NORM in Georgia.

Only the mapping's coverage (which codes map, and to what) is inspected before freezing. Prevalences are
inspected only to apply the rule above.

**Coverage check result (3 Nov 2026, before any external ECG was opened).** `ptbxlToSNOMED.csv` was downloaded
(13,367 bytes, checksum verified). It cannot be used as planned. Its IDs are OMOP concept IDs (for example 4065279
"ECG normal"), not the SNOMED CT codes used in the Challenge 2021 headers (for example 426783006). Translating
between the two needs the OHDSI Athena vocabulary, which requires an account. **Proposed replacement** (still
published and not hand-made):

- Use the Challenge 2021 organisers' own conversion of PTB-XL to SNOMED CT, from the headers in
  `training/ptb-xl/*.hea`. These are labels only, about 21,837 small text files, ≈15 MB.
- Pair each header (record HRxxxxx = PTB-XL ecg_id xxxxx) with that record's PTB-XL SCP statements.
- A SNOMED code is assigned to a superclass when, among the PTB-XL records carrying it, at least 90% carry a
  statement of that superclass. The threshold is fixed now.
  - Only codes seen on at least 10 PTB-XL records are assigned.
  - Codes with no superclass reaching 90% get no label.
  - The full table, with its percentages, goes into the freeze file.
- **Alignment check.** Header age and sex must agree with `ptbxl_database.csv` for at least 99% of records.
  Otherwise the mapping is abandoned.
- Script: `src/external_label_mapping.py`. The header download was approved by the user on 3 Nov 2026.
- **Consequence.** If the organisers mapped NORM to the sinus-rhythm code, NORM gets no external label. It is
  then excluded from the external macro AUROC under the small-label rule above, and that exclusion is reported.

## 4. Inputs

- Pages are drawn with the **unchanged** renderer functions:
  - adaptive gain: `ptbxl_rendering.render_page`;
  - fixed gain: `ptbxl_fixed_gain.render_fixed_gain`.
- They are drawn from the 10-s, 12-lead mV array, with the same config (`experiments/configs/ptbxl_images_v1.json`)
  and the same `preprocess_page`.
- A new loader reads the challenge `.hea/.mat` files. The frozen `load_waveform` is not edited.
- Each network sees the page type it was trained on.
- Normalization is the PTB-XL training-fold statistics. There is no recalibration, fine-tuning or threshold change.
- **Signal reference (secondary).** It needs 100 Hz input. External signals are resampled 500 → 100 Hz with
  `scipy.signal.resample_poly(x, 1, 5)`. This may differ from how PTB-XL's `records100` were made, so the
  signal-reference comparison is labelled secondary.

## 5. Models and inference

- Every completed run is included:
  - all seeds of CNN-only, wide CNN-only, CNN-LSTM, temporal conv and hybrid ViT, with adaptive and fixed gain;
  - PCA + LDA and PCA + SVM;
  - the signal ResNet.
- There is **one inference per run per external dataset**, guarded by a state file, as for fold 10.
- No model, hyperparameter or threshold is chosen using external data.

## 6. Hypotheses and statistics (primary dataset: Georgia)

Bootstrap: records are resampled, since Georgia has no patient IDs. 2,000 resamples, seed 2024. Seed-level
uncertainty uses the hierarchical bootstrap (seeds, then records) of `docs/protocol_amendment_2026-11-03.md`.
Holm correction is applied within each hypothesis family, α = 0.05.

- **H1 (head comparison replicates).** These are the pairwise seed-mean macro AUROC differences among the five
  image heads on adaptive-gain pages: 10 pairs. Prediction from PTB-XL is ViT ≥ temporal conv ≈ CNN-LSTM >
  wide CNN-only ≥ CNN-only > LDA. We report which orderings are confirmed.
- **H2 (fixed gain).** For each network, the fixed-gain run is compared with the adaptive run of the same
  architecture, on HYP AUROC and on macro AUROC: 4 architectures × 2 metrics. Prediction: fixed gain is not
  worse for any network, and is better for CNN-only and temporal conv.
- **H3 (generalization gap).** For each network, external minus PTB-XL fold-10 macro AUROC, descriptive only.
  This shows how much of each model's performance transfers.
- **Secondary.** The same analyses are run on Chapman-Shaoxing + Ningbo. Other secondary analyses are
  calibration (ECE, Brier), the validation-chosen thresholds, and the signal reference.

## 7. Reporting

- All runs and all seeds are reported, including failures and exclusions, with counts at each inclusion step.
- If a result contradicts the PTB-XL results, it is reported as such.
- The plan's SHA-256 and the freeze time are stated in the paper.
