# Dataset audit resumed 2026-11-01

The current frozen data pass the full read-only audit. No data-stage defect was found that invalidates the five completed runs as evaluations of their declared image pipelines. This is technical integrity evidence, not clinical validation or proof that test results never influenced later research choices. The machine-readable results are in `dataset_audit_resume.json` in this directory. No model fitting, inference, label changes, or prior artifact overwriting occurred.

## Fresh checks

- Reconstructed PTB-XL metadata, five-target labels, eligibility and official patient folds from the shipped, checksum-bound metadata. Rechecked all 21,799 generated PNGs and sidecars, decoded-pixel and encoded hashes, all 500 Hz source `.dat`/`.hea` bytes, recorded complete lead/sample coverage, and file inventories. All six saved image/classification/split manifests reconstructed byte-for-byte.
- Decoded and hashed all 21,388 cached learning PNGs. No duplicate source-page or cached-input pixel hashes, and no patient, ECG, encoded-image, decoded-image or cached-input overlap across partitions. All source/cache binding and runtime checks passed.
- Recomputed normalization from **only 17,084 training images / 27,553,075,200 pixels**: mean **0.032673040408934095**, standard deviation **0.09723822329212023**, exactly matching the saved statistics. Independently reapplied preprocessing to three deterministically selected training pages and obtained exact cached pixels. This is not a full independent rerender of every source waveform.
- Rechecked all 928 Mendeley raw-file hashes, the 491 retained original/cache entries, fixed split/input hashes, and duplicate/group separation. All three completed Mendeley runs saved exactly the same split file. CNN and CNN–LSTM normalization both equal freshly recomputed training-only statistics.
- Rejoined Mendeley header IDs: 311 printed-ID proxies; zero IDs cross partitions; 62 groups span folder labels, containing 220 of 491 retained records. This verifies grouping, not patient identity or clinical label correctness.
- The 13 protected PTB-XL manifest/protocol/cache metadata files remained byte-identical through the audit. The full audit took 239.4 seconds. Fresh 100 Hz source byte hashing was not repeated; those alternative representations are unused by the image models and retain historical full-checksum evidence plus freshly checked path coverage.

## Cohorts and labels

| Cohort | Train | Validation | Test | Label task |
|---|---:|---:|---:|---|
| Mendeley deduplicated images | 295 | 98 | 98 | Four mutually exclusive supplied folder labels |
| PTB-XL eligible rendered ECGs | 17,084 | 2,146 | 2,158 | Five overlapping diagnostic-superclass targets |
| PTB-XL eligible patients | 14,823 | 1,917 | 1,877 | One official fold per patient |

Mendeley: 928 filenames collapse to 491 unique pages after 437 exact redundant copies. Retained counts are Normal 142, MI 30, Abnormal 233, History_MI 86. Train/validation/test MI supports are **18/6/6**; one test prediction changes MI recall by 16.67 percentage points. Printed report IDs are unverified patient proxies; clinical adjudication remains empty (`results/readiness/labels/summary.json`: `verified_clinical_labels=0`). The zero-link near-duplicate screen is conservative and does not establish absence of transformed copies.

PTB-XL: all 21,799 source records are rendered; 411 without a mapped diagnostic superclass are explicitly excluded from supervised classification, leaving 21,388 ECGs from 18,617 patients. All present diagnostic SCP keys map through `diagnostic==1`, without a confidence threshold; this agrees with the included `example_physionet.py`. Zero-confidence inclusion is a declared mapping rule, not a parsing accident. All five columns use strict `True`/`False` parsing. There are 5,144 multilabel records; 445 eligible records include NORM plus another target, so converting this task to single-label classification would alter the ground truth.

| PTB-XL positive labels | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| Train | 7,596 | 4,379 | 4,186 | 3,907 | 2,119 |
| Validation | 955 | 540 | 528 | 495 | 268 |
| Test | 963 | 550 | 521 | 496 | 262 |

Training prevalence ranges from 44.46% for NORM to 12.40% for HYP. Both completed PTB-XL models use unweighted BCE and a fixed 0.5 threshold; class-specific/macro metrics are therefore essential. This imbalance is a scientific limitation, not dataset corruption. Mendeley LDA uses prespecified uniform priors and training-only PCA; CNN/CNN–LSTM use training-only mean/std and `n_train/(4*class_count)` weights. Source code: `src/models/lda.py:27`, `src/models/cnn.py:326`, `src/models/cnn_lstm.py:183`, `src/ptbxl_learning_data.py:224`.

## Limitations and defensible claims

- The old claim of **928 independent observations** is invalidated by exact duplicates. The current deduplicated experiments are different experiments; old manuscript scores cannot be silently compared with or substituted for them.
- Mendeley results support exploratory comparison of the frozen four-class image pipelines. They do not establish patient-independent diagnostic accuracy. Cross-label IDs are a provenance concern, not proof that a particular supplied label is wrong. The verified crop truncates tall waveform peaks in one inspected training page, and resizing changes relative aspect ratio by 15.22% (`docs/input_representation_review.md:25`). This limits physiological interpretation but does not invalidate reproducible comparisons using the same fixed inputs.
- Mendeley CNN versus CNN–LSTM changes checkpoint selection as well as architecture, so it is a **pipeline comparison**, not an isolated LSTM ablation (`experiments/configs/cnn_lstm_image_v1.json:28`). Its previously observed test set supports exploratory interpretation.
- PTB-XL CNN–LSTM versus spatiotemporal uses the same eligible records, patient folds, cached pixels, training normalization, optimizer budget and fixed threshold. Their same-cohort comparison is defensible, subject to model/probability verification and the declared single-seed, previously observed-test, head-capacity and dropout limitations. Neither is clinical deployment or external-domain validation.
- The PTB-XL renderer uses one clean synthetic style. All 12 full-length aligned lead rows and calibration pulses remain in the model input, but rasterization/max pooling lose morphology. Across all rendered pages, 420 require gain below 2.5 mm/mV; extreme low-gain signals retain very few pixels per mV. Train-only geometry review and per-record deterministic gain adaptation do not fit test-distribution statistics. Calibration graphics are retained, while numeric metadata/IDs/labels are not predictor features.
- A direct accuracy ranking across Mendeley and PTB-XL is invalid: their cohorts, labels, sample sizes, image geometry and metric semantics differ. An image-feature sequence is not recovered native ECG samples; the Mendeley sequence is not validated physiological time.

## Verification command hazard

`src.generate_ptbxl_images.verify()` writes a new timestamped `data/derived/ptbxl_images_v1/verification.json` (`src/generate_ptbxl_images.py:303`). The learning cache and run protocols bind that file's exact SHA-256. Rerunning the old documented generator `verify` CLI can therefore invalidate downstream bindings even when every image is intact. This audit called only its read-only `_verify_job`, `_binding` and `_csv_bytes` helpers and saved a separate new report. Future rechecks should use the new standalone audit command, with a new output path; never regenerate or overwrite frozen verification records merely to recheck them.
