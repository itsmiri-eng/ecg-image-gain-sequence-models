# PTB-XL image dataset card — ptbxl_images_v1

**Status: complete — all 21,799 pages generated and fully verified on 24 September 2026.** Every source record has a saved PNG and provenance record. No model training is part of this generation task.

**Subsequent model experiments:** the separately versioned [PTB-XL image CNN–LSTM](ptbxl_cnn_lstm_report.md) and [small spatiotemporal network](ptbxl_spatiotemporal_report.md) have both been trained and verified on the eligible image cohort. Their [paired comparison](../results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.md) keeps inputs, patient folds and labels fixed. This card records dataset-generation evidence; its generation-stage test counts and preservation snapshot are historical. The generated pages, labels and patient folds are unchanged.

| Completion evidence | Current status |
|---|---|
| Training-only preview review | Passed for nine deterministic examples |
| Final generated image count | 21,799; no generation failures or reused pages |
| Actual total PNG bytes and generation runtime | 3,370,806,113 bytes (3.37 GB / 3.14 GiB); 488.02 seconds with four workers |
| Full image/manifest/provenance verification | Passed for all 21,799 pages; zero failures; 135.06 seconds |
| Final rendered duplicate/cross-partition collision check | Zero exact decoded-image duplicates, including across partitions |

The dataset overview is available as a [PNG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_images_v1/dataset_overview.png>) or [scalable SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_images_v1/dataset_overview.svg>). It summarizes the frozen source cohort and rendering protocol; full-cohort completion is established by the verification artifacts below.

## Source and purpose

This is a **custom clean image rendering of real PTB-XL waveforms**, created for a separately specified ECG image benchmark. It is not an existing PhysioNet paper-image release and is not a photograph, scan, or waveform synthesized from a generative model. The source is [PTB-XL version 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), whose release is listed under Creative Commons Attribution 4.0. Cite the [versioned dataset DOI](https://doi.org/10.13026/kfzx-aw45) and [original dataset publication](https://doi.org/10.1038/s41597-020-0495-6), and describe this project's rendering procedure as a derived representation.

The local release contains **21,799 ECG records from 18,869 patients**. The prior [source integrity audit](</Users/miriamistanboli/Downloads/ecg swiss/data/audit/ptbxl_inventory.json>) matched all 87,203 shipped-manifest files and found no same-rate exact waveform duplicates. Original WFDB files remain unchanged and are the authoritative numerical signals. The generator additionally checks each selected 500 Hz header/data file against its source checksum.

## Rendered representation

The [frozen configuration](</Users/miriamistanboli/Downloads/ecg swiss/experiments/configs/ptbxl_images_v1.json>) and [renderer](</Users/miriamistanboli/Downloads/ecg swiss/src/ptbxl_rendering.py>) define:

| Property | Specification |
|---|---|
| Source | Native `records500` waveform, decoded by WFDB into calibrated mV |
| Samples | 5,000 samples per lead at 500 Hz; 60,000 points per page |
| Layout | 12 vertically stacked, time-aligned rows; all 10 seconds of every lead |
| Lead order | I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6 |
| Time interval | Samples 0–4,999; times 0.000–9.998 s within the 10-second recording |
| Image | **2,240 × 3,024 pixels**, 8-bit grayscale PNG |
| Grid and scale | 8 pixels/mm; 25 mm/s; 1 mm and 5 mm grid spacing |
| Row geometry | 30 mm pitch; usable waveform band ±12 mm around the calibrated zero baseline |
| Gain | Nominal 10 mm/mV, reduced page-wide by powers of two when needed |
| Calibration | A 1 mV, 0.2-second pulse in each row at the actual displayed scale |
| Raster drawing | OpenCV antialiased lines, one-pixel waveform stroke, PNG compression level 6 |
| Printed content | Lead names and technical time/amplitude/source-rate calibration only |
| Augmentation | None |

Every native sample is supplied to its lead's polyline. There is no waveform filtering, sample removal, baseline recentering, per-lead amplitude rescaling, or duplicated rhythm strip. This custom full-duration layout keeps each lead on a single shared time axis.

Patient IDs, ECG IDs, dates, demographics, diagnostic labels, and SCP codes are **not printed in the pixels**. IDs remain in filenames and structured provenance to support correct linkage and patient grouping. Those metadata must not become model features.

## Gain and quality flags

For page maximum absolute amplitude `A` mV, the renderer uses `10 / 2**k` mm/mV, with the smallest nonnegative integer `k` satisfying `A × gain ≤ 12 mm`. All 12 leads share that gain. The actual gain, seconds/pixel, mV/pixel, calibration pulse geometry, and waveform coverage are recorded per page. Gain is also printed as technical calibration.

The [training-only geometry audit](</Users/miriamistanboli/Downloads/ecg swiss/experiments/protocols/ptbxl_images_v1/train_render_geometry_audit.json>) inspected all 17,418 training records. Record maximum absolute amplitude has median **1.905 mV**, 99th percentile **5.653 mV**, and maximum **32.716 mV**. Only 6.66% fit at nominal 10 mm/mV; most require 5 or 2.5 mm/mV. These statistics informed geometry before full generation, without inspecting test waveforms for style choices.

Flags are retained rather than used to silently remove records:

- `gain_below_2.5_mm_per_mv` records strong page-wide amplitude reduction.
- `flat_lead` records constant source channels; training ECG 12722 has flat V5.
- `calibration_pulse_below_2_pixels` records a calibration pulse too short for reliable raster detail.

Gain reduction prevents coordinate clipping, but it does not make rasterization clinically lossless. At 0.3125 mm/mV, for example, one mV spans only 2.5 pixels. At the nominal horizontal scale, 5,000 samples span 2,000 pixels: **2.5 samples per horizontal pixel**. Small or rapid waveform details may therefore merge. Preserve native waveforms and calibration metadata; the PNG cannot reconstruct every original sample exactly. Signal artifacts are preserved, not diagnosed or corrected.

The completed generation has **420 pages flagged for gain below 2.5 mm/mV** and **one page with a flat lead** (V5 in ECG 12722). These pages remain in their source partitions. No calibration pulse is below two raster pixels. Actual gain counts are:

| Display gain (mm/mV) | Pages |
|---|---:|
| 10 | 1,458 |
| 5 | 14,597 |
| 2.5 | 5,324 |
| 1.25 | 369 |
| 0.625 | 37 |
| 0.3125 | 14 |

## Labels and eligibility

The [label manifest builder](</Users/miriamistanboli/Downloads/ecg swiss/src/ptbxl_image_manifest.py>) maps all present `scp_codes` keys through local `scp_statements.csv` entries marked diagnostic. It takes the union of their diagnostic superclasses in the fixed order **NORM, MI, STTC, CD, HYP**. There is no confidence threshold; present codes with likelihood zero remain included. Original SCP dictionaries and confidence values are preserved separately.

| Binary target | Source records positive for target |
|---|---:|
| NORM | 9,514 |
| MI | 5,469 |
| STTC | 5,235 |
| CD | 4,898 |
| HYP | 2,649 |

These are **overlapping multilabel targets**, not mutually exclusive classes; 5,144 records have more than one mapped superclass. All 21,799 records are rendered. **21,388** have at least one mapped diagnostic superclass and enter the classification manifest. The other **411** retain their pages and original annotations but are marked `no_mapped_diagnostic_superclass` and excluded from supervised classification. They must not be interpreted as confidently all-negative targets or silently relabeled as normal.

No History_MI class, class priority, balancing, or infarction-stage relabeling is introduced. This five-target task is not directly comparable to the four mutually exclusive Mendeley folder classes used in the existing LDA, CNN, and spatial CNN+LSTM experiments.

## Fixed patient partitions

The original patient-respecting assignment is retained: folds 1–8 train, fold 9 validation, fold 10 test, following the [official PTB-XL split recommendation](https://physionet.org/content/ptb-xl/1.0.3/). There is no new random split or search for favorable class distributions. The manifest checks that each patient occupies one official fold; future variants of a record must inherit that fold.

| Partition | Official folds | Pages generated | Classification eligible | Excluded from classification | Patients, all / eligible |
|---|---|---:|---:|---:|---:|
| Train | 1–8 | 17,418 | 17,084 | 334 | 15,023 / 14,823 |
| Validation | 9 | 2,183 | 2,146 | 37 | 1,942 / 1,917 |
| Test | 10 | 2,198 | 2,158 | 40 | 1,904 / 1,877 |
| Total | 1–10 | 21,799 | 21,388 | 411 | 18,869 / 18,617 |

Exact metadata counts, target counts per partition, policies, and hashes are in [label_counts.json](</Users/miriamistanboli/Downloads/ecg swiss/experiments/protocols/ptbxl_images_v1/label_counts.json>). Technical generation/decoding of the fixed test images does not train a model or choose style from their appearance. Test pages are not part of the preview review.

## Artifacts and provenance

The dataset root is `/Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1`. Complete image, classification, split manifests and the passed verification report are saved here:

| Artifact | Meaning |
|---|---|
| [protocol.json](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/protocol.json>) | Frozen configuration, source-row binding, renderer/source hashes, runtime, and preview-review binding |
| [all_images.csv](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/all_images.csv>) | Every source ECG and rendered page, including classification-ineligible records |
| [classification_manifest.csv](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/classification_manifest.csv>) | Eligible multilabel records only |
| [classification_exclusions.csv](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/classification_exclusions.csv>) | Explicitly retained images excluded from classification, with reason |
| [train.csv](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/splits/train.csv>), [validation.csv](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/splits/validation.csv>), [test.csv](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/splits/test.csv>) | Eligible records under the official patient assignment |
| [generation_summary.json](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/generation_summary.json>) | Actual generated counts, encoded PNG bytes, gains, quality flags, manifest hashes, and runtime |
| [generation_state.json](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/generation_state.json>) | Current stage and progress |
| [verification.json](</Users/miriamistanboli/Downloads/ecg swiss/data/derived/ptbxl_images_v1/verification.json>) | Final verification outcome and any failures or rendered-image duplicates |

Each `images/{split}/ecg_XXXXX.png` has a corresponding `provenance/{split}/ecg_XXXXX.json`. The sidecar binds source record, source waveform/header hashes, configuration, encoded PNG hash, decoded-pixel hash, dimensions, actual gain, quality flags, and the complete sample interval of every lead. Image and provenance paths in CSVs are **relative to the dataset root**, not the working directory.

The archived renderer/source snapshot is indexed by [source_snapshot/sha256.json](</Users/miriamistanboli/Downloads/ecg swiss/experiments/protocols/ptbxl_images_v1/source_snapshot/sha256.json>). Preserve that snapshot with the generated protocol so later source edits do not erase the code version used for this cohort.

### CSV parsing and future model inputs

The five target columns are exactly `label_NORM`, `label_MI`, `label_STTC`, `label_CD`, and `label_HYP`. Targets and eligibility are serialized as **`True`/`False` strings** in CSV. Use an explicit boolean mapping or declared dtype; **`bool("False")` is true in Python and is incorrect here**. Reject unexpected values instead of guessing.

Images are the intended predictor inputs; only those five `label_*` columns are targets. Patient/ECG IDs, original SCP dictionaries, paths, split assignment, gain, and quality/provenance metadata are not additional predictor features. Keep gain metadata for physical interpretation and quality checks. The old four-class training runner is **not plug-compatible** with this dataset. No new training script or trained PTB-XL model is included in this generation scope.

## Quality assurance and completion criterion

The [preview review](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_images_v1/preview/preview_review.json>) passed for nine deterministic **training-only** examples, covering represented diagnostic superclasses, amplitude extremes, a near-99th-percentile example, and flat V5. It is a technical rendering review, not clinical validation or adjudication of diagnoses. The known large source artifact and flat channel remain visible and flagged.

Full verification accounted for all 21,799 PNGs and sidecars, decoded each image, matched encoded and decoded hashes, checked dimensions and source identity, reconstructed all six image/classification/split manifests, checked recorded lead/sample coverage, and found no exact rendered-pixel duplicates. Source header/data bytes were rechecked against the shipped checksums. `verification.json` is **`passed`** and `generation_state.json` is **`complete`**. These are technical integrity results, not clinical validation.

At completion of dataset generation, all **57 tests passed**. Production pages exactly matched all nine reviewed training previews. The [generation-stage preservation check](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_images_v1/preservation_verification.json>) recorded all 80 saved artifacts from the three then-completed model experiments, including their central log, as unchanged; the original 66-file baseline preservation check also passed. Seven source/configuration/dependency snapshots matched their originals. No model was trained or tested during that generation task. The later [PTB-XL model report](ptbxl_cnn_lstm_report.md) documents a separate training run, its appended log row, and current verification; it does not alter the generation snapshot.

## Reproduce generation and verification

Run from the existing project environment, which provides WFDB, NumPy, and OpenCV. These commands generate or validate data; they do not train a model:

```bash
cd "/Users/miriamistanboli/Downloads/ecg swiss"
/opt/anaconda3/bin/python -m src.generate_ptbxl_images generate --config experiments/configs/ptbxl_images_v1.json --output data/derived/ptbxl_images_v1 --workers 4
/opt/anaconda3/bin/python -m src.generate_ptbxl_images verify --config experiments/configs/ptbxl_images_v1.json --output data/derived/ptbxl_images_v1 --workers 4
```

Generation requires a matching frozen preview review and at least 8 GiB of free space. Work is streamed through bounded workers. A matching interrupted run can be resumed with the same generation command; existing pages are reused only after source/configuration/provenance checks. A changed renderer, configuration, metadata binding, or runtime must not silently replace an existing cohort. Use a new protocol/output version for intentional changes.

## Scientific limits and subsequent preprocessing

This cohort supports research on images generated by one controlled rendering style. It does not establish performance on real printed pages, camera photographs, different ECG-machine layouts, or clinical deployment. Source acquisition artifacts and annotation uncertainty remain. There is no random paper noise, rotation, scanning distortion, or other domain augmentation in this version.

Do **not** feed the entire 2,240×3,024 page through the old **128×256 whole-page resize/crop**. That configuration was built for a different four-class image source; it would alter this page's aspect ratio and severely compress twelve full lead traces and calibration. Any later preprocessing must be separately specified and validated on development data to preserve needed lead coverage, time/amplitude interpretation, and waveform detail. Native images, patient splits, and rendering provenance should remain immutable. Model comparison requires an explicitly multilabel protocol and does not inherit the old four-class metrics or conclusions.
