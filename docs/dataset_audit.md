# Dataset audit and frozen baseline split

Audited 2026-09-23. The baseline cohort contains **491 unique ECG images**, reduced from 928 source files by excluding 437 redundant exact copies from the experiment manifest. The resulting class distribution is strongly imbalanced: only 30 unique MI images remain. All raw files and the source ZIP remain intact.

The frozen dataset version is `image_dedup_8e04bd56af2f`; the split version is `grouped_cf5a867a99c3`. Both LDA and CNN must consume the same saved split and input cache. These versions describe new baseline experiments, not reproduction of the earlier manuscript's 928-image results.

## Source inventory and integrity

All 928 included source files are JPEG images, decoded as RGB at **2213 pixels wide by 1572 pixels high**. Every enumerated source path was readable and decoded successfully: zero missing paths and zero corrupt images. This verifies the enumerated local files; it does not establish that the original clinical cohort was complete.

The 928 source image entries in `mendely images.zip` match the 928 local cohort files byte-for-byte. There are no missing archive images, byte mismatches or additional local cohort images. A hidden notebook checkpoint image, `ecg files/MI/.ipynb_checkpoints/MI(1)-checkpoint.jpg`, was recognized as an editing artifact and excluded from source enumeration. It was not deleted.

| Folder label | Source files | Unique retained images | Redundant copies excluded | Share of retained cohort |
|---|---:|---:|---:|---:|
| Normal | 284 | 142 | 142 | 28.92% |
| MI | 239 | 30 | 209 | 6.11% |
| Abnormal | 233 | 233 | 0 | 47.45% |
| History_MI | 172 | 86 | 86 | 17.52% |
| **Total** | **928** | **491** | **437** | **100.00%** |

`History_MI` is the analysis label for the existing folder `HIstory of MI`. Labels remain the supplied folder categories; no diagnosis was inferred from an image or changed to improve class balance. In the deduplicated cohort the largest class has 7.77 times as many images as MI. The raw counts concealed this imbalance because duplication rates differed substantially by class.

## Exact duplicates and exclusions

The audit computed a SHA-256 digest of decoded RGB pixel data and dimensions, alongside a file-byte SHA-256 digest. Files with identical decoded pixels were placed in the same exact-copy group. For each group with one folder label, the lexicographically first source path was retained as its reproducible representative. Every excluded path and its retained `canonical_path` is recorded in `data/audit/exclusions.csv`.

The 437 excluded records represent **47.09% of source files**. There were no exact decoded-pixel groups spanning different class labels and no exclusions for unresolved printed IDs. Exclusion means that a redundant record does not enter the processed dataset or split; no source image was physically removed. Different images sharing a printed ID were retained and grouped together rather than deleted.

## Printed report identifiers and related records

The images contain printed six-digit report identifiers. The audit recovered identifiers for all 491 retained images, yielding **311 distinct printed IDs**. Of these IDs, **82 occur on multiple unique images**, **62 span more than one folder class**, and the largest ID group contains **13 unique images**.

These are **report-ID proxies, not verified patient IDs**. Their relationship to a unique patient, examination, encounter or acquisition system has not been established. Keeping equal IDs together prevents a documented source of related-record overlap. It does not prove complete patient independence: one patient could have multiple printed IDs, or an ID could be reused.

The 62 IDs spanning folder categories raise a label-provenance question. They could reflect related examinations, diagnostic changes, reused identifiers or inconsistent source labeling. The audit does not determine which explanation applies and does not relabel these records. Grouping across classes conservatively prevents the same recognized identifier from crossing partitions. Clinical review and original source metadata are needed before treating the four categories as a definitive clinical ground truth.

Identifier recovery used the fixed header region `[125, 65, 245, 100]` in source pixels. Dark connected components were segmented and matched to ten visually labeled digit templates derived from five reference pages (30 reference glyphs). Each accepted digit had zero mean squared error against its template after normalization, with a competing-digit separation of at least 0.09194. All 491 headers met the exact-template criterion.

Independent visual transcription agreed on all 100 primary validation rows, including 99 that were not reference pages. An additional 20-image pilot agreed on all 20; across the two checks there were 120 validation rows covering 115 unique source hashes. These checks support transcription accuracy on this uniform printed font. They are not a calibrated OCR confidence probability or validation of patient identity. Original ID values are kept in the local audit table; the public-facing split uses reproducible opaque group IDs.

## Near-duplicate screening

A separate similarity screen examined 491 unique file images. It considered 1,508 candidate pairs with 63-bit non-DC DCT perceptual-hash distance at most six. A conservative related-image link required crop Pearson correlation at least 0.995 and mean absolute ink difference at most 0.005. **No additional grouping links met both criteria.**

Candidate pairs and the screening summary are retained for inspection. This screen can miss shifted, rescaled or redrawn copies and does not establish that all similar ECG traces were found. No near-duplicate image was removed, and no model performance informed the thresholds or grouping decisions.

## Fixed train validation test split

Groups are connected components linked by equal printed report IDs, identical processed-input hashes and any accepted near-copy links. The final cohort has 311 groups. The split uses seed **42**, the prespecified first fold of shuffled five-fold stratified group splitting for test, then the prespecified first fold of shuffled four-fold stratified group splitting within the development set for validation. Target proportions were 60%/20%/20%. The split was selected before model results and was not searched for favorable performance.

| Folder label | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Normal | 85 | 29 | 28 | 142 |
| MI | 18 | 6 | 6 | 30 |
| Abnormal | 140 | 46 | 47 | 233 |
| History_MI | 52 | 17 | 17 | 86 |
| **Images** | **295** | **98** | **98** | **491** |
| **Groups** | **187** | **63** | **61** | **311** |

Independent inspection of `data/splits/fixed_split.csv` confirmed zero overlap between each pair of partitions for group IDs, source file hashes, decoded-pixel hashes and processed-input hashes. The grouping construction also keeps equal recognized report IDs in a single partition. These are the leakage checks supported by this audit; unrecognized patient relationships and source-specific visual confounding remain possible.

Only **six MI examples** are in the test set and six in validation. One different MI prediction changes that class's test recall by 16.67 percentage points. Class-level estimates and model differences consequently have substantial uncertainty. Accuracy alone would obscure minority-class failure; balanced accuracy, macro F1 and per-class recall must accompany it. No oversampling or synthetic samples were added to the frozen validation/test sets.

## Common image representation

The deterministic common input pipeline crops the image using fractional `(left, top, right, bottom)` coordinates **`[0.03, 0.19, 0.98, 0.96]`**, converts to grayscale, resizes with bilinear interpolation to **256 pixels wide by 128 pixels high**, and inverts/rescales intensity into `[0, 1]`. Each input has shape **`[1, 128, 256]`**. The crop omits the printed header used for grouping; it is fixed rather than learned from outcomes.

This is an image representation, not calibrated ECG time series. Resizing reduces fine detail and the crop may discard peripheral content; these are baseline limitations. The representation remains subject to grid, layout and acquisition confounds. No unvalidated SNR threshold was used to exclude additional images. Model-specific fitted normalization and dimensionality reduction must use training images only; their exact fitted settings belong in each saved experiment.

## Reproducible evidence files

- `data/audit/source_manifest.csv`: all included raw paths, folder labels, dimensions, file hashes and pixel hashes.
- `data/audit/exclusions.csv`: every excluded exact-copy path and its canonical representative.
- `data/audit/archive_provenance.json`: archive/local byte-match verification.
- `data/audit/header_ids.csv`, `header_template_summary.json`, `header_template_validation.csv` and `header_template_pilot_summary.json`: identifier recovery and visual-validation evidence.
- `data/audit/near_duplicate_summary.json`, `near_duplicate_candidates.csv` and `near_duplicate_links.csv`: similarity-screen method, candidates and accepted grouping links.
- `data/audit/dataset_audit.json`: machine-readable counts, policies and hashes summarized here.
- `data/splits/fixed_split.csv`: the exact 491-sample assignment consumed by both models.
- `data/splits/protocol.json`: frozen configuration and audit snapshot.
- `data/processed/inputs_v1.npz`: common processed-image cache, with no raw dataset duplication.
- `experiments/configs/baseline_v1.json`: the declared baseline protocol.

Split SHA-256: `d000c06a280c59bf60b883abbc0711a575b46e69843bb01c8593befde37d9616`.

Input-cache SHA-256: `ae7a5b80d1d8da2d371568a21f1547dd64f26e5d95b7657ec3b738a85f19f60e`.

The separate PTB-XL waveform collection is documented in `docs/ptbxl_secondary_audit.md`; it was not mixed into this image cohort. Earlier 928-image manuscript results use a different effective cohort and split and must not be presented as directly comparable to these deduplicated, grouped baseline experiments.
