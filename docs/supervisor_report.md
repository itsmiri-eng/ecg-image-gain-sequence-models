# ECG research progress — 23 September 2026

**Update, 24 September 2026:** the third model, CNN–LSTM over spatial image features, has now also been executed. Test accuracy is **71.43%**, balanced accuracy **63.15%**, macro F1 **0.6020** and weighted F1 **0.7168**. It recognises more MI and History_MI examples than the original CNN, with lower overall accuracy. The validation-selection rule also changed, so this is a complete-pipeline comparison rather than an isolated LSTM effect. See [CNN–LSTM report](cnn_lstm_report.md), [three-model figure](../results/comparisons/cnn_lstm_image_v1_seed42/comparison.png), and [PTB-XL assessment](ptbxl_migration_assessment.md). The original two-model report below is preserved as the earlier baseline record.

**Two baseline experiments were actually executed and saved: PCA + LDA and a small 2D CNN.** Both use one audited image dataset and one fixed grouped split. These are new reproducible baselines; they do not reproduce or validate the old manuscript's reported 94.62% result.

## Main dataset finding

The 928 source ECG JPEGs contain **437 redundant exact copies**, leaving **491 unique pages**. All originals are preserved; the experiment manifest excludes redundant copies. Every source JPEG matches the retained original archive and decodes correctly at 2213×1572 RGB.

| Class | Original files | Unique images | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Normal | 284 | 142 | 85 | 29 | 28 |
| MI | 239 | 30 | 18 | 6 | 6 |
| Abnormal | 233 | 233 | 140 | 46 | 47 |
| History_MI | 172 | 86 | 52 | 17 | 17 |
| **Total** | **928** | **491** | **295** | **98** | **98** |

The duplicates concealed substantial imbalance: only 30 unique MI pages remain. A random filename split could put identical ECGs in training and testing, so historical results from that procedure cannot establish independent-record performance.

Printed headers yield **311 distinct IDs**, transcribed with exact glyph matching and visual validation. All pages sharing an ID stay in one partition; 82 IDs have multiple unique pages, and 62 occur in more than one class folder. Those IDs are unverified report/patient identifiers. Grouping prevents known ID overlap, but **patient independence and the folder-label clinical definitions remain unverified**. A limited near-copy screen found no additional high-confidence links; it does not exclude every possible related acquisition.

The split was fixed with seed 42 before either model ran: 187/63/61 identity groups in training/validation/test, with no shared group or exact image/input hash across partitions. Dataset version: `image_dedup_8e04bd56af2f`; split version: `grouped_cf5a867a99c3`.

## Executed results

| Model | Test accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| PCA + LDA | **62.24%** (61/98) | **60.32%** | **0.5637** | **0.6423** | **0.8771** |
| Small CNN | **75.51%** (74/98) | **56.86%** | **0.5621** | **0.7310** | **0.9008** |

The CNN improves overall accuracy but does not improve class-balanced performance. Its Normal and Abnormal recalls increase from 75.0%/57.4% to 100.0%/87.2%, while MI recall falls from 3/6 to 1/6 and History_MI from 10/17 to 4/17. Only six MI test images are available, so that sensitivity estimate is especially uncertain. **Do not describe the CNN as generally superior.**

LDA uses 32 training-only principal components and shrinkage covariance estimation with uniform class priors. The CNN has 40,212 parameters, training-only normalization, class weighting, dropout, and validation-based checkpoint selection. Both receive the same grayscale 128×256 trace-region crop; headers/footers are removed. These models classify images, not validated/calibrated ECG waveforms.

LDA's train-to-test accuracy gap is about ten percentage points. CNN training and validation curves show some late overfitting; epoch 25 was restored after all 30 epochs completed. The early-stopping mechanism was enabled but its seven-epoch patience was not exhausted. No tuning used test results. Recorded total runtimes were approximately **1.2 seconds for LDA and 46.5 seconds for CNN on CPU**, excluding dataset auditing, preprocessing and process startup.

## What changed and what is saved

The recovered script was a 1D ResNet over unvalidated traces estimated from whole-page tiling. It had no saved validation split, duplicate protection, early stopping or checkpoint. Its architecture differs from the later manuscript's architecture. Original code, documents, figures and datasets were preserved.

New reusable Python modules provide auditing, fixed preprocessing, grouped splitting, LDA/CNN fitting, evaluation and comparison. Saved evidence includes source hashes, every duplicate exclusion, split membership, fitted PCA/LDA, the CNN checkpoint and architecture, 30-epoch history, train/validation/test predictions and metrics, confusion matrices, runtime/environment/configuration records, and a central experiment CSV. Six focused tests pass, and saved prediction metrics match the comparison table.

The existing PTB-XL collection was separately audited: 21,799 records, 18,869 patients, all 87,203 checksum entries valid, no patient overlap across official folds, and one flagged constant-V5 record. It was not mixed into the image experiments.

## Materials to show tomorrow

1. This report and `dataset_audit.md`, especially the duplicate and split table.
2. `../results/comparisons/dataset_counts.png` and `comparison.png`.
3. Both saved test confusion matrices and `../results/comparisons/per_class_comparison.csv`.
4. `../models/cnn/cnn_v1_seed42/training_curve.png` and the saved experiment log.
5. `../README.md` for exact environment and reproduction commands.

The next requested progression step is to **confirm what LSM means** and obtain the exact name/reference for the remembered convolution approach. Those placeholders remain unimplemented. Before CNN+LSTM, validate lead/time extraction and minority-class labels, seek more independent MI data, and predefine further validation without repeatedly tuning against this now-observed test set. ViT is deferred.
