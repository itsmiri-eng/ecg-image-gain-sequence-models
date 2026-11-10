# PTB-XL image CNN+LSTM experiment report

**Completed and verified run: `ptbxl_cnn_lstm_image_v1_seed42`.** Held-out test macro ROC-AUC is **0.8698** (95% patient-bootstrap interval **0.8593–0.8805**). The checkpoint was selected at epoch **15 of 19 completed epochs** using validation data only.

This report is derived from saved predictions, metrics, training history, and provenance. It performs no model inference, refitting, threshold selection, or new bootstrap sampling.

## Cohort and model

The source is the custom clean rendering of real native 500 Hz, 10-second, 12-lead [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) signals. Use the [versioned dataset DOI](https://doi.org/10.13026/kfzx-aw45) and [original publication](https://doi.org/10.1038/s41597-020-0495-6), with attribution to this derived image-rendering procedure. The [dataset card](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_image_dataset.md>) documents source licensing (CC BY 4.0), labels, rendering and exclusions.

All 21,799 records have rendered pages; 21,388 have at least one of the five mapped diagnostic superclasses and enter classification. The 411 unmapped records are excluded from classification, not treated as all-negative or normal. Targets are overlapping NORM, MI, STTC, CD and HYP labels derived from all present diagnostic SCP keys, including keys with zero recorded likelihood. Official patient folds 1–8, 9 and 10 define training, validation and test respectively.

| Partition | Eligible ECGs | Unique patients | Official folds |
|---|---:|---:|---|
| Train | 17,084 | 14,823 | 1–8 |
| Validation | 2,146 | 1,917 | 9 |
| Test | 2,158 | 1,877 | 10 |

No ECG or patient identifier overlaps were found between the saved prediction partitions.

Input preprocessing retains every complete lead row and its calibration pulse: original page rows y=80:2960, full width x=0:2240, then grayscale-to-ink conversion and nonoverlapping 2×2 maximum-ink pooling. The tensor is **12 × 120 × 1120**, with one channel per aligned lead. A single global mean and standard deviation fitted on training pixels standardize all partitions. Patient IDs, ECG IDs, SCP dictionaries, gain values and other numeric metadata are not model features.

The **18,025-parameter** model starts with depthwise lead filtering; pointwise convolutions then combine aligned lead channels. Three convolution stages produce a 24 × 15 × 70 map, pooled to four vertical bands and arranged into **70 spatial steps × 96 features**. A single unidirectional **LSTM with hidden size 32** is followed by mean pooling, dropout and five logits. The sequence includes calibration/margins and denotes image-feature columns, not recovered 500 Hz waveform samples. Five independent sigmoid probabilities need not sum to one.

Training used unweighted BCE with logits and AdamW, seed 42, batch size 16, learning rate 0.001, weight decay 0.0001, dropout 0.2, gradient clipping at 1, and 4 CPU threads. The cosine schedule has minimum learning rate 1e-05. The prespecified budget was 20 epochs with patience 4. No augmentation, pretrained weights, class weighting or hyperparameter search was used.

Checkpoint selection strictly maximized validation macro ROC-AUC, then macro average precision, then minimized validation probability BCE; the earliest exact tie was retained. All label decisions use the prespecified **probability ≥ 0.5** rule. There was one final test inference after the checkpoint, threshold, inputs and code were frozen; no test-guided tuning or calibration was performed.

## Selected-checkpoint results

| Metric | Train | Validation | Test | Test 95% patient interval |
|---|---:|---:|---:|---|
| Macro ROC-AUC (primary) | 0.9003 | 0.8816 | 0.8698 | 0.8593–0.8805 |
| Macro average precision | 0.7614 | 0.7134 | 0.7035 | 0.6815–0.7251 |
| Macro F1 | 0.6631 | 0.6171 | 0.6131 | 0.5903–0.6317 |
| Micro F1 | 0.7190 | 0.6839 | 0.6779 | 0.6599–0.6950 |
| Label accuracy | 0.8673 | 0.8507 | 0.8495 | 0.8413–0.8575 |
| Exact-match accuracy | 0.5708 | 0.5200 | 0.5259 | 0.5031–0.5465 |

Macro scores average the five labels equally. Micro F1 pools all record-label decisions. Label accuracy counts correct individual binary decisions; exact-match accuracy requires all five decisions on an ECG to be correct. These accuracies are not interchangeable. Supports overlap and therefore do not sum to the ECG count.

| Label | Train positives | Validation positives | Test positives | Test ROC-AUC | Test AP | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NORM | 7,596 | 955 | 963 | 0.9127 | 0.8769 | 0.7652 | 0.8764 | 0.8170 |
| MI | 4,379 | 540 | 550 | 0.8289 | 0.6410 | 0.7000 | 0.4327 | 0.5348 |
| STTC | 4,186 | 528 | 521 | 0.9034 | 0.7487 | 0.7807 | 0.5739 | 0.6615 |
| CD | 3,907 | 495 | 496 | 0.8462 | 0.7177 | 0.8020 | 0.4899 | 0.6083 |
| HYP | 2,119 | 268 | 262 | 0.8577 | 0.5332 | 0.7025 | 0.3244 | 0.4439 |

At the fixed threshold, the lowest test recall is for **HYP**: 85/262 positives detected (0.3244); 177 positives missed and 36 false positives. This describes the saved held-out error profile and is not a basis for retuning this test set.

Selected-checkpoint train minus validation differences are **+0.0186** for macro ROC-AUC and **+0.0460** for macro F1. These gaps alone do not establish an overfitting mechanism or isolate the contribution of the LSTM. Only one configuration and training seed were evaluated.

The epoch training curve contains online, pre-update batch BCE averaged by record count. It is not a full training-partition metric computed after each epoch, and it includes training-time dropout. The training metrics in the tables above were evaluated separately at the selected checkpoint with dropout disabled. Validation metrics in the history use the complete validation partition after each epoch.

Training/evaluation runtime recorded by the runner was 181.83 minutes; training-module runtime was 173.45 minutes. Early stopping: triggered.

## Figures

[Five test ROC curves (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/roc_curves.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/roc_curves.svg>)

[Five test precision–recall curves (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/precision_recall_curves.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/precision_recall_curves.svg>)

[Per-label precision, recall and F1 (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/per_label_metrics.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/per_label_metrics.svg>)

[Partition summary with test uncertainty (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/partition_summary.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/partition_summary.svg>)

[Training curve (online training loss and epoch-end validation metrics)](</Users/miriamistanboli/Downloads/ecg swiss/models/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/training_curve.png>)

[Five test binary confusion matrices at the fixed 0.5 threshold](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/test_confusion_matrices.png>)

## Uncertainty and limitations

Saved 95% intervals use **500 patient-cluster percentile bootstrap resamples**, seed 2024, over 1,877 test patients. Each sampled patient contributes all of their ECGs, with multiplicity when resampled. Intervals quantify test-sampling uncertainty conditional on this fitted model. They do not cover training-seed variation, model-selection uncertainty, annotation error, or external-domain shift. No multiplicity adjustment or bias correction is applied.

| Metric | Defined bootstrap resamples | Undefined resamples |
|---|---:|---:|
| Macro ROC-AUC (primary) | 500 | 0 |
| Macro average precision | 500 | 0 |
| Macro F1 | 500 | 0 |
| Micro F1 | 500 | 0 |
| Label accuracy | 500 | 0 |
| Exact-match accuracy | 500 | 0 |

ROC-AUC requires positive and negative truth examples, and AP requires positives. Full-five-label ranking scores remain undefined if a requisite label outcome is missing; intervals use only defined resamples. No defined-label-only average substitutes for the primary metric. Precision, recall and F1 use zero for zero denominators. Probability BCE uses float64 clipping and can differ from logits BCE at saturated probabilities.

These clean, consistently rendered images do not establish performance on photographs, scanned paper, other ECG-machine layouts, or clinical deployment. Acquisition artifacts and annotation uncertainty persist. Page-wide gain reduction can compress small features; rasterization and maximum-ink pooling merge fine details and thicken strokes. Retained calibration pulses document scale, but the model does not explicitly reconstruct physical amplitudes or source samples. No external or prospective clinical validation was performed.

This is a five-label PTB-XL patient cohort, while the prior Mendeley runs used four mutually exclusive folder labels from a different image source. A direct accuracy ranking against those LDA, CNN or spatial CNN+LSTM results would confound cohort, targets, preprocessing and evaluation. No such ranking is made. The test results are now exposed; subsequent development must retain that fact and cannot present the same test cohort as newly untouched evidence.

## Reproducibility evidence

- Completed UTC: `2026-09-24T14:12:50.334971+00:00`.
- Frozen configuration SHA-256: `02a9979bfdc33a82d440ebbb9aa98fb7d98cb44c710513123b562078c399c72b`.
- Selected checkpoint SHA-256: `34c1b784b0658185c885531b94a1743801c9d0f8182e7db7aa29a7ac6d872a7d`.
- Training-only checkpoint replay matched saved probabilities exactly; prior protected artifacts were unchanged.
- [Passed run verification](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/verification.json>); [run record](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/run.json>); [frozen configuration](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/config.json>).
- Verification checks saved-artifact integrity and consistency. It does not independently prove the absence of unrecorded inference; bootstrap interval bounds are structurally checked, not numerically resampled again.
- [Provenance/source hashes](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/provenance.json>); [training metadata](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/training_metadata.json>); [history](</Users/miriamistanboli/Downloads/ecg swiss/models/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/history.csv>).
- [Patient-bootstrap results](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/test_patient_bootstrap.json>); [report input/output hashes](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/figures/report_manifest.json>).

Regenerate reporting outputs from this completed run (default refuses existing reporting files):

```bash
cd "/Users/miriamistanboli/Downloads/ecg swiss"
/opt/anaconda3/bin/python -m src.report_ptbxl_run --run-id ptbxl_cnn_lstm_image_v1_seed42
```

Use `--overwrite` only to replace this command's Markdown/figure/manifest outputs; saved predictions, metrics, models and provenance remain read-only.
