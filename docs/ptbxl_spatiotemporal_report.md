# PTB-XL small image spatiotemporal network report

**Completed and verified run: `ptbxl_spatiotemporal_image_v1_seed42`.** Held-out test macro ROC-AUC is **0.8803** (95% patient-bootstrap interval **0.8705–0.8904**). The checkpoint was selected at epoch **16 of 20 completed epochs** using validation data only.

This report is derived from saved predictions, metrics, training history, and provenance. It performs no model inference, refitting, threshold selection, or new bootstrap sampling. The test cohort was previously examined for the CNN–LSTM, so this next-model evaluation is exploratory rather than a fresh confirmatory test.

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

The **17,161-parameter** model starts with depthwise lead filtering; pointwise convolutions then combine aligned lead channels. Three convolution stages produce a 24 × 15 × 70 map, pooled to four vertical bands and arranged into **70 spatial steps × 96 features**. A **96-to-32 projection and two residual temporal-convolution blocks** replace the LSTM. Each block contains two kernel-3 left-padded convolutions, at dilation 1 in the first block and 2 in the second. Each convolution is followed by LayerNorm across 32 channels independently at each position, ReLU and dropout; each residual sum is followed by ReLU. Mean pooling, dropout and a linear head produce five logits. The head's receptive field is 13 supplied feature positions before whole-sequence averaging. The sequence includes calibration/margins and denotes image-feature columns, not recovered 500 Hz waveform samples. Five independent sigmoid probabilities need not sum to one. Left padding makes the temporal head causal given its inputs, but the upstream image CNN/GroupNorm and global pooling use whole-image context: this is an offline classifier, not streaming ECG processing. The design is inspired by [Bai et al.](https://arxiv.org/abs/1803.01271), not an exact reproduction or a claim of superior ECG performance.

Training used unweighted BCE with logits and AdamW, seed 42, batch size 16, learning rate 0.001, weight decay 0.0001, dropout 0.2, gradient clipping at 1, and 4 CPU threads. The cosine schedule has minimum learning rate 1e-05. The prespecified budget was 20 epochs with patience 4. No augmentation, pretrained weights, class weighting or hyperparameter search was used.

Checkpoint selection strictly maximized validation macro ROC-AUC, then macro average precision, then minimized validation probability BCE; the earliest exact tie was retained. All label decisions use the prespecified **probability ≥ 0.5** rule. There was one final test inference after the checkpoint, threshold, inputs and code were frozen; no test-guided tuning or calibration was performed.

## Selected-checkpoint results

| Metric | Train | Validation | Test | Test 95% patient interval |
|---|---:|---:|---:|---|
| Macro ROC-AUC (primary) | 0.9093 | 0.8889 | 0.8803 | 0.8705–0.8904 |
| Macro average precision | 0.7866 | 0.7348 | 0.7325 | 0.7131–0.7537 |
| Macro F1 | 0.6865 | 0.6521 | 0.6409 | 0.6200–0.6613 |
| Micro F1 | 0.7457 | 0.7164 | 0.7085 | 0.6915–0.7254 |
| Label accuracy | 0.8778 | 0.8630 | 0.8608 | 0.8524–0.8689 |
| Exact-match accuracy | 0.5957 | 0.5629 | 0.5616 | 0.5393–0.5832 |
| Weighted F1 (positive-support weights) | 0.7348 | 0.7031 | 0.6939 | Not bootstrapped |

Macro scores average the five labels equally. Micro F1 pools all record-label decisions. Label accuracy counts correct individual binary decisions; exact-match accuracy requires all five decisions on an ECG to be correct. These accuracies are not interchangeable. Supports overlap and therefore do not sum to the ECG count.

| Label | Train positives | Validation positives | Test positives | Test ROC-AUC | Test AP | Test precision | Test recall | Test specificity | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NORM | 7,596 | 955 | 963 | 0.9198 | 0.8894 | 0.7945 | 0.8754 | 0.8176 | 0.8330 |
| MI | 4,379 | 540 | 550 | 0.8670 | 0.7125 | 0.7392 | 0.5618 | 0.9322 | 0.6384 |
| STTC | 4,186 | 528 | 521 | 0.9122 | 0.7816 | 0.7674 | 0.6334 | 0.9389 | 0.6940 |
| CD | 3,907 | 495 | 496 | 0.8563 | 0.7423 | 0.7826 | 0.5444 | 0.9549 | 0.6421 |
| HYP | 2,119 | 268 | 262 | 0.8459 | 0.5364 | 0.6887 | 0.2786 | 0.9826 | 0.3967 |

At the fixed threshold, the lowest test recall is for **HYP**: 73/262 positives detected (0.2786); 189 positives missed and 33 false positives. This describes the saved held-out error profile and is not a basis for retuning this test set.

Selected-checkpoint train minus validation differences are **+0.0204** for macro ROC-AUC and **+0.0344** for macro F1. These gaps alone do not establish an overfitting mechanism or isolate the contribution of the temporal-convolution head. Only one configuration and training seed were evaluated.

The epoch training curve contains online, pre-update batch BCE averaged by record count. It is not a full training-partition metric computed after each epoch, and it includes training-time dropout. The training metrics in the tables above were evaluated separately at the selected checkpoint with dropout disabled. Validation metrics in the history use the complete validation partition after each epoch.

Recorded runner elapsed-timer duration: **182.62 minutes**; recorded training-module elapsed-timer duration: **175.61 minutes**. The run's UTC wall span was **253.71 minutes**, computed from completion minus start timestamps. The recorded performance-counter timers include execution and waits; they are not pure CPU execution time and may exclude system-suspend time on this host. The UTC span includes sleep/suspend gaps, subject to system-clock corrections. A timer-versus-wall difference alone cannot quantify time asleep. These durations describe this execution and are not a controlled speed benchmark.

Early stopping: not triggered before the epoch cap.

## Figures

[Five test ROC curves (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/roc_curves.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/roc_curves.svg>)

[Five test precision–recall curves (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/precision_recall_curves.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/precision_recall_curves.svg>)

[Per-label precision, recall and F1 (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/per_label_metrics.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/per_label_metrics.svg>)

[Partition summary with test uncertainty (PNG)](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/partition_summary.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/partition_summary.svg>)

[Training curve (online training loss and epoch-end validation metrics)](</Users/miriamistanboli/Downloads/ecg swiss/models/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/training_curve.png>)

[Five test binary confusion matrices at the fixed 0.5 threshold](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/test_confusion_matrices.png>)

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

This is a five-label PTB-XL patient cohort, while the prior Mendeley runs used four mutually exclusive folder labels from a different image source. A direct accuracy ranking against those LDA, CNN or spatial CNN+LSTM results would confound cohort, targets, preprocessing and evaluation. No such ranking is made. The PTB-XL test result had already been observed in the CNN–LSTM experiment before this run was specified. The same-cohort architecture comparison is exploratory and uses paired patient-level uncertainty; further development cannot present this cohort as newly untouched evidence. Head capacity and dropout placement also differ. Several-seed and independent-data evaluations remain future work.

See the [paired comparison with the preserved PTB-XL CNN–LSTM](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.md>) for matched model differences; that file is produced separately from saved verified predictions.

## Reproducibility evidence

- Completed UTC: `2026-09-25T23:17:59.306508+00:00`.
- Frozen configuration SHA-256: `ee601962031a9f54665d7b9284380979f29cbf68f038d9f2bb891b607c376762`.
- Selected checkpoint SHA-256: `d5f390a876024eca15f353c3ebe78756302ee4b1c8522d81cc44ecb00c095141`.
- Training-only checkpoint replay matched saved probabilities exactly; prior protected artifacts were unchanged.
- [Passed run verification](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/verification.json>); [run record](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/run.json>); [frozen configuration](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/config.json>).
- Verification checks saved-artifact integrity and consistency. It does not independently prove the absence of unrecorded inference; bootstrap interval bounds are structurally checked, not numerically resampled again.
- [Provenance/source hashes](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/provenance.json>); [training metadata](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/training_metadata.json>); [history](</Users/miriamistanboli/Downloads/ecg swiss/models/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/history.csv>).
- [Patient-bootstrap results](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/test_patient_bootstrap.json>); [report input/output hashes](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/figures/report_manifest.json>).

Regenerate reporting outputs from this completed run (default refuses existing reporting files):

```bash
cd "/Users/miriamistanboli/Downloads/ecg swiss"
/opt/anaconda3/bin/python -m src.report_ptbxl_spatiotemporal --run-id ptbxl_spatiotemporal_image_v1_seed42
```

Use `--overwrite` only to replace this command's Markdown/figure/manifest outputs; saved predictions, metrics, models and provenance remain read-only.
