# PTB-XL image hybrid ViT report

**Completed and verified: `ptbxl_hybrid_vit_image_v1_seed42`.** Test macro ROC-AUC is **0.8902** (saved 95% patient-bootstrap interval **0.8809–0.9000**). Selected checkpoint: epoch **19 of 20**.

Hybrid − TCN (primary): **+0.0099** macro ROC-AUC; 95% paired interval **[+0.0043, +0.0155]**. The interval lies above zero; hybrid test macro ROC-AUC is higher conditional on these fitted models and this cohort.

This is an exploratory evaluation on a test cohort already examined for the CNN–LSTM and TCN. Reporting reads verified saved predictions and performs two new paired bootstrap comparisons; it performs no inference, refitting, threshold tuning or calibration.

## Cohort, representation and architecture

The source is [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) ([version DOI](https://doi.org/10.13026/kfzx-aw45); [original publication](https://doi.org/10.1038/s41597-020-0495-6)), rendered into a single clean synthetic page style from native 500 Hz, ten-second, 12-lead signals. Of 21,799 rendered records, 21,388 map to at least one diagnostic superclass; 411 unmapped records are excluded rather than assigned all-negative or normal labels. Official patient folds 1–8 / 9 / 10 define train / validation / test. All present diagnostic SCP keys, including zero recorded likelihood, contribute to the superclass union.

Targets overlap: **NORM** = normal ECG; **MI** = myocardial infarction; **STTC** = ST/T changes; **CD** = conduction disturbance; **HYP** = hypertrophy.

| Partition | ECGs | Patients |
|---|---:|---:|
| Train | 17,084 | 14,823 |
| Validation | 2,146 | 1,917 |
| Test | 2,158 | 1,877 |

The unchanged **12 × 120 × 1120** input retains complete aligned lead rows and calibration pulses. Grayscale inversion and isotropic 2×2 maximum-ink pooling precede global training-only standardization. IDs, labels and numeric metadata are not predictor features.

The **23,881-parameter** model uses the same depthwise/pointwise CNN encoder family as the two references, independently trained from scratch. Its **70 image-feature tokens × 96 features** are projected to width 32, receive learned positional embeddings, and pass through two independently initialized pre-norm Transformer encoder layers with four attention heads, feedforward width 64, GELU and dropout. Final LayerNorm, mean token pooling and a five-logit head produce independent sigmoid probabilities. It is a CNN-feature hybrid ViT, not a vanilla raw-patch ViT or a recovered-waveform model. The global attention, encoder context and pooling make this an offline classifier; image tokens are not native waveform samples or a causal streaming signal.

Seed 42; unweighted BCE with logits; AdamW; batch 16; learning rate 0.001; weight decay 0.0001; dropout 0.2; gradient clipping 1; cosine schedule to 1e-05; cap 20 epochs, patience 4. Actual device: **cpu**; requested device: auto; configured CPU threads: 4. No augmentation, pretraining, class weighting or configuration search was used. The fixed decision threshold is probability ≥ 0.5.

Checkpoint selection lexicographically maximizes validation macro ROC-AUC, then macro average precision, then minimizes probability BCE; earliest exact ties are retained. The verified bookkeeping records one final test inference after checkpoint/configuration freezing and an exact training-only checkpoint replay. Artifact verification cannot rule out unrecorded activity.

## Selected-checkpoint metrics

| Metric | Train | Validation | Test | Test 95% patient interval |
|---|---:|---:|---:|---|
| Macro ROC-AUC (primary) | 0.9259 | 0.9016 | 0.8902 | 0.8809–0.9000 |
| Macro average precision | 0.8224 | 0.7592 | 0.7519 | 0.7321–0.7710 |
| Macro F1 | 0.7406 | 0.6840 | 0.6717 | 0.6523–0.6895 |
| Micro F1 | 0.7758 | 0.7286 | 0.7140 | 0.6972–0.7311 |
| Exact-match accuracy | 0.6351 | 0.5811 | 0.5718 | 0.5497–0.5930 |
| Label accuracy | 0.8882 | 0.8643 | 0.8579 | 0.8496–0.8661 |
| Positive-support weighted F1 | 0.7716 | 0.7210 | 0.7056 | Not bootstrapped |
| Probability BCE | 0.2734 | 0.3293 | 0.3445 | Not bootstrapped |

Macro metrics weight labels equally; micro F1 pools binary decisions. Label accuracy measures correct record-label decisions, while exact-match accuracy requires all five labels correct for each ECG. Neither is a four-class single-label accuracy, and neither supports a direct ranking against the separate Mendeley experiments.

| Test label | Positives | ROC-AUC | AP | Precision | Recall | Specificity | F1 | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NORM | 963 | 0.9233 | 0.8968 | 0.7710 | 0.8847 | 0.7883 | 0.8240 | 852 | 111 | 253 | 942 |
| MI | 550 | 0.8685 | 0.7232 | 0.7307 | 0.4982 | 0.9372 | 0.5924 | 274 | 276 | 101 | 1507 |
| STTC | 521 | 0.9163 | 0.7718 | 0.7300 | 0.7006 | 0.9175 | 0.7150 | 365 | 156 | 135 | 1502 |
| CD | 496 | 0.8719 | 0.7655 | 0.8068 | 0.5726 | 0.9591 | 0.6698 | 284 | 212 | 68 | 1594 |
| HYP | 262 | 0.8708 | 0.6021 | 0.5865 | 0.5305 | 0.9483 | 0.5571 | 139 | 123 | 98 | 1798 |

Lowest test recall at 0.5: **MI**, 274/550 positives detected (0.4982), with 276 false negatives and 101 false positives. This describes the saved result; it does not trigger threshold tuning.

Train minus validation gaps: macro ROC-AUC **+0.0243**; macro F1 **+0.0566**. These gaps do not isolate an overfitting mechanism.

## Comparisons and uncertainty

Hybrid − CNN–LSTM (secondary): **+0.0204** macro ROC-AUC; 95% paired interval **[+0.0140, +0.0272]**. The interval lies above zero; hybrid test macro ROC-AUC is higher conditional on these fitted models and this cohort.

[Full paired comparison with TCN and CNN–LSTM](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_hybrid_vit_image_v1_seed42/comparison.md>) preserves all six metrics, all five label deltas and both improvements and deteriorations. Capacity and dropout topology differ, so this is not an attention-only ablation.

Single-model intervals are the saved 500 patient-cluster percentile resamples, seed 2024; this report checks their structure and point estimates rather than resampling them. The two paired comparisons are newly recomputed from saved predictions using 500 shared-patient draws each, seed 2024. All intervals are conditional on these fitted models and exclude training-seed variability, adaptive model selection, clinical annotation uncertainty and external-domain shift. There is no multiplicity adjustment. Undefined ranking metrics/resamples remain explicitly undefined; no defined-label-only score replaces the primary metric.

Clean synthetic pages do not establish performance on clinical scans, photographs, other layouts or prospective populations. Gain reduction, rasterization and pooling lose fine detail. The retained calibration pulses do not constitute validated physical-signal reconstruction. Further development cannot call this repeatedly observed test cohort untouched.

## Training history and recorded runtime

The plotted training loss is online, pre-update batch BCE with dropout and changing weights, not a full epoch-end training evaluation. The table's training metrics use the selected checkpoint with dropout disabled; validation history evaluates the complete validation split after each epoch.

Recorded runner elapsed-timer duration: **206.27 minutes**; recorded training-module elapsed-timer duration: **197.69 minutes**. The run's UTC wall span was **206.28 minutes**, computed from completion minus start timestamps. The recorded performance-counter timers include execution and waits; they are not pure CPU execution time and may exclude system-suspend time on this host. The UTC span includes sleep/suspend gaps, subject to system-clock corrections. A timer-versus-wall difference alone cannot quantify time asleep. These durations describe this execution and are not a controlled speed benchmark.

Device/backend differences, thread use, thermal conditions, concurrent work, sleep/suspend and differing stopping epochs prevent a controlled CPU-speed or efficiency conclusion.

## Figures and provenance

[Roc curves PNG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/roc_curves.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/roc_curves.svg>)

[Precision recall curves PNG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/precision_recall_curves.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/precision_recall_curves.svg>)

[Per label metrics PNG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/per_label_metrics.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/per_label_metrics.svg>)

[Partition summary PNG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/partition_summary.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/partition_summary.svg>)

[Training history PNG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/training_history.png>) · [SVG](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/figures/training_history.svg>)

[Verification](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/verification.json>) · [Configuration](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/config.json>) · [Saved predictions and metrics](</Users/miriamistanboli/Downloads/ecg swiss/results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/run.json>) · [Dataset card](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_image_dataset.md>)

Checkpoint SHA-256: `cfc590331ed3dd344bc4dfc0ca2c9e78609d99f51d526434d29788c793c2cfea`. Configuration SHA-256: `ffad856c5fe9fba2877c75b135bdf26e58f8fdc0b7c83369ec866ba5acd9860e`.

Report and comparison manifests bind every read numerical artifact and every generated output. No old result or model artifact is rewritten.
