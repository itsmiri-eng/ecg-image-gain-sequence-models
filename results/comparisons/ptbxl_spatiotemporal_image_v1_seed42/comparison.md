# Exploratory PTB-XL model comparison

Previous: `ptbxl_cnn_lstm_image_v1_seed42`. New: `ptbxl_spatiotemporal_image_v1_seed42`.

**Primary test difference (new − previous): macro ROC-AUC +0.0105; 95% paired patient-bootstrap interval [+0.0045, +0.0167].** The primary interval lies above zero: the new model has higher test macro ROC-AUC under this conditional, exploratory patient-bootstrap comparison.

Exploratory comparison: the test cohort was inspected for the previous model before this next model was developed. Patient-bootstrap intervals quantify sampling uncertainty conditional on these two fitted models, their fixed 0.5 threshold and this observed cohort. They exclude training-seed variability, architecture-search/adaptive-selection uncertainty and external-domain generalization. No multiplicity adjustment or bias correction is applied. Test outcomes do not revise either checkpoint or the separately reported validation ranking.

Both completed runs passed artifact verification. ECG IDs, patient IDs, truth labels and row order match exactly in all three partitions; cache, preprocessing, official split and threshold bindings match. Only saved predictions were evaluated.

## Validation ordering (separate from test)

Ordering: **new higher**, using Lexicographic (validation macro ROC-AUC, macro AP, -probability BCE). Descriptive validation ordering of separately checkpoint-selected runs; independent of the test comparison. No checkpoint is changed.

## Partition metrics

Exact-match accuracy requires all five labels correct for an ECG. Label accuracy counts correct individual record-label decisions. Labels overlap; predictions are independent probabilities at the fixed 0.5 threshold.

| Partition | Model | Macro ROC-AUC | Macro AP | Macro F1 | Micro F1 | Exact-match accuracy | Label accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| train | Previous | 0.9003 | 0.7614 | 0.6631 | 0.7190 | 0.5708 | 0.8673 |
| train | New | 0.9093 | 0.7866 | 0.6865 | 0.7457 | 0.5957 | 0.8778 |
| validation | Previous | 0.8816 | 0.7134 | 0.6171 | 0.6839 | 0.5200 | 0.8507 |
| validation | New | 0.8889 | 0.7348 | 0.6521 | 0.7164 | 0.5629 | 0.8630 |
| test | Previous | 0.8698 | 0.7035 | 0.6131 | 0.6779 | 0.5259 | 0.8495 |
| test | New | 0.8803 | 0.7325 | 0.6409 | 0.7085 | 0.5616 | 0.8608 |

## Paired test differences

500 patient-cluster resamples, seed 2024, linear 2.5th/97.5th percentiles. Each patient draw contributes every ECG, including full multiplicity for repeat draws, to both models. Undefined paired replicates are omitted and counted separately for each metric.

| Metric | New − previous | 95% paired interval | Defined / 500 |
|---|---:|---|---:|
| Macro ROC-AUC (primary) | +0.0105 | [+0.0045, +0.0167] | 500 |
| Macro average precision | +0.0290 | [+0.0150, +0.0427] | 500 |
| Macro F1 | +0.0278 | [+0.0088, +0.0461] | 500 |
| Micro F1 | +0.0306 | [+0.0183, +0.0442] | 500 |
| Exact-match accuracy | +0.0357 | [+0.0203, +0.0514] | 500 |
| Label accuracy | +0.0113 | [+0.0058, +0.0173] | 500 |

## Per-label decisions at 0.5

| Partition | Label | Previous precision | New precision | Previous recall | New recall | Previous F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | NORM | 0.8418 | 0.8508 | 0.8734 | 0.8748 | 0.8573 | 0.8627 |
| train | MI | 0.7138 | 0.7428 | 0.5394 | 0.6330 | 0.6145 | 0.6835 |
| train | STTC | 0.7772 | 0.7867 | 0.6266 | 0.6818 | 0.6938 | 0.7305 |
| train | CD | 0.8415 | 0.8315 | 0.5301 | 0.6025 | 0.6504 | 0.6987 |
| train | HYP | 0.7251 | 0.7860 | 0.3808 | 0.3223 | 0.4994 | 0.4572 |
| validation | NORM | 0.7904 | 0.8090 | 0.8806 | 0.8827 | 0.8331 | 0.8443 |
| validation | MI | 0.6676 | 0.6886 | 0.4315 | 0.5611 | 0.5242 | 0.6184 |
| validation | STTC | 0.7558 | 0.7696 | 0.6155 | 0.6515 | 0.6785 | 0.7056 |
| validation | CD | 0.8152 | 0.8045 | 0.4990 | 0.5818 | 0.6190 | 0.6753 |
| validation | HYP | 0.6397 | 0.7117 | 0.3246 | 0.2948 | 0.4307 | 0.4169 |
| test | NORM | 0.7652 | 0.7945 | 0.8764 | 0.8754 | 0.8170 | 0.8330 |
| test | MI | 0.7000 | 0.7392 | 0.4327 | 0.5618 | 0.5348 | 0.6384 |
| test | STTC | 0.7807 | 0.7674 | 0.5739 | 0.6334 | 0.6615 | 0.6940 |
| test | CD | 0.8020 | 0.7826 | 0.4899 | 0.5444 | 0.6083 | 0.6421 |
| test | HYP | 0.7025 | 0.6887 | 0.3244 | 0.2786 | 0.4439 | 0.3967 |

## Model size and recorded durations

Training and run timers retain the recorded elapsed `time.perf_counter` values. The UTC wall span is computed separately from the saved run start and completion timestamps.

| Model | Parameters | Selected epoch / epochs run | Training recorded timer (s) | Run recorded timer (s) | Run UTC wall span (s) | Seed |
|---|---:|---:|---:|---:|---:|---:|
| Previous | 18,025 | 15 / 19 | 10407.2 | 10909.7 | 10910.1 | 42 |
| New | 17,161 | 16 / 20 | 10536.6 | 10957.3 | 15222.3 | 42 |

Recorded elapsed timers are not pure CPU time and can account for host sleep differently from UTC timestamps. The UTC wall span includes intervening sleep and pauses and can be affected by clock adjustments. Host sleep, thermals and competing load can differ between runs; these durations support no speed or efficiency conclusion.

![Test metrics and paired differences](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.png>)

Machine-readable files: comparison.json, partition_metrics.csv, per_label_metrics.csv and metric_differences.csv.
