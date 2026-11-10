# PTB-XL hybrid ViT: paired exploratory comparisons

Hybrid − TCN (primary): **+0.0099** macro ROC-AUC; 95% paired interval **[+0.0043, +0.0155]**. The interval lies above zero; hybrid test macro ROC-AUC is higher conditional on these fitted models and this cohort.

TCN is the prespecified primary reference; CNN–LSTM is secondary. Both comparisons preserve every metric, including deterioration. All differences below are hybrid minus reference. No checkpoint or threshold is chosen from these test results.

Exploratory comparison: the test cohort was inspected for the previous model before this next model was developed. Patient-bootstrap intervals quantify sampling uncertainty conditional on these two fitted models, their fixed 0.5 threshold and this observed cohort. They exclude training-seed variability, architecture-search/adaptive-selection uncertainty and external-domain generalization. No multiplicity adjustment or bias correction is applied. Test outcomes do not revise either checkpoint or the separately reported validation ranking.

All train/validation/test ECG IDs, patient IDs, image paths, five truth columns and row order match; source manifests, cache hashes, preprocessing, training-only normalization, folds, class order, selection rule and threshold match. Passed verification records bind unchanged saved artifacts.

Each comparison uses 500 patient-cluster draws, seed 2024, with all records and full multiplicity for every sampled patient. The same group order and seed also yield the same draw sequence across the two comparisons. Percentile limits use linear 2.5th/97.5th quantiles. Undefined paired replicates are counted and omitted; no multiplicity adjustment is applied.

## TCN (primary)

Hybrid − TCN (primary): **+0.0099** macro ROC-AUC; 95% paired interval **[+0.0043, +0.0155]**. The interval lies above zero; hybrid test macro ROC-AUC is higher conditional on these fitted models and this cohort.

Validation ordering: **hybrid higher**, using macro ROC-AUC, then macro AP, then lower probability BCE. This is descriptive ordering of separately selected checkpoints, independent of the test comparison.

| Test metric | Reference | Hybrid | Hybrid − reference | 95% paired interval | Defined / 500 |
|---|---:|---:|---:|---|---:|
| Macro ROC-AUC (primary) | 0.8803 | 0.8902 | +0.0099 | [+0.0043, +0.0155] | 500 |
| Macro average precision | 0.7325 | 0.7519 | +0.0194 | [+0.0081, +0.0308] | 500 |
| Macro F1 | 0.6409 | 0.6717 | +0.0308 | [+0.0141, +0.0466] | 500 |
| Micro F1 | 0.7085 | 0.7140 | +0.0056 | [-0.0073, +0.0182] | 500 |
| Exact-match accuracy | 0.5616 | 0.5718 | +0.0102 | [-0.0057, +0.0282] | 500 |
| Label accuracy | 0.8608 | 0.8579 | -0.0029 | [-0.0086, +0.0030] | 500 |

| Test label | Δ ROC-AUC | Δ AP | Δ Recall | Δ Specificity | Reference TP / FN | Hybrid TP / FN | Δ TP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORM | +0.0035 | +0.0074 | +0.0093 | -0.0293 | 843 / 120 | 852 / 111 | +9 / -9 |
| MI | +0.0015 | +0.0107 | -0.0636 | +0.0050 | 309 / 241 | 274 / 276 | -35 / +35 |
| STTC | +0.0040 | -0.0098 | +0.0672 | -0.0214 | 330 / 191 | 365 / 156 | +35 / -35 |
| CD | +0.0156 | +0.0232 | +0.0282 | +0.0042 | 270 / 226 | 284 / 212 | +14 / -14 |
| HYP | +0.0249 | +0.0657 | +0.2519 | -0.0343 | 73 / 189 | 139 / 123 | +66 / -66 |

Labels with lower hybrid recall at 0.5: **MI**. These are descriptive per-label differences without per-label confidence intervals. Positive ΔFN means more missed positive labels; positive ΔTP means more detected positive labels.

## CNN–LSTM (secondary)

Hybrid − CNN–LSTM (secondary): **+0.0204** macro ROC-AUC; 95% paired interval **[+0.0140, +0.0272]**. The interval lies above zero; hybrid test macro ROC-AUC is higher conditional on these fitted models and this cohort.

Validation ordering: **hybrid higher**, using macro ROC-AUC, then macro AP, then lower probability BCE. This is descriptive ordering of separately selected checkpoints, independent of the test comparison.

| Test metric | Reference | Hybrid | Hybrid − reference | 95% paired interval | Defined / 500 |
|---|---:|---:|---:|---|---:|
| Macro ROC-AUC (primary) | 0.8698 | 0.8902 | +0.0204 | [+0.0140, +0.0272] | 500 |
| Macro average precision | 0.7035 | 0.7519 | +0.0484 | [+0.0332, +0.0621] | 500 |
| Macro F1 | 0.6131 | 0.6717 | +0.0586 | [+0.0401, +0.0772] | 500 |
| Micro F1 | 0.6779 | 0.7140 | +0.0361 | [+0.0213, +0.0501] | 500 |
| Exact-match accuracy | 0.5259 | 0.5718 | +0.0459 | [+0.0267, +0.0623] | 500 |
| Label accuracy | 0.8495 | 0.8579 | +0.0084 | [+0.0014, +0.0147] | 500 |

| Test label | Δ ROC-AUC | Δ AP | Δ Recall | Δ Specificity | Reference TP / FN | Hybrid TP / FN | Δ TP / FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| NORM | +0.0106 | +0.0199 | +0.0083 | +0.0050 | 844 / 119 | 852 / 111 | +8 / -8 |
| MI | +0.0396 | +0.0822 | +0.0655 | +0.0006 | 238 / 312 | 274 / 276 | +36 / -36 |
| STTC | +0.0128 | +0.0231 | +0.1267 | -0.0312 | 299 / 222 | 365 / 156 | +66 / -66 |
| CD | +0.0258 | +0.0478 | +0.0827 | -0.0048 | 243 / 253 | 284 / 212 | +41 / -41 |
| HYP | +0.0131 | +0.0690 | +0.2061 | -0.0327 | 85 / 177 | 139 / 123 | +54 / -54 |

Labels with lower hybrid recall at 0.5: **none**. These are descriptive per-label differences without per-label confidence intervals. Positive ΔFN means more missed positive labels; positive ΔTP means more detected positive labels.

## Capacity, device and timing

| Model | Parameters | Actual device | Selected / completed epochs | Training timer (s) | Run timer (s) | UTC wall span (s) |
|---|---:|---|---:|---:|---:|---:|
| TCN (primary) | 17,161 | cpu | 16 / 20 | 10536.6 | 10957.3 | 15222.3 |
| CNN–LSTM (secondary) | 18,025 | cpu | 15 / 19 | 10407.2 | 10909.7 | 10910.1 |
| Hybrid ViT | 23,881 | cpu | 19 / 20 | 11861.3 | 12376.4 | 12376.9 |

The encoder family, inputs, seed and optimizer/epoch cap match, but heads, parameter counts, dropout placement, learning trajectories and stopping epochs differ. This is a single-seed pipeline comparison, not a capacity-controlled isolation of attention. Recorded elapsed timers include overhead and waits; they are not pure CPU time. CPU thread counts, backend selection, thermals, concurrent load and sleep/suspend can affect timings. UTC wall spans and performance-counter durations are reported separately; these runs do not constitute a controlled speed or efficiency benchmark, even when all actual devices are CPU.

Label meanings: NORM = normal ECG; MI = myocardial infarction; STTC = ST/T changes; CD = conduction disturbance; HYP = hypertrophy. Labels overlap. Label accuracy counts individual binary decisions; exact-match accuracy requires all five labels correct for an ECG.

![All six scores and paired differences](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_hybrid_vit_image_v1_seed42/comparison.png>)

![All five per-label differences](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_hybrid_vit_image_v1_seed42/per_label_differences.png>)

CSV evidence includes every partition, model, all six summary metrics, and per-label AUC/AP/precision/recall/specificity/F1/confusion counts and deltas.
