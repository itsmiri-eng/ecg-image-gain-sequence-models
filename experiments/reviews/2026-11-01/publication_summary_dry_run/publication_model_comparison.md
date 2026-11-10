# Publication model comparison

Generated only from saved, completed, verified runs. CSV files retain full floating-point precision; displayed metrics are rounded to six decimals. No fitting, model inference or resampling is performed. Cohorts are separate and are never ranked against one another.

Accuracy means the correct single-class fraction for Mendeley and exact label-set match for PTB-XL. PTB-XL label accuracy is separate. Balanced accuracy means mean class recall for Mendeley and mean over labels of (sensitivity + specificity)/2 for PTB-XL. Undefined specificity or binary balanced accuracy remains blank in CSV; saved zero-division precision/recall/F1 policy is retained.

Neural parameter counts exclude PCA/LDA (NA). Dates and git commits come from saved artifacts; unavailable commit IDs are not replaced with the current checkout. Training-loop, module, runner elapsed and UTC wall-span times are distinct; system suspension can enlarge UTC spans. Missing numeric observations are blank, while NA denotes inapplicable metadata.

## Mendeley four-class

| Model | Test N | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AP | Label accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LDA | 98 | 0.622449 | 0.603176 | 0.563743 | 0.642350 | 0.877085 | unavailable | unavailable |
| CNN | 98 | 0.755102 | 0.568575 | 0.562121 | 0.731042 | 0.900774 | unavailable | unavailable |
| CNN_LSTM | 98 | 0.714286 | 0.631548 | 0.602020 | 0.716840 | 0.894367 | unavailable | unavailable |

| Model | Status | Verification | Parameters | Selected epoch | Epochs | Training seconds | Runner seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LDA | completed | verified | NA | NA | NA | 0.232756 | 1.214809 |
| CNN | completed | verified | 40212 | 25 | 30 | 43.702135 | 46.471704 |
| CNN_LSTM | completed | verified | 44532 | 27 | 30 | 48.188910 | 51.051234 |

### Test sensitivity and specificity

| Model | Label | Positive support | Sensitivity | Specificity | Precision | F1 | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LDA | Normal | 28 | 0.750000 | 0.842857 | 0.656250 | 0.700000 | 0.888776 |
| LDA | MI | 6 | 0.500000 | 0.934783 | 0.333333 | 0.400000 | 0.893116 |
| LDA | Abnormal | 47 | 0.574468 | 0.960784 | 0.931034 | 0.710526 | 0.906550 |
| LDA | History_MI | 17 | 0.588235 | 0.777778 | 0.357143 | 0.444444 | 0.819898 |
| CNN | Normal | 28 | 1.000000 | 0.728571 | 0.595745 | 0.746667 | 0.931633 |
| CNN | MI | 6 | 0.166667 | 0.989130 | 0.500000 | 0.250000 | 0.855072 |
| CNN | Abnormal | 47 | 0.872340 | 1.000000 | 1.000000 | 0.931818 | 0.984147 |
| CNN | History_MI | 17 | 0.235294 | 0.950617 | 0.500000 | 0.320000 | 0.832244 |
| CNN_LSTM | Normal | 28 | 0.928571 | 0.800000 | 0.650000 | 0.764706 | 0.925000 |
| CNN_LSTM | MI | 6 | 0.500000 | 0.934783 | 0.333333 | 0.400000 | 0.844203 |
| CNN_LSTM | Abnormal | 47 | 0.744681 | 0.980392 | 0.972222 | 0.843373 | 0.957864 |
| CNN_LSTM | History_MI | 17 | 0.352941 | 0.913580 | 0.461538 | 0.400000 | 0.850399 |

## PTB-XL five-label

| Model | Test N | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro AUROC | Macro AP | Label accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PTBXL_Image_CNN_LSTM | 2158 | 0.525950 | 0.731083 | 0.613100 | 0.660314 | 0.869801 | 0.703503 | 0.849490 |
| PTBXL_Image_Spatiotemporal | 2158 | 0.561631 | 0.751975 | 0.640854 | 0.693883 | 0.880258 | 0.732456 | 0.860797 |

| Model | Status | Verification | Parameters | Selected epoch | Epochs | Training seconds | Runner seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PTBXL_Image_CNN_LSTM | completed | verified | 18025 | 15 | 19 | 10405.803539 | 10909.669200 |
| PTBXL_Image_Spatiotemporal | completed | verified | 17161 | 16 | 20 | 10535.258610 | 10957.267697 |

### Test sensitivity and specificity

| Model | Label | Positive support | Sensitivity | Specificity | Precision | F1 | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PTBXL_Image_CNN_LSTM | NORM | 963 | 0.876428 | 0.783264 | 0.765186 | 0.817038 | 0.912740 |
| PTBXL_Image_CNN_LSTM | MI | 550 | 0.432727 | 0.936567 | 0.700000 | 0.534831 | 0.828926 |
| PTBXL_Image_CNN_LSTM | STTC | 521 | 0.573896 | 0.948687 | 0.780679 | 0.661504 | 0.903446 |
| PTBXL_Image_CNN_LSTM | CD | 496 | 0.489919 | 0.963899 | 0.801980 | 0.608260 | 0.846160 |
| PTBXL_Image_CNN_LSTM | HYP | 262 | 0.324427 | 0.981013 | 0.702479 | 0.443864 | 0.857734 |
| PTBXL_Image_Spatiotemporal | NORM | 963 | 0.875389 | 0.817573 | 0.794533 | 0.833004 | 0.919776 |
| PTBXL_Image_Spatiotemporal | MI | 550 | 0.561818 | 0.932214 | 0.739234 | 0.638430 | 0.867016 |
| PTBXL_Image_Spatiotemporal | STTC | 521 | 0.633397 | 0.938913 | 0.767442 | 0.694006 | 0.912239 |
| PTBXL_Image_Spatiotemporal | CD | 496 | 0.544355 | 0.954874 | 0.782609 | 0.642093 | 0.856314 |
| PTBXL_Image_Spatiotemporal | HYP | 262 | 0.278626 | 0.982595 | 0.688679 | 0.396739 | 0.845943 |

## Run-specific limits and provenance

- **lda_v1_seed42**: Fixed baseline; no test tuning; printed-ID groups are not verified patient IDs One configuration/seed; no external clinical validation. Printed-ID groups are not verified patient IDs; small test cohort with few MI cases. Neural parameter and epoch counts do not apply to PCA/LDA. Artifacts: `results/lda/lda_v1_seed42`; verification: `["experiments/reviews/2026-11-01/results_audit.json","experiments/logs/delivery_manifest.json"]`.
- **cnn_v1_seed42**: Fixed baseline; no test tuning; printed-ID groups are not verified patient IDs Deterministic on this runtime; cross-device/version equality is not guaranteed. One configuration/seed; no external clinical validation. Printed-ID groups are not verified patient IDs; small test cohort with few MI cases. Artifacts: `results/cnn/cnn_v1_seed42`; verification: `["experiments/reviews/2026-11-01/results_audit.json","experiments/logs/delivery_manifest.json"]`.
- **cnn_lstm_image_v1_seed42**: Spatial image sequence; validation macro F1 selection; previously observed test exploratory; no test tuning; not an isolated architecture ablation Both architecture and checkpoint-selection criterion differ from original CNN (minimum validation loss). This is a pipeline comparison, not an isolated LSTM architecture ablation. CPU determinism configured; identical results across software/hardware versions are not guaranteed One configuration/seed; no external clinical validation. Printed-ID groups are not verified patient IDs; small test cohort with few MI cases. Artifacts: `results/cnn_lstm/cnn_lstm_image_v1_seed42`; verification: `["experiments/reviews/2026-11-01/results_audit.json","experiments/logs/cnn_lstm_delivery_manifest.json"]`.
- **ptbxl_cnn_lstm_image_v1_seed42**: Five-label PTB-XL task; primary macro AUROC=0.869801; subset_accuracy=0.525950; label_accuracy=0.849490. In the legacy experiments/experiment_log.csv only, single-label accuracy/balanced_accuracy columns are intentionally blank; no direct Mendeley ranking. One seed/config, fixed0.5threshold, one final test inference. different five-label PTB-XL cohort and task; no direct accuracy ranking versus four-class Mendeley runs CPU determinism configured; cross-version/hardware identity is not guaranteed One configuration/seed; no external clinical validation. Clean pages rendered from real ECG recordings; fixed 0.5 threshold; repeated test-set observation makes comparisons exploratory. Artifacts: `results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42`; verification: `["results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/verification.json"]`.
- **ptbxl_spatiotemporal_image_v1_seed42**: Five-label PTB-XL task; primary macro AUROC=0.880258; subset_accuracy=0.561631; label_accuracy=0.860797. In the legacy experiments/experiment_log.csv only, single-label accuracy/balanced_accuracy columns are intentionally blank; no direct Mendeley ranking. One seed/config, fixed0.5threshold, one final test inference. Same-fold CNNLSTM comparison is exploratory; prior test result already observed. Exploratory same-cohort comparison to saved CNNLSTM; same inputs/folds/threshold/optimizer/budget, one configuration/seed each. Architecture/head capacity and dropout topology differ; no external or seed robustness established. No Mendeley accuracy ranking. CPU determinism configured; cross-version/hardware identity is not guaranteed One configuration/seed; no external clinical validation. Clean pages rendered from real ECG recordings; fixed 0.5 threshold; repeated test-set observation makes comparisons exploratory. Artifacts: `results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42`; verification: `["results/ptbxl_spatiotemporal/ptbxl_spatiotemporal_image_v1_seed42/verification.json"]`.

All partition metrics are in `publication_model_comparison.csv`; all partition label confusions and sensitivity/specificity are in `publication_model_per_label_metrics.csv`. The expanded master log includes preprocessing, hyperparameters, timestamps, runtimes, sources and exclusions. `publication_model_comparison.provenance.json` binds input/output hashes. Original per-run artifacts and `experiment_log.csv` remain authoritative and unchanged.
