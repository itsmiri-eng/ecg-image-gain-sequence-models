# ECG image classification experiment notes

Status recorded 1 November 2026: five completed formal runs and one hybrid ViT experiment actively training. These notes use the fresh saved-artifact audit, not the older manuscript's reported experiments. Historical motivations are reconstructed from protocols and code; this document does not retrospectively preregister them.

Evidence: [fresh audit](../experiments/reviews/2026-11-01/results_audit.json), [all partition metrics](../experiments/reviews/2026-11-01/all_run_metrics.csv), [per-class and per-label metrics](../experiments/reviews/2026-11-01/all_run_per_label_metrics.csv), and [checkpoint/comparison supplement](../experiments/reviews/2026-11-01/results_supplement.json). The audit recomputed metrics from saved predictions and checked artifact provenance; it did not retrain the models, repeat test inference, or rerun bootstrap resampling. Its failure list is empty. Historical manuscript numbers remain outside the formal results tables.

## Two separate questions and cohorts

**Mendeley, four mutually exclusive supplied folder labels:** Normal, MI, Abnormal and History_MI. The 928 source images reduce to 491 unique retained images after 437 exact-copy exclusions. The fixed split contains 295 training, 98 validation and 98 test images. Conservative groups connect printed IDs, identical model inputs and accepted near-copy links. Printed IDs are unverified patient proxies, and labels have not been clinically adjudicated. Only six MI images occur in each development-validation/test partition. The source is [Mendeley v2](https://data.mendeley.com/datasets/gwbz3fsgp8/2), attributed to Ali Haider Khan and Muzammil Hussain; retain `Khan2021MendeleyV2` in the bibliography.

All three runs use the same 1×128×256 grayscale-ink page crop, bilinear resizing and saved group split (`image_dedup_8e04bd56af2f`, `grouped_cf5a867a99c3`). Fixed cropping clips tall waveform peaks in one inspected training case and resizing changes aspect ratio. A separate one-case preview was never adopted. These are spatial images, not calibrated waveform samples. See [dataset audit](../docs/dataset_audit.md) and [input review](../docs/input_representation_review.md).

**PTB-XL, five overlapping diagnostic labels:** NORM, MI, STTC, CD and HYP. Custom clean pages were rendered from all 21,799 native 500 Hz, ten-second, twelve-lead records in v1.0.3. Classification includes 21,388 records with at least one mapped diagnostic superclass; 411 unmapped records remain excluded. All present diagnostic SCP keys are used, including likelihood zero. Official folds 1–8/9/10 give 17,084/2,146/2,158 ECGs from 14,823/1,917/1,877 patients, with disjoint patients. Cite both the [versioned release](https://physionet.org/content/ptb-xl/1.0.3/) (`Wagner2022PTBXL103`) and [dataset paper](https://pubmed.ncbi.nlm.nih.gov/32451379/) (`Wagner2020PTBXL`). The original paper's older release counts must not replace the local v1.0.3 counts.

The frozen PTB-XL input is 12×120×1120: twelve complete aligned lead-image rows and calibration pulses, header/footer removed, inverted grayscale and isotropic 2×2 maximum-ink pooling. All PTB-XL models share training-only global mean/std normalization. The CNN produces 70 columns with 96 image features per column. Numeric IDs, diagnoses, demographics and page-gain values are not predictors. Calibration/margin pixels remain visible. Rasterization, page gain and pooling can lose fine morphology. This custom rendering is not an ECG-Image-Kit-generated cohort. See [dataset card](../docs/ptbxl_image_dataset.md).

No metric below directly ranks a Mendeley model against a PTB-XL model. Accuracy is a single class decision on Mendeley; PTB-XL exact match requires five correct binary decisions, while label accuracy counts individual decisions. Macro AUROC is the prespecified primary PTB-XL ranking metric; five sigmoid outputs use a fixed threshold of 0.5 for decision metrics.

## Verified completed results

Mendeley values below are validation/test, respectively. AUROC is macro one-versus-rest; there is no claimed uncertainty interval from this one-seed comparison.

| Run | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro AUROC |
|---|---:|---:|---:|---:|---:|
| `lda_v1_seed42` | 0.6224 / 0.6224 | 0.5448 / 0.6032 | 0.5296 / 0.5637 | 0.6370 / 0.6423 | 0.8542 / 0.8771 |
| `cnn_v1_seed42` | 0.7143 / 0.7551 | 0.5091 / 0.5686 | 0.4607 / 0.5621 | 0.6493 / 0.7310 | 0.8176 / 0.9008 |
| `cnn_lstm_image_v1_seed42` | 0.7347 / 0.7143 | 0.6115 / 0.6315 | 0.6001 / 0.6020 | 0.7355 / 0.7168 | 0.8303 / 0.8944 |

PTB-XL values below are validation/test. Each row is a single fitted seed, not a repeated-run average.

| Run | Macro AUROC | Macro AP | Macro F1 | Micro F1 | Exact match | Label accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `ptbxl_cnn_lstm_image_v1_seed42` | 0.8816 / 0.8698 | 0.7134 / 0.7035 | 0.6171 / 0.6131 | 0.6839 / 0.6779 | 0.5200 / 0.5259 | 0.8507 / 0.8495 |
| `ptbxl_spatiotemporal_image_v1_seed42` | 0.8889 / 0.8803 | 0.7348 / 0.7325 | 0.6521 / 0.6409 | 0.7164 / 0.7085 | 0.5629 / 0.5616 | 0.8630 / 0.8608 |

| Run | Train macro F1 | Parameters | Epochs / selected | Recorded training seconds | Runner elapsed seconds |
|---|---:|---:|---:|---:|---:|
| LDA | 0.6740 | PCA + LDA, neural count not applicable | Not applicable | 0.23 | 1.21 |
| Mendeley CNN | 0.4360 | 40,212 | 30 / 25 | 43.70 | 46.47 |
| Mendeley CNN–LSTM | 0.6177 | 44,532 | 30 / 27 | 48.19 | 51.05 |
| PTB-XL CNN–LSTM | 0.6631 | 18,025 | 19 / 15 | 10,405.80 | 10,909.67 |
| PTB-XL spatiotemporal | 0.6865 | 17,161 | 20 / 16 | 10,535.26 | 10,957.27 |

Runtimes describe these executions, not a controlled speed benchmark. The spatiotemporal UTC wall span was 15,222.35 seconds, materially longer than its elapsed timer. System sleep/waits and timer definitions prevent attributing duration differences solely to architecture.

## Run 1 PCA plus LDA

**Motivation and hypothesis.** Establish how much label discrimination a small linear statistical model obtains from the frozen images, before assigning value to learned convolutional representations.

**Architecture and method.** Flatten 32,768 pixel features; fit centered randomized PCA with 32 components on the 295 training images; fit least-squares shrinkage LDA (`shrinkage=auto`, uniform four-class priors). Seed 42, no hyperparameter search, no learned preprocessing from validation/test, and no per-pixel variance scaling. Training-only PCA fit and uniform priors were freshly confirmed in the saved pipeline.

**Results and interpretation.** Test accuracy 0.6224, macro F1 0.5637 and balanced accuracy 0.6032. Test correct counts are Normal 21/28, MI 3/6, Abnormal 27/47 and History_MI 10/17. LDA has lower aggregate accuracy than CNN but better History_MI recall and a slightly higher macro F1. It is a useful reference against majority-class-dominated accuracy.

**Limits and next experiment.** One seed/split, unverified clinical labels, small MI support and coarse/cropped inputs limit inference. Add a separately specified PTB-XL multilabel PCA plus binary-LDA baseline on the same patient folds, with all feature fitting restricted to training. It has not been run. Evidence: [saved run](../results/lda/lda_v1_seed42/run.json), [frozen protocol](../experiments/configs/baseline_v1.json).

## Run 2 Mendeley image CNN

**Motivation and hypothesis.** Test whether local learned image features improve the first statistical baseline under the same cohort and pixels.

**Architecture and method.** Three Conv2d stages, widths 16/32/64 with GroupNorm, ReLU and max pooling; adaptive 2×4 spatial pooling; 512→32→4 classifier with dropout 0.25. Training-only global normalization and inverse-frequency class weights; AdamW, learning rate 0.001, weight decay 0.0001, batch 16, maximum 30 epochs, patience 7. Select strict minimum weighted validation cross-entropy, retaining earlier exact ties. All 30 epochs ran; epoch 25 was restored.

**Results and interpretation.** Test accuracy rises to 0.7551, but macro F1 is 0.5621 and balanced accuracy 0.5686. Test correct counts are 28/28, 1/6, 41/47 and 4/17 in class order. Validation recalls are 1/6 for MI and 0/17 for History_MI. The selected model also learns these minorities poorly in training, so its failure cannot be described simply as excellent training performance followed by overfitting.

**Limits and next experiment.** Overall accuracy obscures class imbalance; neither classification error nor mixed printed-ID labels proves a wrong diagnosis. A matched PTB-XL CNN-only head and development-only imbalance study remain useful. Do not change this saved checkpoint or retune its test results. Evidence: [saved run](../results/cnn/cnn_v1_seed42/run.json), [error review](../docs/minority_error_review.md).

## Run 3 Mendeley spatial CNN–LSTM

**Motivation and hypothesis.** Combine left-to-right spatial features to improve balanced class recognition while retaining the frozen image inputs.

**Architecture and method.** The three convolution stages produce 64×8×16 features; adaptive pooling gives two vertical bands and sixteen horizontal columns. Each column has 128 features; one unidirectional LSTM has hidden size 32. Mean sequence output, dropout 0.25 and a linear layer yield four logits. AdamW, training-only class weights/normalization, batch 16, maximum 30 epochs, patience 7 and gradient clipping at 1. Select validation macro F1, then balanced accuracy, then negative weighted loss; retain earliest exact tie. All 30 epochs ran; epoch 27 was selected.

**Results and interpretation.** Validation macro F1 0.6001 and test macro F1 0.6020 exceed the other Mendeley runs. Test balanced accuracy is 0.6315, while accuracy drops from CNN's 0.7551 to 0.7143. Correct counts are 26/28, 3/6, 35/47 and 6/17. The tradeoff favors more balanced classification on this split; LDA still detects more History_MI examples.

**Limits and next experiment.** This is a spatial sequence, not validated ECG time. Head, pooling, gradient clipping and checkpoint-selection rule differ from CNN, preventing an isolated LSTM-effect claim. Test outcomes were already observed before this experiment. Future controlled work should align selection rules and use development-only repeated seeds rather than claim superiority from the six MI test cases. Evidence: [protocol](../experiments/configs/cnn_lstm_image_v1.json), [report](../docs/cnn_lstm_report.md).

## Run 4 PTB-XL lead-image CNN–LSTM

**Motivation and hypothesis.** Establish a larger image benchmark with explicit patient identities and native overlapping diagnostic labels, preserving the complete ten-second interval in every lead.

**Architecture and method.** Depthwise lead filtering and pointwise lead mixing feed three small spatial convolution stages. Pooling yields 70×96 image-feature columns. A one-layer unidirectional LSTM with 32 hidden units, mean pooling, dropout 0.2 and five logits completes the 18,025-parameter model. Weights are trained from scratch. Unweighted BCEWithLogitsLoss, AdamW at 0.001 with weight decay 0.0001, cosine decay to 0.00001, clipping at 1, batch 16, four CPU threads, seed 42, maximum 20 epochs and patience 4. Selection maximizes validation macro AUROC, then macro AP, then minimizes probability BCE, with earliest exact ties. Threshold is fixed at 0.5; no augmentation or class weighting. Nineteen epochs ran and epoch 15 was selected.

**Results and interpretation.** Test macro AUROC 0.8698, macro AP 0.7035 and macro F1 0.6131 establish the first formal baseline for this task. Test sensitivity is 0.8764 NORM, 0.4327 MI, 0.5739 STTC, 0.4899 CD and 0.3244 HYP. HYP misses 177/262 positives. This is performance on one clean rendering of PTB-XL, not evidence of generalization to paper scans or a replication of the historical 600-image experiment.

**Limits and next experiment.** One seed, one rendering, label uncertainty and threshold-dependent sensitivity remain. Replace the recurrent head under unchanged inputs and training rules; this was executed in Run 5. Complete matched PTB-XL CNN-only and LDA baselines before claiming a general architecture ranking. Evidence: [protocol](../docs/ptbxl_cnn_lstm_protocol.md), [report](../docs/ptbxl_cnn_lstm_report.md).

## Run 5 PTB-XL small spatiotemporal network

**Motivation and hypothesis.** Test a compact residual dilated-convolution head as an alternative to the LSTM on identical lead-image features. The sequence-modeling rationale comes from [Bai et al.](https://arxiv.org/abs/1803.01271) (`Bai2018TCN`); the project model is not an exact reproduction of their architecture.

**Architecture and method.** Reuse the same spatial encoder architecture, trained from scratch. Project the 96 feature channels to 32, then apply two residual blocks, each with two kernel-3 left-padded convolutions, at dilation 1 and 2 respectively. Each convolution has per-position channel LayerNorm, ReLU and dropout. Residual sums use ReLU; whole-sequence mean, dropout 0.2 and a linear head yield five logits. The head sees 13 feature positions before global pooling; the full image classifier remains offline. The 17,161-parameter model uses Run 4's fitting/selection/threshold rules. Twenty epochs ran; epoch 16 was restored.

**Results and interpretation.** Test macro AUROC 0.8803 versus 0.8698, macro AP 0.7325 versus 0.7035, and macro F1 0.6409 versus 0.6131. Validation macro AUROC also increases, from 0.8816 to 0.8889. The saved paired patient-bootstrap AUROC difference is +0.01046, 95% interval 0.00447–0.01674. Test MI recall rises from 0.4327 to 0.5618; HYP recall falls from 0.3244 to 0.2786, with 189/262 HYP positives missed. The provisional stronger overall result has a clinically relevant minority-label weakness.

**Limits and next experiment.** These conditional intervals use 500 patient-cluster resamples of saved predictions, seed 2024; they exclude training-seed variation, adaptive choices and external-domain variation. Test exposure predates this specification, making comparison exploratory. Capacity and dropout placement differ, so improvement cannot be attributed solely to temporal convolution. Next: one frozen hybrid ViT comparison, then prespecified repeated-seed and matched-head/position ablations. Evidence: [protocol](../docs/ptbxl_spatiotemporal_protocol.md), [report](../docs/ptbxl_spatiotemporal_report.md), [paired comparison](../results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.json).

## Run 6 PTB-XL hybrid ViT under implementation

**Run ID:** `ptbxl_hybrid_vit_image_v1_seed42`. **Status at preparation of these notes: implementation in progress; not fitted; no results available.** Update this section only from actual saved and verified run artifacts.

**Motivation and hypothesis.** Learned interactions between distant image-feature columns may improve the existing local convolutional sequence readout. The hypothesis is uncertain; poorer ranking, lower sensitivity or no useful gain are valid outcomes. The [original ViT paper, section 3.1, Hybrid Architecture](https://arxiv.org/html/2010.11929v2#S3.SS1) explicitly describes feeding CNN feature-map embeddings to a Transformer (`Dosovitskiy2021ViT`). This project uses that hybrid principle, with a much smaller custom head and mean readout.

**Specified architecture.** `LeadImageHybridViT`, 23,881 parameters: the existing spatial CNN → 70×96 image-feature columns → linear projection to width 32 plus learned positions → two independently initialized pre-norm TransformerEncoderLayers, four attention heads, feed-forward width 64 and GELU → LayerNorm, mean pooling, dropout and five logits. It has no class token, no pretrained weights and no raw-patch ViT backbone. Tokens denote spatial image columns including calibration/margins. The parameter count must be reconfirmed in the final frozen source/probe before fitting.

**Specified method.** Reuse PTB-XL folds, frozen 12×120×1120 inputs, normalization, five labels, seed 42, batch 16, maximum 20 epochs, patience 4, AdamW/loss/scheduler/clipping, validation selection and threshold 0.5. Freeze source, configuration and preservation records before training. Compare saved aligned predictions with both PTB-XL baselines. Increased capacity and different dropout/position mechanisms are confounders; this is a practical model comparison rather than proof of attention's isolated effect.

**Pending results.** Training start/completion and verification: **PENDING**. Epochs/selected epoch: **PENDING**. Training/validation/test macro AUROC, AP, F1 and per-label outcomes: **PENDING**. Checkpoint, predictions, source hashes, uncertainty, runtime and comparison paths: **PENDING**. No improvement claim is permitted before these fields are populated from the completed run.

**Pending interpretation and next experiment.** Report gains and regressions under the frozen metrics. Decide further development using development evidence. Prespecify at least three seeds and a controlled position/attention ablation if resources permit; never choose a seed, threshold or architecture from test results.

## Manuscript boundaries and unresolved work

The current evidence supports a reproducible, exploratory benchmark of compact classifiers on two separately defined image tasks. It does not establish a novel architecture, a universal best ECG model, external transfer, causal physiological timing, calibrated digitization or clinical readiness. Additional renderings of an existing patient's signals are not independent patients. The historical manuscript's claims and conflicting archived figures cannot be copied into these result tables.

LSM and the remembered “Portal convolutions” method remain unresolved. ARC is a literature lead only. Obtain the exact full method name/reference before implementing either placeholder. The [historical audit](../experiments/reviews/2026-11-01/historical_material_audit.md) records the evidence and bibliography corrections. The [publication plan](PUBLICATION_PLAN.md) sets the remaining work and claim boundaries for 1 December.


Architecture figure: [PNG](../results/publication/architecture_comparison_v1/ptbxl_architectures.png) and [SVG](../results/publication/architecture_comparison_v1/ptbxl_architectures.svg), generated from the frozen specifications and visually checked. This diagram shows model structure; it does not imply measured superiority.
