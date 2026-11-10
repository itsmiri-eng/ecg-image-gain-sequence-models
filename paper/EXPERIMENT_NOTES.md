# ECG image classification experiment notes

Status updated 1 November 2026 (evening): seven completed formal runs (three Mendeley, four PTB-XL); the PTB-XL CNN-only baseline and repeated-seed study are running. These notes use the fresh saved-artifact audit, not the older manuscript's reported experiments. Historical motivations are reconstructed from protocols and code; this document does not retrospectively preregister them.

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
| `ptbxl_hybrid_vit_image_v1_seed42` | 0.9016 / 0.8902 | 0.7592 / 0.7519 | 0.6840 / 0.6717 | 0.7286 / 0.7140 | 0.5811 / 0.5718 | 0.8643 / 0.8579 |
| `ptbxl_v2_lda_seed42` (PCA + LDA) | 0.8006 / 0.7883 | 0.5913 / 0.5695 | 0.4220 / 0.4034 | see run.json | 0.3635 / 0.3596 | see run.json |

| Run | Train macro F1 | Parameters | Epochs / selected | Recorded training seconds | Runner elapsed seconds |
|---|---:|---:|---:|---:|---:|
| LDA | 0.6740 | PCA + LDA, neural count not applicable | Not applicable | 0.23 | 1.21 |
| Mendeley CNN | 0.4360 | 40,212 | 30 / 25 | 43.70 | 46.47 |
| Mendeley CNN–LSTM | 0.6177 | 44,532 | 30 / 27 | 48.19 | 51.05 |
| PTB-XL CNN–LSTM | 0.6631 | 18,025 | 19 / 15 | 10,405.80 | 10,909.67 |
| PTB-XL spatiotemporal | 0.6865 | 17,161 | 20 / 16 | 10,535.26 | 10,957.27 |
| PTB-XL hybrid ViT | 0.7406 | 23,881 | 20 / 19 | 11,858.41 | 12,376.70 |
| PTB-XL PCA + LDA | 0.4477 | PCA 256 × 25,200 + 5 LDAs | Not applicable | 27.08 | 127.74 |

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

## Run 6 PTB-XL hybrid ViT (CNN features + small Transformer)

**Run ID:** `ptbxl_hybrid_vit_image_v1_seed42`. Completed 1 November 2026, 14:55 UTC; saved-artifact verification passed ([verification](../results/ptbxl_hybrid_vit/ptbxl_hybrid_vit_image_v1_seed42/verification.json), [report](../docs/ptbxl_hybrid_vit_report.md)).

**Model.** `LeadImageHybridViT`, 23,881 parameters: the shared spatial CNN → 70×96 image-feature columns → linear projection to width 32 plus learned positions → two independently initialized pre-norm Transformer encoder layers (4 heads, FFN 64, GELU) → LayerNorm, mean pooling, dropout 0.2, five logits. No class token, no pretraining, no raw-patch ViT backbone. This follows the hybrid principle of [Dosovitskiy et al., §3.1](https://arxiv.org/html/2010.11929v2#S3.SS1) (`Dosovitskiy2021ViT`) at a very small scale; it is not a novel architecture.

**Motivation.** The supervisor's progression ends with a ViT. A full-size ViT is not justified by 17k training images on a CPU-only laptop, so the protocol tests the question a ViT asks — does global attention across positions help? — on the same CNN features as Runs 4–5.

**Hypothesis.** Global attention between distant feature columns (e.g. relating different beats or different parts of the page) improves on local recurrent/convolutional readouts. A null or negative result was explicitly allowed.

**Method.** Identical to Runs 4–5: same inputs, folds, training-only normalization, seed 42, AdamW lr 0.001, weight decay 1e-4, cosine to 1e-5, batch 16, gradient clip 1, unweighted BCE, max 20 epochs, patience 4, validation selection on (macro AUROC, macro AP, −BCE), fixed threshold 0.5, one test inference after freezing. CPU, 4 threads. 20 epochs ran; epoch 19 selected. Runtime 206 min.

**Results.**

| Metric | Train | Validation | Test [95% patient CI] |
|---|---:|---:|---:|
| Macro AUROC | 0.9259 | 0.9016 | 0.8902 [0.8809, 0.9000] |
| Macro AP | 0.8224 | 0.7592 | 0.7519 [0.7321, 0.7710] |
| Macro F1 | 0.7406 | 0.6840 | 0.6717 [0.6523, 0.6895] |
| Exact match | 0.6351 | 0.5811 | 0.5718 |

Test sensitivity / specificity per label: NORM 0.885/0.788, MI 0.498/0.937, STTC 0.701/0.918, CD 0.573/0.959, HYP 0.531/0.948. Paired test differences (500 shared patient resamples): ViT − TCN macro AUROC +0.0099 [+0.0043, +0.0155]; ViT − CNN–LSTM +0.0204 [+0.0140, +0.0272].

**Interpretation.** Did it improve? Yes on every aggregate metric, on validation and test, by about 0.01 macro AUROC over the TCN. Which labels improved? HYP sensitivity rose sharply (0.279 → 0.531; 139 vs 73 of 262 detected), STTC (0.633 → 0.701) and CD (0.544 → 0.573). Which worsened? MI sensitivity fell versus TCN (0.562 → 0.498, 274 vs 309 of 550), though it remains above CNN–LSTM; NORM specificity fell (0.818 → 0.788). Overfitting: train–validation macro AUROC gap 0.024, and validation was still improving at epoch 19 of 20, so the model looks budget-limited rather than overfit. Complexity: +6.7k parameters over the TCN for +0.01 AUROC; still tiny in absolute terms. Robustness: single seed; the patient bootstrap does not measure seed variation, so the gain is not yet shown to be robust. Hypothesis: supported in this single-seed exploratory comparison; not yet established.

**Limitations.** Capacity, dropout topology and positional embeddings differ from the TCN, so this is not an attention-only ablation. The test fold had been observed before the model was specified. Clean renderings only.

**Next step.** Repeated seeds 43/44 under the bit-identical v2 trainer (running), then a positional-embedding or capacity-matched ablation.

## Run 7 PTB-XL PCA + LDA baseline

**Run ID:** `ptbxl_v2_lda_seed42`. Completed 1 November 2026 (runtime 128 s, of which 27 s fitting). [run.json](../results/ptbxl_v2/ptbxl_v2_lda_seed42/run.json), [config](../experiments/configs/ptbxl_v2_lda.json), code `src/run_ptbxl_lda.py`.

**Model.** Statistical baseline matching the supervisor's step 1 on the main dataset. Each cached 12×120×1120 input is reduced by 8×8 area averaging to 12×15×140 = 25,200 features (a stated representation difference from the networks, needed because PCA on 1.6 M pixels per ECG does not fit in 8 GB of memory). Randomized PCA (fitted on training images only) → five independent binary shrinkage LDAs (lsqr solver, Ledoit–Wolf shrinkage, empirical training priors, matching the unweighted networks). PCA dimension chosen from the prespecified grid {16, 32, 64, 128, 256} by validation macro AUROC.

**Motivation / hypothesis.** How much of the label information is linearly accessible from coarse pixels? Expected to be clearly below the CNNs.

**Results.** Validation AUROC by PCA size: 0.695 (16), 0.701 (32), 0.717 (64), 0.788 (128), 0.801 (256) → 256 selected (66.7% variance explained). Test macro AUROC 0.7883 [0.7757, 0.8013], macro AP 0.5695, macro F1 0.4034, exact match 0.3596. Train/validation/test macro AUROC 0.814 / 0.801 / 0.788 (little overfitting). Test sensitivity: NORM 0.709, MI 0.267, STTC 0.167, CD 0.290, HYP 0.153, with high specificity (0.76–0.99): at threshold 0.5 the linear model rarely calls a minority label. Per-label AUROC 0.74–0.82.

**Interpretation.** All neural models exceed it by 0.08–0.10 macro AUROC (CNN–LSTM − LDA +0.0815 [+0.0721, +0.0911]). Learned local features matter for this task. The F1 gap is larger than the AUROC gap because of threshold behaviour at low prevalence.

**Limitations.** The selected PCA size is the grid's upper edge and validation AUROC was still rising, so this baseline is conservative; a validation-only follow-up with 512/1024 components is planned and must be reported as such. Coarser input than the networks.

## Run 7b PTB-XL PCA + LDA, wider PCA grid (follow-up)

**Run ID:** `ptbxl_v2_lda_wide_seed42` ([config](../experiments/configs/ptbxl_v2_lda_wide.json)). Same pipeline as Run 7 with grid {256, 512, 1024}. Specified after Run 7 selected its grid edge; Run 7's test result had already been seen, so this is a disclosed follow-up, reported alongside Run 7, not a replacement chosen for its test score. Selection used validation only.

**Results.** Validation macro AUROC 0.8010 (256), **0.8056 (512, selected)**, 0.8018 (1024): an interior optimum, so the baseline is now adequately tuned on its main hyperparameter. Test macro AUROC 0.7939 [0.7822, 0.8065], macro AP 0.5762, macro F1 0.4269, exact match 0.3726. Test sensitivity NORM 0.723, MI 0.258, STTC 0.190, CD 0.327, HYP 0.195. Train/validation macro AUROC 0.833 / 0.806. Runtime 113 s.

**Interpretation.** Widening the grid changed test macro AUROC by only +0.006; the gap to the neural models (CNN–LSTM − wide LDA +0.0759 [+0.0668, +0.0843]) is not an artefact of an under-tuned baseline. This run is the statistical reference in the master table.

## Run 8 PTB-XL CNN-only baseline (running)

**Run IDs:** `ptbxl_v2_cnn_pool_seed42`, `_seed43`, `_seed44`. Started 1 November ~15:00 UTC. Code `src/models/ptbxl_cnn_pool.py`, trainer `src/training/ptbxl_trainer.py`, config `experiments/configs/ptbxl_v2_cnn_pool.json`.

**Model.** The identical shared CNN encoder (same layers and construction order as Runs 4–6), pooled to 70×96, then the mean over the 70 columns → dropout 0.2 → Linear(96, 5). 1,705 parameters. No sequence module.

**Motivation / hypothesis.** The supervisor's step 4 on the main dataset, and the direct "no sequence head" reference for Runs 4–6: if the sequence heads matter, all three should beat this model; if this model is close, the gains come mainly from the CNN.

**Method.** Same training protocol as Runs 4–6 under the shared v2 trainer. **Results: pending; do not cite until `run.json` exists.**

**Known limitation.** 10–14× fewer parameters than the sequence models; a capacity-matched pooled head is a planned ablation.

## Seed study (running)

Prespecified seeds 42, 43, 44 for each PTB-XL network, all reported. Seed 42 of the CNN–LSTM, TCN and ViT is the existing v1 run: on real training data the v2 shared trainer reproduces each v1 trainer's weights exactly ([check](../experiments/reviews/2026-11-01/trainer_equivalence_check.json)). Outcome summaries will be mean ± SD over seeds plus per-seed values; the patient bootstrap remains a separate, conditional measure of test-sampling uncertainty.

## Error analysis 1 (seed 42; saved predictions only)

**Method.** `python -m src.analyze_errors_ptbxl` → [summary](../results/error_analysis/ptbxl_seed42/summary.md). Sensitivity at threshold 0.5 for MI, HYP, CD and STTC, stratified by rendered page gain, number of co-occurring labels, age band, PTB-XL sex code, and noise/drift annotation; false-negative overlap across models. Validation is the interpretive partition; test values descriptive. No tests of significance; small strata are noisy.

**Findings.**
1. **Page gain carries hypertrophy information.** The renderer reduces page gain for high-amplitude ECGs. HYP training prevalence is 2.1% at 10 mm/mV, 6.3% at 5, 30.8% at 2.5 and 38.0% at ≤1.25. Gain alone (no image) gives HYP AUROC 0.741 (validation) / 0.719 (test). On validation pages at standard 5–10 mm/mV gain (89 HYP positives) sensitivity is 0.02–0.04 for LDA/CNN–LSTM/TCN and 0.19 for the ViT; at 2.5 mm/mV it is 0.28–0.67. However, within a single gain level the models still rank HYP well above chance (validation HYP AUROC at 5 mm/mV: LDA 0.767, CNN–LSTM 0.808, TCN 0.771, ViT 0.835; at 2.5 mm/mV 0.764–0.819), and the ViT is best in both strata. So the models use waveform content beyond gain; the near-zero sensitivity at standard gain mainly reflects the fixed 0.5 threshold where HYP prevalence is ~6%. Gain still conveys amplitude in a way that fixed-gain printouts (10 mm/mV, sometimes 5) do not, so between-stratum performance may not transfer.
2. **Shared hard cases.** On validation, 154/540 MI and 120/268 HYP positives are missed by every model; 443 MI and 234 HYP are missed by at least one. Errors are only partly model-specific.
3. **Isolated vs co-occurring labels.** Single-label MI and HYP are harder than co-occurring ones (validation MI 0.48–0.49 vs 0.59–0.61 for TCN/ViT; HYP 0.06–0.38 vs 0.37–0.56).
4. **Demographics.** No large sensitivity difference between PTB-XL sex codes for the neural models (within ≈0.05). Patients under 50 have lower sensitivity for every label and model (small strata: 32–60 positives).

**Implication.** Report HYP results stratified by gain (AUROC within stratum and sensitivity); treat gain-controlled rendering (single fixed gain, or gain normalized out) as a priority robustness experiment before any claim about hypertrophy detection from paper ECGs.

## Gain-controlled rendering experiment (started 1 November 2026)

**Protocol** (written before any fixed-gain result): [docs/ptbxl_fixed_gain_protocol.md](../docs/ptbxl_fixed_gain_protocol.md). **Motivation:** error analysis 1 showed that the adaptive page gain carries hypertrophy information. **Hypotheses:** H1 adaptive-gain models lose performance on fixed-gain pages; H2a with fixed gain HYP discrimination is maintained or improves (amplitude becomes directly comparable); H2b MI/STTC may worsen (small deflections drawn at half size).

**Dataset.** `ptbxl_images_fixedgain2p5_v1` (`src/ptbxl_fixed_gain.py`): every page at 2.5 mm/mV, otherwise identical to v1; same 21,388 ECGs, labels and folds. 154 pages (121/17/16 train/val/test) have points clipped at the row edge (ECGs > 6 mV); none excluded. Verification: cohort identical to v1, all images hash-checked, and all 5,135 pages already at 2.5 mm/mV in v1 are pixel-identical. Build 11 min; 1.0 GB.

**Q1 transfer (no retraining, validation only)** — [summary](../results/fixed_gain/transfer_validation/summary.md). Macro AUROC change, fixed − original pages, paired patient bootstrap: LDA −0.0494 [−0.0596, −0.0391]; CNN–LSTM −0.0099 [−0.0147, −0.0053]; TCN −0.0037 [−0.0082, +0.0008]; ViT −0.0048 [−0.0088, −0.0006]. Predictions on unchanged pages were identical (check passed). On re-drawn pages (formerly 5/10 mm/mV, 1,586 ECGs), ranking barely changed for the networks but decisions shifted: NORM sensitivity fell (ViT 0.914 → 0.884; TCN 0.897 → 0.825) and HYP sensitivity rose (ViT 0.191 → 0.281; TCN 0.022 → 0.169). *Reading:* H1 is supported weakly for the networks (small AUROC loss) and strongly for LDA. The decision shift fits the networks having learned the 2.5-mm/mV layout as a prior for hypertrophy (HYP training prevalence 31% at that gain), while ranking within a page relies mostly on waveform content.

**Q2 retraining, PCA + LDA** (`ptbxl_v2_fg_lda_seed42`, grid 256/512/1024 → 512). Validation macro AUROC 0.7895 vs 0.8056 adaptive; test 0.7811 [single inference] vs 0.7939. Per-label validation AUROC fixed vs adaptive: HYP 0.853 vs 0.842 (up), NORM 0.806 vs 0.835, MI 0.760 vs 0.774, STTC 0.722 vs 0.763, CD 0.807 vs 0.814. For the linear baseline, H2a and H2b are both supported: consistent scale helps HYP, lower resolution hurts the small-deflection labels.

**Q2 retraining, networks (queued).** `ptbxl_v2_fg_{hybrid_vit,temporal_conv,cnn_lstm,cnn_pool}_seed42` start automatically when the current seed queues finish (expected 2–3 November). **Do not cite until `run.json` exists.**

## Results update, 7 November 2026 (five seeds, capacity-matched head, external test)

This section supersedes the "running"/"pending" notes above for Run 8, the seed study and the fixed-gain retraining. All numbers come from saved predictions. The protocols were frozen before the runs or data they govern: `docs/protocol_amendment_2026-11-03.md` and `docs/external_test_plan.md`, with the freeze recorded in `experiments/external/plan_freeze.json`.

**Runs.**
- 21 amendment runs were completed on the PC GPU between 5 and 7 Nov: seeds 45/46 for the four networks on adaptive and fixed gain, and the capacity-matched CNN-only head (`cnn_pool_wide`, 17,065 parameters) for seeds 42–46.
- Interrupted runs were moved aside unedited, with the causes documented (Windows sign-outs), in `experiments/abandoned_runs/2026-11-0{3,5,6}_*`. None had reached test inference.

**PTB-XL, five seeds** ([summary](../results/comparisons/ptbxl_heads_hierarchical/summary.md)). Hierarchical seed × patient bootstrap, Holm over 10 pairs. Test is exploratory.

| Model | Test macro AUROC (mean ± SD) |
|---|---|
| CNN-only | 0.830 ± 0.004 |
| CNN-only, capacity-matched | 0.865 ± 0.005 |
| CNN-LSTM | 0.877 ± 0.007 |
| Temporal conv | 0.879 ± 0.003 |
| Hybrid ViT | 0.886 ± 0.005 |

- **Head capacity alone explains about 0.035 of the CNN-only gap:** wide minus CNN-only +0.035 [0.029, 0.042].
- **Mixing information along the row adds a further 0.011–0.020 beyond capacity.** All three sequence heads beat the capacity-matched head after Holm, on test, on validation and in the GPU-only sensitivity set.
- **The sequence heads cannot be separated on PTB-XL test.** ViT − temporal conv +0.006 [−0.001, 0.013]; ViT − LSTM +0.009, Holm p 0.054. On validation, ViT > temporal conv reaches significance (+0.008, Holm p 0.03).
- **The seed-42 claim of a stepwise 0.01 gain per head does not hold.**

**Fixed gain, matched GPU seeds 43–46** ([summary](../results/fixed_gain/retrained_comparison/summary.md)). Mean fixed − adaptive test macro AUROC:

| Model | Fixed − adaptive |
|---|---|
| CNN-only | +0.033 |
| Temporal conv | +0.016 |
| CNN-LSTM | +0.012 |
| ViT | +0.005 |

Validation shows the same order. No architecture is worse with fixed gain.

**External confirmatory test** ([Georgia](../results/external/georgia/summary.md), [Chapman-Shaoxing + Ningbo](../results/external/china/summary.md)).
- **Data.** Each of 52 runs was inferred once per dataset. Every reloaded model first reproduced its saved PTB-XL validation probabilities (max difference 4.3e-4, the Mac-CPU ViT; networks ≤ 7e-7).
- **Georgia (USA).** 6,047 of 10,344 records were included. Labels STTC, CD and HYP are eligible: MI has 7 positives and NORM is unmappable.
- **China.** 9,004 of 45,152 records were included. MI, STTC, CD and HYP are eligible. 2 Ningbo records were excluded because their files fail PhysioNet's published checksum (dated amendment).
- **Bootstrap.** Records are resampled, since there are no patient IDs.

Mean external macro AUROC, adaptive / fixed gain:

| Model | Georgia | China |
|---|---|---|
| CNN-only | 0.652 / 0.707 | 0.717 / 0.768 |
| Wide CNN-only | 0.700 | 0.777 |
| CNN-LSTM | 0.723 / 0.738 | 0.799 / 0.817 |
| Temporal conv | 0.707 / 0.738 | 0.811 / 0.824 |
| ViT | 0.730 / 0.737 | 0.815 / 0.819 |
| PCA + LDA | 0.666 | 0.733 |
| Signal ResNet1D | 0.788 | 0.856 |

1. **Every model loses performance out of distribution.** Georgia: −0.12 to −0.17 macro AUROC against PTB-XL test on the same labels. China: −0.03 to −0.10. Part of this comes from the label mapping, which is conservative: T-wave-abnormal and Q-wave codes fall below the 90% rule.
2. **H1 (head ordering) largely replicates.**
   - In both datasets, CNN-only < wide CNN-only < {CNN-LSTM, ViT}, all significant after Holm.
   - ViT ≥ the other image heads in both datasets.
   - Temporal conv is unstable. In Georgia it is no better than the wide head (+0.007, n.s.) and below CNN-LSTM (−0.016) and ViT (−0.023). In China it is tied with ViT.
3. **H2 (fixed gain) is confirmed.**
   - Fixed-gain training improves macro and HYP AUROC for every architecture in both external sets, all Holm-significant except ViT macro AUROC in China (+0.004, n.s.).
   - The effect is largest for CNN-only (Georgia +0.055 macro, +0.129 HYP) and smallest for ViT. This is the same order as on PTB-XL, now on data never seen before.
   - With fixed gain, the three sequence heads are equal externally (Georgia 0.737–0.738; China 0.817–0.824).
4. **H3: the image representation still costs performance externally.** The signal model is best everywhere: about +0.05 over the best image model in Georgia and +0.03 in China.

**Limitations specific to the external test.**
- NORM is not evaluated externally, and MI is not evaluated in Georgia.
- Labels come from a data-driven SNOMED → superclass mapping, not from cardiologist relabelling.
- The external data were rendered by the same synthetic renderer (clean pages), so this tests population, device and labelling shift, not real-paper artefacts.
- The 100 Hz resampling for the signal model differs from PTB-XL's own `records100`, so the signal comparison is secondary.

**Process notes.**
- The analysis step of the automated chain was skipped twice, because PowerShell treated warnings on stderr as failure. `analyze` was then run by hand. It reads saved predictions only, so no inference was repeated.
- `src/external_test.py` gained the checksum-exclusion status and patient-cluster resampling after the Georgia cache was built. The rendering functions were unchanged.
- Tests: 239 passed, 3 skipped.

## Manuscript boundaries and unresolved work

The current evidence supports a reproducible, exploratory benchmark of compact classifiers on two separately defined image tasks. It does not establish a novel architecture, a universal best ECG model, external transfer, causal physiological timing, calibrated digitization or clinical readiness. Additional renderings of an existing patient's signals are not independent patients. The historical manuscript's claims and conflicting archived figures cannot be copied into these result tables.

LSM and the remembered “Portal convolutions” method remain unresolved. ARC is a literature lead only. Obtain the exact full method name/reference before implementing either placeholder. The [historical audit](../experiments/reviews/2026-11-01/historical_material_audit.md) records the evidence and bibliography corrections. The [publication plan](PUBLICATION_PLAN.md) sets the remaining work and claim boundaries for 1 December.


Architecture figure: [PNG](../results/publication/architecture_comparison_v1/ptbxl_architectures.png) and [SVG](../results/publication/architecture_comparison_v1/ptbxl_architectures.svg), generated from the frozen specifications and visually checked. This diagram shows model structure; it does not imply measured superiority.
