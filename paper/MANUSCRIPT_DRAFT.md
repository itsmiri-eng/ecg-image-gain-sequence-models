# Manuscript draft (working)

**Working title:** From linear discriminants to compact attention: a controlled, patient-separated comparison of ECG image classifiers on rendered PTB-XL

**Authors:** Miriam Istanboli, Mohamed Yousry Elgendi (order and affiliations to be confirmed)

**Status (1 November 2026):** skeleton with verified seed-42 results. Every number here comes from saved run artifacts listed in [EXPERIMENT_NOTES.md](EXPERIMENT_NOTES.md). Items marked **[PENDING]** must be filled only from completed runs. The older manuscript ("Domain Shift and Format Sensitivity…", archived ZIPs) is historical material; its numbers are not used here.

Rules for this draft: no novelty claim for any architecture; no clinical-use claim; no significance claim without the stated test; Mendeley and PTB-XL results are never ranked against each other.

---

## Abstract [draft; finalize after seed study]

**Background.** Many ECGs exist only as paper printouts or images. Image classifiers are often compared across different datasets, splits and training budgets, which makes architecture claims hard to interpret.
**Methods.** We rendered all 21,799 ten-second, 12-lead PTB-XL v1.0.3 records as clean paper-style images and classified five overlapping diagnostic superclasses (NORM, MI, STTC, CD, HYP) using the official patient-separated folds (17,084 / 2,146 / 2,158 ECGs). Following a fixed progression of increasing complexity — PCA + linear discriminant analysis, a CNN, CNN–LSTM, CNN + temporal convolution, and a compact CNN-feature Vision Transformer — every neural model shared the same image encoder, inputs, optimizer, budget, validation-only checkpoint selection and a single test evaluation. We report macro AUROC (primary), per-label sensitivity and specificity, patient-cluster bootstrap intervals and variation across three training seeds.
**Results.** [Seed-42 values; replace with seed means ± SD when available.] PCA + LDA reached test macro AUROC 0.794; CNN–LSTM 0.870, temporal convolution 0.880 and the compact hybrid ViT 0.890 (95% patient-bootstrap CI 0.881–0.900). The hybrid ViT was the only model to detect most hypertrophy cases (sensitivity 0.53 vs 0.28–0.32), whereas the temporal-convolution model had the highest MI sensitivity (0.56). CNN-only baseline and seed variability: **[PENDING]**.
**Conclusions.** [PENDING — depends on whether the ~0.01 AUROC steps exceed seed variability.]

## 1. Introduction

- Clinical motivation: paper/scanned ECG archives; image-based classification as an alternative or complement to digitization. Cite ECG-Image-Kit (`Shivashankara2024ECGImageKit`) and digitization work (`Krones2024Digitization`, `Wang2025RotatedECG`).
- Problem: architecture comparisons for ECG images are confounded by different datasets, label definitions, non-patient splits, training budgets and single-seed reporting.
- Research question: on one fixed, patient-separated image benchmark, how much does each step of increasing model complexity (linear → CNN → recurrent → temporal convolution → attention) change ranking performance, per-diagnosis sensitivity and stability?
- Contribution (to be confirmed by results): (i) an auditable, reproducible benchmark protocol (renderer, splits, hashes, one test inference per model); (ii) a matched comparison in which all networks share one encoder and training budget; (iii) per-label error and seed-variability analysis showing where complexity helps and where it does not. Not claimed: a new architecture, state-of-the-art performance, clinical readiness.

## 2. Related work / background

- PTB-XL dataset (`Wagner2020PTBXL`, `Wagner2022PTBXL103`) and its signal-based benchmark (`Strodthoff2021PTBXLBenchmark`). Note: signal-based results are a different input modality and must not be presented as directly comparable numbers.
- Sequence models: LSTM (`Hochreiter1997LSTM`), temporal convolutional networks (`Bai2018TCN`), Vision Transformer and its hybrid CNN-feature variant (`Dosovitskiy2021ViT`).
- ECG image generation and digitization (`Shivashankara2024ECGImageKit`, `Krones2024Digitization`, `Wang2025RotatedECG`).
- [TODO: targeted literature search for image-based PTB-XL / paper-ECG classification studies; verify every reference before citing.]

## 3. Dataset

- Source: PTB-XL v1.0.3, 21,799 records, 18,869 patients, 500 Hz, 10 s, 12 leads (CC BY 4.0).
- Labels: all diagnostic SCP codes mapped to five superclasses, union, no likelihood threshold; 411 records with no diagnostic superclass excluded → 21,388 eligible.
- Splits: official folds 1–8 / 9 / 10 → 17,084 / 2,146 / 2,158 ECGs; 14,823 / 1,917 / 1,877 patients; no patient overlap (verified).
- Label prevalence per split: [TODO table from the cache manifest].
- Audit: checksums of all source and rendered files; zero exact duplicate rendered images, including across partitions.
- Supplementary cohort (Mendeley paper-ECG images, `Khan2021MendeleyV2`): 928 files → 491 unique; four classes; reported separately as a preliminary case study because patient identity and labels cannot be verified.

## 4. Rendering and preprocessing

- Renderer: 2,240 × 3,024 px page, 12 stacked full-length lead rows, 25 mm/s, page-wide gain 10/2^k mm/mV, 1-mV calibration pulses, grid, no identifiers or labels printed (details in `docs/ptbxl_image_dataset.md`).
- Model input: 12 lead rows cropped, inverted, 2×2 max-ink pooling → 12 × 120 × 1120; one global mean/std from training pixels only; no augmentation.
- LDA input: additional 8×8 area averaging → 25,200 features (memory constraint; stated as a representation difference).
- Known information loss: rasterization, low page gain on high-amplitude ECGs, pooling.

## 5. Models

| Step | Model | Parameters | Sequence/readout |
|---|---|---:|---|
| 1 | PCA + LDA | PCA 512 components (validation-selected) + 5 binary shrinkage LDAs (`LedoitWolf2004`) | — |
| 4 | CNN (pooled) | 1,705 | Mean over 70 feature columns → linear |
| 5 | CNN–LSTM | 18,025 | 1-layer LSTM (32) → mean → linear |
| 6 | CNN + temporal convolution | 17,161 | 2 residual dilated causal conv blocks (32 ch) → mean → linear |
| 7 | CNN + compact ViT | 23,881 | Linear 96→32 + learned positions → 2 pre-norm Transformer layers (4 heads) → mean → linear |

Shared encoder (all networks): depthwise per-lead 5×5 convolution, pointwise lead mixing, two further depthwise-separable stages with GroupNorm/ReLU, adaptive pooling to 24 channels × 4 bands × 70 columns → 70 tokens of 96 features. Steps 2 (LSM) and 3 (supervisor's convolution method) are unresolved and not part of this comparison [confirm with supervisor].

## 6. Experimental setup

- Training: AdamW (lr 1e-3, weight decay 1e-4), cosine decay to 1e-5, batch 16, gradient clipping 1.0, unweighted BCE, dropout 0.2, max 20 epochs, early stopping patience 4, no class weighting/augmentation/pretraining. CPU, 4 threads, PyTorch 2.9.1.
- Selection: checkpoint maximizing validation macro AUROC (ties: macro AP, then −BCE). LDA: PCA size by validation macro AUROC; a first grid {16–256} selected its upper edge, so a disclosed follow-up grid {256, 512, 1024} was run (optimum 512, interior). Both runs are reported.
- Test policy: one inference per run after freezing; fold 10 had been observed for earlier models, so all comparisons are labelled exploratory.
- Seeds: 42, 43, 44 for every network; all reported. Seed-42 runs for the three sequence models predate the shared trainer; the shared trainer reproduces them bit-for-bit.
- Reproducibility: configs in `experiments/configs/`, code snapshot and SHA-256 hashes stored with each run; commands in README.

## 7. Evaluation metrics

Primary: macro AUROC over five labels. Secondary: macro AP, macro/micro/weighted F1, exact match (all five labels correct), label accuracy, and per label: sensitivity, specificity, precision, F1, AUROC, AP, confusion counts. Fixed threshold 0.5 (no threshold tuning). Uncertainty: 95% patient-cluster percentile bootstrap (500 resamples) for each model; paired patient bootstrap for differences. Seed variability: mean ± SD over three seeds. Planned statistical comparison across seeds: [decide before seeing seeds 43/44 — e.g. report paired differences per seed and their mean; no significance claim from three seeds alone].

## 8. Results

### 8.1 Main comparison (seed 42; test fold 10, 2,158 ECGs)

| Model | Macro AUROC [95% CI] | Macro AP | Macro F1 | Exact match | Val macro AUROC |
|---|---|---:|---:|---:|---:|
| PCA + LDA (PCA 512, validation-selected from 256–1024) | 0.7939 [0.7822, 0.8065] | 0.5762 | 0.4269 | 0.3726 | 0.8056 |
| CNN (pooled) | [PENDING] | | | | |
| CNN–LSTM | 0.8698 [0.8593, 0.8805] | 0.7035 | 0.6131 | 0.5259 | 0.8816 |
| CNN + temporal conv | 0.8803 [0.8705, 0.8904] | 0.7325 | 0.6409 | 0.5616 | 0.8889 |
| CNN + compact ViT | 0.8902 [0.8809, 0.9000] | 0.7519 | 0.6717 | 0.5718 | 0.9016 |

Paired differences (test, macro AUROC): CNN–LSTM − LDA +0.0759 [+0.0668, +0.0843]; TCN − CNN–LSTM +0.0105 [+0.0045, +0.0167]; ViT − TCN +0.0099 [+0.0043, +0.0155]. Source: `results/comparisons/ptbxl_master/`.

### 8.2 Per-label sensitivity (test, threshold 0.5)

| Model | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| PCA + LDA (PCA 512) | 0.723 | 0.258 | 0.190 | 0.327 | 0.195 |
| CNN–LSTM | 0.876 | 0.433 | 0.574 | 0.490 | 0.324 |
| CNN + temporal conv | 0.875 | 0.562 | 0.633 | 0.544 | 0.279 |
| CNN + compact ViT | 0.885 | 0.498 | 0.701 | 0.573 | 0.531 |

### 8.3 Seed variability [PENDING]
### 8.4 Training behaviour
Train / validation / test macro AUROC: LDA (PCA 512) 0.833 / 0.806 / 0.794; ViT 0.926 / 0.902 / 0.890. The ViT selected epoch 19 of 20 [add learning curves figure for all models].

## 9. Ablation studies [PLANNED]
1. Sequence head vs none: CNN (pooled) vs the three sequence models (running).
2. Capacity-matched pooled head (same parameter budget, no cross-column mixing) — separates capacity from sequence modelling.
3. ViT without positional embeddings — tests whether position information matters.
4. LDA PCA size beyond 256 — done (Run 7b): optimum 512, test +0.006 over PCA 256.

### 9.1 Gain-controlled rendering [in progress]
Protocol frozen before results (`docs/ptbxl_fixed_gain_protocol.md`). Same ECGs re-drawn at a fixed 2.5 mm/mV. Transfer of adaptive-gain models (validation): macro AUROC −0.004 to −0.010 for networks, −0.049 for LDA; decision-level shift (NORM sensitivity down, HYP up) on re-drawn pages. LDA retrained at fixed gain: validation macro AUROC 0.790 vs 0.806; HYP AUROC 0.853 vs 0.842; STTC 0.722 vs 0.763. Network retraining [PENDING].

## 10. Error analysis [first pass done, seed 42]
- Page gain as an information channel: HYP prevalence rises from 2% (10 mm/mV) to 31–38% (≤2.5 mm/mV); gain alone gives HYP AUROC 0.74 (validation) / 0.72 (test). HYP sensitivity at standard 5–10 mm/mV gain is 0.02–0.19 across models at threshold 0.5, yet within-gain HYP AUROC remains 0.77–0.84 (validation), i.e. the models discriminate beyond gain and the sensitivity loss is largely a threshold/prevalence effect. Source: `results/error_analysis/ptbxl_seed42/`.
- Shared failures: 154/540 validation MI and 120/268 HYP positives missed by every model.
- Isolated (single-label) MI/HYP harder than co-occurring; patients < 50 years lower sensitivity; no large difference by sex code.
- [PENDING: repeat with seeds 43/44; gain-controlled rendering experiment.]

## 11. Discussion [outline]
- Learned features vs linear: large gap (≈0.08–0.10 AUROC).
- Sequence readouts: consistent ~0.01 steps at seed 42; whether these exceed seed variation decides the main conclusion.
- Label-specific trade-offs: attention helps HYP and STTC; temporal convolution best for MI. A single macro score hides these.
- Comparison to signal-based PTB-XL literature only qualitatively (different modality).
- Practical: all models are tiny (< 25k parameters) and CPU-trainable.

## 12. Limitations
Clean synthetic renderings, not real scans/photos; adaptive page gain conveys amplitude (and thus hypertrophy) information in a way fixed-gain printouts do not; single dataset/population; label noise in PTB-XL annotations; test fold previously observed (exploratory); three seeds; capacity and dropout differ between heads; information loss from rasterization and pooling; no external validation; no clinical evaluation.

## 13. Conclusion [PENDING]

## Data and code availability [TODO]
PTB-XL is public (CC BY 4.0). Rendering code, configs, splits and analysis scripts to be released [repository location to be decided]. Large rendered images and checkpoints will not be committed to Git.

## Figures (planned)
1. Pipeline + rendering example. 2. Architecture diagram (exists: `results/publication/architecture_comparison_v1/`). 3. Main comparison with CIs (exists: `results/comparisons/ptbxl_master/master_comparison.png`). 4. Per-label sensitivity/specificity. 5. Seed dispersion. 6. Learning curves.
