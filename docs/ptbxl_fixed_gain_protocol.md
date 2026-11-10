# Gain-controlled rendering experiment — protocol (frozen 1 November 2026, before any fixed-gain result)

## Why

The v1 renderer halves the page gain (10 → 5 → 2.5 … mm/mV) whenever an ECG's largest deflection would leave its row band. Gain is therefore a function of maximum amplitude, and high voltage is a hypertrophy criterion: in training data HYP prevalence is 2.1% at 10 mm/mV, 6.3% at 5, 30.8% at 2.5 and 38.0% at ≤1.25, and page gain alone gives HYP AUROC 0.741 (validation). The same page also shows different waveform sizes for the same voltage, depending on the ECG. We cannot tell from v1 results whether the models read amplitude from the waveform (physiology a cardiologist uses) or from the gain/calibration layout (a rendering artefact), nor how resolution loss would change results.

## Research questions and hypotheses (stated before results)

- **Q1 Transfer.** Do models trained on adaptive-gain pages keep their performance when the same validation ECGs are drawn at one fixed gain?
  *H1:* performance drops, concentrated in ECGs whose gain changed (those originally at 5 or 10 mm/mV, ≈75% of pages); pages already drawn at 2.5 mm/mV in v1 are re-drawn identically, so their predictions must not change (built-in check).
- **Q2 Retraining.** If every page uses one fixed gain, how do discrimination and per-label sensitivity compare with the adaptive-gain models (same architecture, seed, budget)?
  *H2a:* HYP AUROC is maintained or improves within the formerly 5-mm/mV stratum, because amplitude becomes directly comparable across pages.
  *H2b:* MI and STTC AUROC may decrease, because small deflections on formerly 5-mm/mV pages are drawn at half size (0.1 mV = 0.25 mm = 2 px before pooling).
  Either direction is a valid, reportable result.

## Rendering (dataset `ptbxl_images_fixedgain2p5_v1`)

- Identical to `ptbxl_images_v1` (same native 500 Hz WFDB waveforms, page size, layout, grid, lead order, 25 mm/s, calibration pulse, header/footer text style, OpenCV drawing) **except** the page gain is fixed at **2.5 mm/mV** for every ECG.
- 2.5 mm/mV is the largest power-of-two gain at which ≥98% of training pages (16,810/17,084 = 98.4%, from the existing training-only geometry audit) fit in the ±12-mm band. No validation or test information was used for this choice.
- Waveform points may use the full ±15-mm row. Points beyond the row boundary are clipped to it and counted per page (`clipped_point_count`); no ECG is excluded, so the cohort, labels and patient folds are exactly those of v1 (21,388 eligible; 17,084 / 2,146 / 2,158).
- Model input: the same crop, inversion and 2×2 max pooling (`preprocess_page`), so inputs are 12 × 120 × 1120. Training-only normalization is recomputed on the fixed-gain training images for retraining; transfer evaluation uses each model's own original normalization.
- Only the cached model inputs plus a small training-only preview of full pages are stored (≈1.3 GB).

## Analyses

1. **Transfer (Q1), no training:** seed-42 PCA+LDA (wide grid), CNN–LSTM, temporal-conv and hybrid ViT evaluated on fixed-gain **validation** pages. Outcomes: macro and per-label AUROC, sensitivity at 0.5, stratified by original gain. Test-fold transfer is deferred to the final analysis (one inference per model).
2. **Retraining (Q2):** PCA+LDA (grid {256, 512, 1024}) and the four networks (CNN pooled, CNN–LSTM, temporal conv, hybrid ViT), seed 42, protocol-v2 trainer and hyperparameters unchanged. Selection on validation; one test inference per run.
3. **Primary comparison:** fixed-gain-trained vs adaptive-gain-trained model of the same architecture and seed, on validation (paired patient bootstrap, 500 resamples): macro AUROC and HYP AUROC, overall and within original-gain strata (5 and 2.5 mm/mV). Test differences reported once, as exploratory (fold 10 previously observed).
4. Seeds 43/44 for the fixed-gain condition only if seed-42 differences are comparable to seed variability measured in the running seed study.

## What this can and cannot show

It can separate amplitude-on-waveform from gain-dependent layout cues on this synthetic benchmark and quantify the resolution trade-off. It cannot show performance on real printouts, which typically use 10 mm/mV with overlapping traces; that requires real or overlap-rendered images and is out of scope here.
