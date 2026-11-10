# Secondary analyses: calibration, thresholds, saliency (frozen 2 Nov 2026, before computing any of them)

No model is trained. Every number comes from saved predictions or from frozen seed-42 checkpoints.
The fixed 0.5 threshold stays the primary decision rule; everything here is secondary. Test results are
exploratory because fold 10 was observed before.

## A. Calibration (`src/analyze_calibration_thresholds.py`)

- Runs: every completed PTB-XL run with saved predictions (image v1/v2 seeds, PCA + LDA, PCA + SVM,
  fixed-gain runs, signal-domain reference).
- Per run, partition (validation, test) and label: Brier score; expected calibration error (ECE) with
  10 equal-width probability bins, weighted by bin count; mean predicted probability vs prevalence.
  Macro values are the unweighted means over the five labels.
- Figure: reliability diagrams (test, seed 42) for PCA + LDA, CNN-only, CNN-LSTM, temporal conv, hybrid
  ViT and the signal reference.

## B. Validation-chosen thresholds

- Per run and label, the threshold is chosen on **validation only**: the value in 0.01, 0.02, ..., 0.99
  that maximizes validation F1; ties go to the value closest to 0.5.
- Applied unchanged to test. Reported: test macro F1, per-label sensitivity and specificity, next to the
  same quantities at 0.5. AUROC does not depend on thresholds and is not repeated.

## C. Saliency: does the HYP output use the calibration pulse?

- Geometry (from `src/ptbxl_rendering.py` and the v1 protocol: 8 px/mm, 25 mm left margin, 2x2 pooling):
  in each 120 x 1120 lead row of the model input, the 1 mV calibration pulse lies in columns 48-87
  (12-22 mm), the waveform in columns 100-1099. On adaptive-gain pages the pulse height encodes the page
  gain; on fixed-gain pages it is the same on every page.
- Models: seed-42 CNN-only, CNN-LSTM, temporal conv and hybrid ViT, each in its adaptive-gain version
  (scored on adaptive pages) and its fixed-gain retrained version (scored on fixed-gain pages).
- ECGs: all 268 HYP-positive validation ECGs and the first 268 HYP-negative validation ECGs in cache
  order. Validation only; no test pages.
- Attribution: integrated gradients of the HYP logit, 16 steps, baseline = a blank page (zero ink,
  normalized with the model's own training statistics); absolute attributions summed per region.
- Outcome: share of total attribution in the pulse columns (48-87) compared with their area share
  (40/1120 = 3.6%), and in the waveform columns; mean per model with a 95% bootstrap interval over ECGs
  (1,000 resamples, seed 2024). NORM logit as a secondary check.
- Prediction stated before computing: if the adaptive-gain models read gain from the pulse, their pulse
  share for HYP will be clearly above its area share and above the fixed-gain models' share. A low share
  does not rule out gain use, because gain also changes the size of every waveform.
- Figure: mean absolute attribution per column (HYP logit), adaptive vs fixed gain, per model.

## D. Pulse-swap intervention (added 2 Nov 2026 after C, frozen before computing D)

C showed where the HYP output attends; D tests whether the pulse changes the output (`src/analyze_pulse_swap.py`).

- Models: the four seed-42 adaptive-gain networks on adaptive-gain **validation** pages (all 2,146), and,
  as a control, the four seed-42 fixed-gain networks on fixed-gain validation pages.
- Intervention: only columns 48-87 of every lead row are replaced, in raw ink space before each model's own
  normalization. Reference blocks come from fixed reference pages: the first validation page in cache order
  with v1 gain 10, 5 and 2.5 mm/mV. Conditions:
  - `original`: unchanged page;
  - `pulse_10`, `pulse_5`, `pulse_2.5`: every page gets the pulse block of that gain;
  - `sham`: every page gets the block of a *different* reference page with its *own* gain (pages at
    ≤1.25 mm/mV are excluded from sham) - tests that copying a block itself changes nothing;
  - `mask`: the block is blank (no ink).
- Outcomes per condition, overall and by original page gain: mean HYP probability and its mean change from
  `original` (95% patient-cluster bootstrap, 1,000 resamples, seed 2024); HYP AUROC and sensitivity at 0.5;
  macro AUROC. NORM probability as a secondary outcome.
- Prediction stated before computing: if adaptive-gain models read gain from the pulse, a short pulse
  (`pulse_2.5`, as on high-voltage pages) raises HYP probability on pages originally at 5-10 mm/mV, a tall
  pulse (`pulse_10`) lowers it on pages originally at 2.5 mm/mV, and `sham` changes almost nothing. The
  fixed-gain control models never saw the pulse vary, so `pulse_10` should move them little.
