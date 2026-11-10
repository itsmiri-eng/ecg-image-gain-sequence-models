# Summary for Dr. Elgendi: ECG-image model comparison (10 November 2026)

*Prepared for Miriam Istanboli's meeting. All numbers come from saved results. The PTB-XL test fold had been
looked at before the models were fixed, so PTB-XL results are exploratory; the external test is confirmatory.*

## What was done

- **Data and labels.** All 21,388 labelled PTB-XL ECGs, drawn as calibrated paper-style pages, with the five
  diagnostic superclasses (NORM, MI, STTC, CD, HYP). Official patient folds: 1–8 train, 9 validation, 10 test.
- **Models**, all sharing one 1,220-parameter CNN encoder and differing only in the head, with **5 training seeds
  each**:
  - CNN-only;
  - a **capacity-matched** CNN-only head, with the same size as the sequence heads but unable to relate
    positions along a row;
  - CNN-LSTM;
  - CNN + temporal convolution;
  - CNN + small ViT.
- **References:** PCA + LDA, PCA + linear SVM (our reading of "vsm"), and a 1D ResNet on the raw signals.
- **Rendering conditions.** Two page renderings: the original **adaptive gain** (gain reduced for high-voltage
  ECGs) and **fixed gain** (every page at 2.5 mm/mV).
- **External confirmatory test,** run once under an analysis plan frozen before the data were opened:
  - Georgia, USA (6,047 ECGs);
  - Chapman-Shaoxing + Ningbo, China (9,004 ECGs).
- **Statistics.** Hierarchical bootstrap (seeds, then patients) with Holm correction.

## Main findings

1. **Head comparison (PTB-XL test, 5 seeds).** CNN-only 0.830 < capacity-matched 0.865 < CNN-LSTM 0.876 ≈
   temporal conv 0.879 ≈ ViT 0.886 macro AUROC.
   - Model size explains about 0.035 of the gain over CNN-only.
   - Relating information along the row adds a further 0.011–0.020, which is significant after Holm.
   - The three sequence heads are not separable on PTB-XL.
2. **A rendering cue.**
   - **The cue.** Adaptive page gain encodes amplitude, and amplitude encodes hypertrophy: HYP prevalence is 2%
     at 10 mm/mV and 31% at 2.5 mm/mV. Gain alone predicts HYP with AUROC 0.72–0.74.
   - **What the models do with it.** Saliency and a calibration-pulse swap show that whole-row models (ViT,
     LSTM) read the calibration pulse as a voltage scale. That is legitimate, as cardiologists do the same.
   - **How big the artefact is.** A post-hoc decomposition puts the pure artefact (the "gain prior") at only
     +0.03 to +0.08 in HYP probability.
3. **Fixed-gain rendering helps every model and hurts none.**
   - **PTB-XL macro / HYP AUROC:** CNN-only +0.030 / +0.073, temporal conv +0.016 / +0.043,
     LSTM +0.012 / +0.014, ViT +0.006 / +0.010 (ViT not significant).
   - **Best image model:** fixed-gain temporal conv, 0.895.
4. **External confirmation.**
   - Fixed-gain training improved **hypertrophy AUROC for every architecture in both external cohorts**, again
     most for the local models.
   - The head ordering CNN-only < capacity-matched < {LSTM, ViT} replicated. Temporal conv was unstable
     externally.
   - All models lose performance on new hospitals: about −0.15 macro AUROC in the USA and −0.06 in China.
5. **Image vs signal.** The signal model stays best: 0.928 on PTB-XL, 0.788 in Georgia and 0.856 in China.
   Compact image models lose about 0.03–0.05.

## Limitations

- All pages are clean synthetic renderings; there are no real scans or photos.
- NORM could not be scored externally, and MI not in Georgia, because of how those datasets are labelled.
- External labels come from a data-driven mapping.
- The models are small by design.
- Seed 42 of four networks was trained on a CPU and the rest on a GPU.
- The PTB-XL test fold was reused.

## Ready for you to review

- An IEEE-format manuscript draft (target IEEE JBHI), with 5 figures, 4 tables and 37 references verified
  against Crossref/DataCite. Affiliations are left blank on purpose.
- The full reproducibility package: code, frozen protocols and every run record.

## Questions for you

1. Did "vsm" in the notes mean SVM, or a state-space model (Mamba)? And was "portal convolutions" temporal
   convolutions?
2. Target journal (IEEE JBHI, IEEE TBME, or Computers in Biology and Medicine), author order and affiliations.
3. Should a standard pretrained image model (ResNet-18) be added as a further reference before submission?
