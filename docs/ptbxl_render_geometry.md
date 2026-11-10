# PTB-XL training-only render geometry audit

Audited **17,418 training ECGs from 15,023 patients** (official folds 1–8 only). No validation/test waveform was inspected. Each record has 12 leads × 5,000 samples at native 500 Hz, covering 10 seconds; the last sample occurs at 9.998 seconds.

All **209,016 channel headers** use mV, gain 1,000 ADC units/mV, baseline 0, and WFDB format 16. Lead order is I, II, III, AVR, AVL, AVF, V1–V6. There were **0 missing/nonfinite/sentinel records**. Five complete records decoded with wfdb.rdsamp matched calibrated NumPy decoding exactly.

Flat-channel flags: **1 training record** (ECG 12722: V5). No exclusion was made.

## Training amplitude distribution

Maximum absolute amplitude is taken over every sample and lead in a record. Extrema include noise/artifacts; no centering or filtering was applied.

| Quantile | Record maximum absolute amplitude (mV) | Record maximum per-lead peak-to-peak (mV) |
|---|---:|---:|
| 0 | 0.422 | 0.75 |
| 0.5 | 1.905 | 2.659 |
| 0.9 | 3.205 | 4.4493 |
| 0.95 | 3.83515 | 5.36315 |
| 0.99 | 5.65296 | 8.34066 |
| 0.995 | 7.3105 | 10.3775 |
| 0.999 | 14.5406 | 18.6135 |
| 1 | 32.716 | 65.408 |

Global extrema: **-32.692 to 32.716 mV**. Fixed 10 mm/mV gain with one symmetric row pitch sufficient for all training extrema would require about **660.32 mm per row**, including a 6 mm total margin. Twelve such rows would be impractically tall.

## Fixed pitch with disclosed page-wide gain changes

The current proposal is **8 pixels/mm**, **25 mm/s**, and nominal **10 mm/mV**. Each full 10-second lead spans **2,000 pixels**. A 30 mm pitch gives 240 pixels per row; reserving 3 mm above/below the usable waveform band leaves ±12 mm. Reduce gain by powers of two only as needed to fit the page maximum absolute amplitude. Use one shared gain for all leads; retain the calibrated zero baseline.

| Actual gain (mm/mV) | Training records | Fraction | Pixels per mV at 8 px/mm |
|---|---:|---:|---:|
| 10 | 1,160 | 6.66% | 80 |
| 5 | 11,665 | 66.97% | 40 |
| 2.5 | 4,257 | 24.44% | 20 |
| 1.25 | 292 | 1.68% | 10 |
| 0.625 | 33 | 0.19% | 5 |
| 0.3125 | 11 | 0.06% | 2.5 |

Only **6.66%** retain nominal 10 mm/mV; most use 5 or 2.5 mm/mV. The **44 records** needing ≤0.625 mm/mV have ≤5 pixels/mV; the 11 at 0.3125 mm/mV have **2.5 pixels/mV**. Small details cannot be fully represented at those gains even though the traces fit. Actual gain must remain in image annotation and metadata, with a correctly sized 1 mV calibration pulse.

A 50 mm pitch with usable ±22 mm would retain 10 mm/mV for **11,450/17,418 (65.74%)**. Its waveform canvas is 2,000×4,800 rather than 2,000×2,880, a **1.67×** pixel cost before margins. The JSON contains other fixed-pitch options; this is a geometric tradeoff, not a classification-driven recommendation.

## Faithfulness, raster limits, and storage

Twelve stacked, time-aligned full-length leads avoid omissions from displaying only 2.5-second windows per lead. All 5,000 native points per lead may be plotted, but they map to **2.5 samples per horizontal pixel** at the current resolution. This is not invertible waveform preservation. Keep the original WFDB signals and rendering metadata.

The proposed complete page, **2,240×3,024 grayscale uint8**, is **6.774 MB uncompressed per image**, or **147.66 GB** for 21,799 records. Actual PNG compression depends on grid, strokes and antialiasing; measure previews before predicting disk use. Stream one record at a time rather than constructing a full raster cohort in memory.

The lower 4 pixels/mm candidate was superseded by 8 pixels/mm before rendering. Amplitude quantiles and gain assignments do not change with pixel density; only raster resolution and footprint change.

Training preview IDs: **20002** (minimum maximum amplitude, 0.422 mV, fold 4), **10565** (maximum, 32.716 mV, fold 4), **20187** (near the 99th percentile, 5.655 mV, fold 8), and **12722** (flat V5, fold 2).

wfdb is importable. ecg_plot, ecg_image_kit, ecg_imagekit, and ecgimagekit are not importable. No package was installed.

Detailed per-lead extrema, quantiles, gain counts, preview paths and provenance are in `experiments/protocols/ptbxl_images_v1/train_render_geometry_audit.json`. No renderer, waveform, old artifact, or image was modified by this audit.
