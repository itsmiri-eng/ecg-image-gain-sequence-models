# Input representation review

Reviewed 2026-09-24. Eight training images were inspected to assess the fixed image representation before planning another model. The cached inputs are reproducible and contain recognizable ECG page structure, but **one reviewed example has trace peaks cut by the upper crop boundary**. The resize also changes aspect ratio and visibly reduces fine detail. These inputs are suitable for a small image-classification baseline with stated limitations; this review does not validate them as calibrated waveforms or establish that they preserve all diagnostic information.

## Scope and selection

The review selected the first two lexicographically sorted `sample_id` values per folder class from the **training partition only**, requiring distinct `group_id` values within each class. Selection used `data/splits/fixed_split.csv`; it did not use predictions, errors or performance. No test image was inspected, and no validation example was needed. The eight examples are a deterministic technical spot-check, not a representative estimate of defect prevalence. Folder labels below are supplied metadata, not clinical interpretations.

The source crop at native resolution, a scaled overview and the cached input were inspected for each selected image. A full-page crop-boundary sheet additionally checked what lies outside the retained region. Native-resolution details were examined to assess loss of visible structure. No diagnosis or label was reassigned.

## Geometry and reproducibility

All selected sources are 2213 by 1572 pixels. The configured fractional crop `[0.03, 0.19, 0.98, 0.96]` becomes integer coordinates **`(66, 298, 2168, 1509)`**, giving a **2102 by 1211 pixel** crop. Grayscale conversion, bilinear resizing and intensity inversion produce a **`[1, 128, 256]`** array in `[0, 1]`.

The source crop has width-to-height ratio 1.7358; the output ratio is 2.0. Thus the transformation stretches horizontal geometry by **15.22% relative to vertical geometry**. It reduces the width by 8.21 times and the height by 9.46 times. Each output pixel consequently summarizes a substantial source area, rather than representing an original acquisition sample.

For all eight examples, rerunning the existing `preprocess_image` function produced arrays **exactly equal** to the corresponding frozen cache entries. Recomputed input SHA-256 hashes matched the saved split. This verifies the selected cache entries against the configured implementation; it does not establish physiological validity.

## Observed content

All eight crops exclude the printed report-identifier/demographic header and footer metadata. The previews retain three rows of four lead panels and a fourth, longer rhythm strip. Lead-name text, calibration-pulse shapes, grid texture and vertical panel separators remain in the input. The preprocessing does not remove all non-waveform marks. The full-page review export masks its identifier header for display; this display-only masking does not alter raw data or model inputs.

At 256 by 128 pixels, major row structure and many peaks remain visible, while narrow strokes, lead lettering and small trace variations are blurred or reduced to a few pixels. The grid and separators remain visible as potential nonclinical features. No inspected input was blank, constant or numerically saturated: ink means ranged from 0.05239 to 0.07531, standard deviations from 0.07186 to 0.11463 and maxima from 0.62745 to 0.78431. These are pixel statistics, not signal-quality or clinical-readability measurements.

For **`01e055478056e70588b3`**, source `ecg files/Abnormal/HB(59).jpg`, tall traces in the first panel extend above the graph border into the whitespace. Their upper portions exist in the original image but are removed by the fixed crop starting at y = 298. This is a confirmed crop-induced loss, not merely a display artifact. The upper-edge dark-pixel fraction exceeds 0.5 ink intensity at three of 256 cached columns (1.17%). A saved native-pixel illustration shows the omitted trace portions. No obvious outer-boundary trace truncation was identified in the other seven reviewed examples, but that observation is limited to this inspection and is not a guarantee of full trace preservation.

Within source region `(200, 180, 650, 298)`, a grayscale threshold below 128 identifies four narrow connected trace components reaching the old crop boundary. Their upper y coordinates are 222, 226, 229 and 232, so **66–76 source pixels of vertical extent** are omitted from these four peaks. Their bounding boxes and the explicitly limited detection rule are saved in `clipping_detail.json`. This describes visible crop loss in one case, not an automated clinical waveform measure.

| Supplied class | Training sample ID | Input SHA-256 prefix | Mean ink | Ink standard deviation | Visual finding |
|---|---|---|---:|---:|---|
| Normal | `056b1c90d8b60e4846b1` | `9d4f1f1eb646` | 0.05629 | 0.08065 | Header/footer excluded; panel layout and rhythm strip retained |
| Normal | `0c2ef965519ba2ce415e` | `948c9668c50d` | 0.05311 | 0.07978 | Header/footer excluded; panel layout and rhythm strip retained |
| MI | `1ddd77a4141874cbc65b` | `d41a91195393` | 0.05239 | 0.07186 | Header/footer excluded; low-resolution attenuation visible |
| MI | `426e33afdf7c583eadbb` | `8e9ab6ea428e` | 0.05783 | 0.08702 | Header/footer excluded; panel layout and rhythm strip retained |
| Abnormal | `00a9243e7385f8eb8f9f` | `4350bbe58b55` | 0.06526 | 0.10411 | Header/footer excluded; grid and separators remain visible |
| Abnormal | `01e055478056e70588b3` | `03902fb28721` | 0.07531 | 0.11463 | Confirmed truncation of tall first-panel traces at upper crop boundary |
| History_MI | `10b7b351a4e3a369a09b` | `4bf34aac8ea4` | 0.05896 | 0.08831 | Header/footer excluded; panel layout and rhythm strip retained |
| History_MI | `29917828d7bcdabe9459` | `cbd310de8288` | 0.05322 | 0.07394 | Header/footer excluded; low-resolution attenuation visible |

Full source, decoded-pixel and input hashes, crop geometry, edge statistics and cache-match results are saved in `results/readiness/inputs/development_input_review.json`. The phrase “low-resolution attenuation” describes visible image detail and does not classify an ECG or explain model performance.

## Implications for the next model

The existing baseline outputs and frozen inputs remain unchanged. A future preprocessing variant should address crop-boundary loss and compare a larger, aspect-preserving representation on development data under a separately versioned protocol. This review identifies candidate changes; it does not establish that they improve classification. A broader training/validation image audit is needed before treating any replacement crop as valid for the cohort.

For this illustrated training case only, **`(66, 200, 2168, 1509)`** is a bounded candidate crop for review. It adds 98 source pixels above the existing crop and leaves 22 pixels above the highest of the four measured peaks. It should be paired with aspect-preserving resizing plus padding, rather than stretching to the target shape. This candidate is not validated for all records or ready to replace the frozen input cache. It can also retain technician/referring-physician header text on the right, so preserving tall traces while excluding identifying header content remains an explicit unresolved requirement. A fixed header mask must not be assumed safe because waveform strokes may extend into that same region. Any development-only candidate comparison needs its own version and coverage checks before a new model experiment.

For CNN–LSTM planning, an image column is **not an established time sample**. Different horizontal panels contain different named leads, three lead rows are accompanied by a separate rhythm strip, and a flattened raster mixes panel transitions with tracing geometry. The source contains nominal gain/speed text and calibration graphics, but this review did not validate their pixel-to-time or pixel-to-millivolt relationship, lead synchronization, per-panel durations or reconstruction accuracy. The crop removes the footer settings, and resizing changes geometry.

A CNN–LSTM over ordered image features could be evaluated as an explicitly defined spatial-sequence model. It must not be described as a calibrated temporal ECG model merely because an LSTM consumes image columns. A waveform-based CNN–LSTM would first require validated lead localization and ordering, trace extraction, time/amplitude calibration, and agreement with suitable waveform reference data. None of those steps was implemented or validated in this review.

## Saved inspection artifacts

All new image-review artifacts are under `results/readiness/inputs/`:

- `development_input_review.json`: selection policy, eight paths and hashes, geometry, image statistics and exact cache verification.
- `review_01_Normal.jpg` through `review_08_History_MI.jpg`: eight source-crop/cache comparisons with a native-pixel detail.
- `<sample_id>_crop_native.png`: the selected crops at original pixel resolution.
- `<sample_id>_cached128x256.png`: grayscale display versions of the selected frozen inputs. Display conversion does not replace the floating-point cache.
- `source_crop_boundaries.jpg`: eight source overviews with the configured crop outlined in red and the identifier header masked for display.
- `upper_boundary_clipping.png`: native-pixel evidence for the confirmed crop-boundary loss.
- `clipping_detail.json`: four measured component boxes above the old crop boundary and the bounded candidate for this one case.

No source image, configuration, frozen split, processed cache, model output or model code was changed. This review makes no clinical diagnosis, new model-performance claim or claim of exhaustive cohort-wide quality validation.
