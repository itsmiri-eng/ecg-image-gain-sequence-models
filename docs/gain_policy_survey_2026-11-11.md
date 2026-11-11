# How public ECG-image resources handle gain (survey, 11 November 2026)

**Question.** Do widely used public ECG-image generators and datasets draw pages at one fixed gain, at an
amplitude-dependent (adaptive) gain, or with clipping or overlap of large deflections? Is the gain randomized?

**Method.** Primary sources only: each resource's own paper and its actual code or documentation, read on
11 Nov 2026 (in the browser, with WebFetch, or as raw GitHub source text saved to a scratch folder and read, never
run). Every claim below gives the exact URL and the file/line or section. Anything not found in a primary source
is written as **not stated**. Nothing was installed or executed.

## Summary

| Resource | Gain when drawing | Amplitude-dependent / adaptive? | Large deflections | Gain randomized? |
|---|---|---|---|---|
| ECG-Image-Kit (Shivashankara et al. 2024) | fixed 10 mm/mV (0.5 mV per 5-mm major grid; page label "10mm/mV"; 1-mV calibration pulse) | no | traces are drawn at a fixed row offset with no per-row clipping in the code, so they can overlap neighbouring rows (inferred from code; the paper does not discuss it) | no (no gain option; only the pixel resolution can be randomized, which scales the grid with the trace) |
| George B. Moody PhysioNet Challenge 2024 data and code | images made with ECG-Image-Kit (above) | not stated beyond ECG-Image-Kit | not stated | not stated |
| ECG-Image-Database (Reyna et al. 2024, arXiv) | generated with ECG-Image-Kit, then printed/scanned/photographed; grid described as 0.5 mV per 5-mm square | not stated | clipping not mentioned | not stated |
| Sangha et al. 2022 (Nat Commun) | images made with the `ecg-plot` Python library (v0.2.8, the release current at the paper's access date); that version draws every lead at one fixed scale (≈10.16 mm/mV at the nominal figure size) | no | no per-row clipping in the library code; traces can overlap neighbouring rows | not stated (paper varies start time, layout, colour scheme; no gain variation mentioned) |
| Khan et al. 2021, "ECG Images dataset of Cardiac and COVID-19 Patients" (real printouts, EDAN SERIES-3) | not stated | not stated | not stated | not applicable (real device printouts) |

**Conclusion.** In none of the public generators or datasets checked was an amplitude-dependent (adaptive) page
gain found: the synthetic generators draw at one fixed, standard gain (10 mm/mV) and let large deflections run into
neighbouring rows. The adaptive gain studied in this project is therefore a property of this project's own
renderer, not a documented feature of those resources. For real device printouts, gain is not stated in the dataset
papers checked. Manufacturer manuals describing automatic-gain settings on electrocardiographs exist according to
search-engine snippets, but the pages could not be opened (bot-protection checks, which were not bypassed), so they
are **not verified and not cited**.

## Details and sources

### 1. ECG-Image-Kit

- Repository: https://github.com/alphanumericslab/ecg-image-kit, folder `codes/ecg-image-generator`, read at
  commit `27b90f56896c9fc78b05a83ca14844ea2637aa0b` (main on 11 Nov 2026).
- `ecg_plot.py`
  (https://github.com/alphanumericslab/ecg-image-kit/blob/27b90f56896c9fc78b05a83ca14844ea2637aa0b/codes/ecg-image-generator/ecg_plot.py):
  - lines 12–14: `standard_values` sets `'y_grid_size' : 0.5` (mV per major grid square) and
    `'y_grid_inch' : 5/25.4` (5 mm per major square) → 10 mm/mV, the same for every record;
  - lines 131–132: comment "Each big grid corresponds to 0.2 seconds and 0.5 mV";
  - lines 250–253: the calibration pulse is an array of ones (1 mV) for 0.2 s;
  - line 322: `y_vals = ecg[leadName] + y_offset`: the trace in mV is added to a fixed row offset; no scaling by
    the signal's amplitude and no clipping to the row;
  - line 487: the page is labelled `'10mm/mV'`.
- `gen_ecg_image_from_data.py` (same commit): the command-line options (lines 21–81) contain no gain or amplitude
  option; line 107 randomizes only the resolution in dots per inch (`--random_resolution`), which changes pixels
  per mm for the whole page, grid included.
- Paper: Shivashankara et al., *Physiol. Meas.* 2024, doi:10.1088/1361-6579/ad4954; open-access copy
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11135178/, §3.1 "Background": ECGs are conventionally printed at
  25 mm/s and "a vertical scale of 1 mV per 10 mm", with a 1-mV, 0.2-s calibration pulse. Gain reduction,
  adaptive gain and clipping: **not stated**.

### 2. George B. Moody PhysioNet Challenge 2024

- Challenge page: https://moody-challenge.physionet.org/2024/, sections "Data", "Data Processing" and
  "Accessing the Data": the images are synthetic images generated from PTB-XL waveforms with ECG-Image-Kit
  (`gen_ecg_images_from_data_batch.py`), with artifacts such as shifting, rotation, creases and font changes.
  Gain, mm/mV and clipping: **not stated** on the page.
- Code: https://github.com/physionetchallenges/python-example-2024 (example code; uses ECG-Image-Kit for images).
- Hence the Challenge images inherit ECG-Image-Kit's fixed 10 mm/mV (item 1). Whether the hidden test images used
  any other gain: **not stated**.

### 3. ECG-Image-Database

- Reyna et al., arXiv:2409.16612 (v1), https://arxiv.org/html/2409.16612v1:
  - §3 "Generating realistic ECG images from time-series data": made with ECG-Image-Kit;
  - §5.4.1: printed on a LaserJet Pro M501 at 600 dpi on US Letter paper, then scanned/photographed;
  - Appendix A.2 "Printed ECG": 1-mm squares = 0.1 mV, 5-mm squares = 0.5 mV (i.e. 10 mm/mV).
- Adaptive gain, randomized gain, clipping: **not stated**.

### 4. Sangha et al. 2022

- *Nat. Commun.* 13, 1583 (2022), doi:10.1038/s41467-022-29153-3, https://www.nature.com/articles/s41467-022-29153-3:
  - Results, "Performance of image-based multilabel classification", and Methods, "Data preprocessing": "The ECGs
    were transformed to the corresponding printed images using the python library ecg-plot"; variations were the
    start offset, layout (standard/alternate, shuffled leads) and colour scheme. Gain: **not stated**.
  - Reference list: "ECG Plot Python Library, accessed 22 September 2021, https://pypi.org/project/ecg-plot/".
- `ecg-plot` release history (https://pypi.org/project/ecg-plot/#history): 0.2.8 on 10 Jan 2021 was the current
  release on 22 Sep 2021 (the next, 0.3.0, is from 21 Oct 2026).
- Source of 0.2.8: https://github.com/dy1901/ecg_plot/blob/3855e9450e1410cb439a0d91c55ec6a6f65ff868/ecg_plot/ecg_plot.py
  (commit of 10 Jan 2021, `setup.py` version 0.2.8), function `plot`:
  - line 122: figure size `secs*columns` inches wide and `rows * row_height / 5` inches high;
  - lines 124–131: the axes fill the whole figure;
  - lines 137–138: the vertical range is `rows*row_height/2` mV, so 1 mV = 0.4 inch ≈ 10.16 mm at the nominal size,
    the same for every record;
  - line 151: major y grid every 0.5 mV;
  - lines 167 and 183–188: each lead is drawn as signal + fixed row offset; no per-row clipping and no
    amplitude-dependent scaling.
  - No calibration pulse is drawn in this version (no pulse code in the file).
- The authors' repository https://github.com/CarDS-Yale/ECG-DualNet holds example images and a README only (no
  plotting code), so the exact `row_height` they used: **not stated**.

### 5. Khan et al. 2021 (real printouts)

- *Data in Brief* 34, 106762 (2021), doi:10.1016/j.dib.2021.106762, https://pmc.ncbi.nlm.nih.gov/articles/PMC7820911/,
  specifications table: images from EDAN SERIES-3 devices at 500 Hz. Gain, calibration and clipping: **not stated**.
  The gain printed on individual pages was not audited here.

### 6. Not verified (not used in the paper)

- Search-engine snippets for electrocardiograph manuals (Schiller AT-2 on manualslib.de; Edan SE-1201 on
  manualslib.com and manualshive.com) describe automatic-gain settings that reduce limb or chest leads to 5 mm/mV
  when amplitudes are large. All three pages returned bot-protection checks or HTTP 403 and were not opened. These
  claims are therefore not verified and not cited.

## Clinical convention (for the Methods sentence on 10 mm/mV)

- Hancock et al., AHA/ACCF/HRS recommendations, part V (chamber hypertrophy), *Circulation* 119, e251–e261 (2009),
  doi:10.1161/CIRCULATIONAHA.108.191097, https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.108.191097,
  section on left ventricular hypertrophy voltage criteria: "Using normal standardization, 10 mm equals 1 mV".
- Kligfield et al. 2007 (part I, doi:10.1161/CIRCULATIONAHA.106.180200), which the manuscript cited for "10 mm/mV with
  a 1-mV calibration pulse", was searched in full text: it states the 25 mm/s format but **not** the 10 mm/mV gain or
  the calibration pulse. The manuscript citation was corrected accordingly (Hancock 2009 and Shivashankara 2024).
