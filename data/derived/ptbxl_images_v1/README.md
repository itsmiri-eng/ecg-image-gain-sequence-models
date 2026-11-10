# PTB-XL image dataset — v1

21,799 grayscale ECG pages generated directly from the local PTB-XL 1.0.3 recordings. Each 2,240 × 3,024 PNG shows all twelve leads over ten seconds. The PNG files total 3.37 GB, with additional manifests and provenance stored alongside them.

For classification, use `splits/train.csv` (17,084 images), `splits/validation.csv` (2,146), and `splits/test.csv` (2,158). These preserve the official patient folds. The 411 records without a mapped diagnostic superclass remain in `all_images.csv` and `classification_exclusions.csv`; they are not normal or confidently all-negative targets.

Targets are the five overlapping binary columns `label_NORM`, `label_MI`, `label_STTC`, `label_CD`, and `label_HYP`. CSV booleans are written as `True`/`False`; parse them explicitly. Paths are relative to this directory. Patient/ECG IDs, SCP annotations and other provenance are not model inputs.

Each page in `images/{split}/` has a matching source/calibration record in `provenance/{split}/`. Actual gain is printed and saved. There are 420 pages flagged for low gain and one with a flat V5 lead. These remain in the dataset. Native waveforms are the authoritative numerical signal; these clean rendered images are not clinically lossless representations or real paper scans.

See [the dataset card](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_image_dataset.md>) for methods, limitations, source attribution, reproduction commands and the overview figure. Use `generation_state.json` for completion state and `verification.json` for the full integrity-check result. The existing four-class Mendeley training code requires a separate multilabel configuration and suitable image preprocessing before use with this cohort.
