# Domain Shift and Format Sensitivity in ECG Image Classification Pipelines

An end-to-end research pipeline that reconstructs digital ECG signals from paper-style ECG images and evaluates how well a trained classifier generalises across different image sources, rendering conventions, and dataset formats.

## Project purpose and framing

The central question is not whether a single ECG image classifier can achieve high accuracy on its training distribution — it can. The question is whether that accuracy transfers when the image source changes. This project quantifies the answer for one specific pipeline trained on the Mendeley ECG dataset and zero-shot evaluated on PTB-XL-derived images: performance collapses to approximately chance level, showing that strong within-dataset results do not imply robustness across image formats or datasets.

---

## File structure

```
ecg-image-domain-shift/
├── ecg_domain_shift_full_pipeline.ipynb   # Main notebook (run top to bottom)
├── ptbxl_generate.py                      # Generates PTB-XL images and manifest
├── ptbxl_manifest.csv                     # 600-row manifest (relative image paths)
├── mendeley_model_weights.pth             # Trained ResNet weights (Mendeley dataset)
├── figures/
│   ├── ablation_comparison.png            # Multi-component pipeline comparison
│   ├── cm_final_enhanced_pipeline.png     # Mendeley test confusion matrix
│   ├── cm_ptbxl_domain_adapted.png        # E7 fold-10 confusion matrix
│   ├── cm_ptbxl_zero_shot_best.png        # Best zero-shot fold-10 confusion matrix
│   ├── confusion_matrix_snr25.png         # High-quality subset confusion matrix
│   ├── degradation_experiment.png         # Robustness degradation line plot
│   ├── domain_shift_analysis_a.png        # Zero-shot full cohort (N=600)
│   └── domain_shift_analysis_b.png        # Zero-shot + E7, fold-10 (N=45)
└── results/
    ├── mendeley_metrics.json              # Mendeley test-set metrics
    ├── mendeley_split_seed42.csv          # Train/test image split index (seed 42)
    ├── ptbxl_analysis_b_metrics.json      # Fold-10 zero-shot metrics (E1–E6)
    ├── ptbxl_common_test.csv              # Per-image fold-10 predictions (E1–E7)
    ├── ptbxl_e7_metrics.json              # E7 domain-adapted metrics
    ├── ptbxl_finetuning_history.csv       # E7 epoch-by-epoch training log
    ├── ptbxl_zero_shot_full.csv           # Full-cohort zero-shot metrics (E1–E6)
    └── split_summary.json                 # Fold counts and patient-overlap proof
```

Datasets and generated images are **not included**. See Dataset requirements below.

---

## Dataset requirements

### Mendeley ECG dataset
- Source: https://data.mendeley.com/datasets/gwbz3fsgp8/2
- Four-class image dataset: Normal, MI, Abnormal, History_MI
- Set `MENDELEY_DATASET_PATH` in Block 0 of the notebook to the folder that contains those four subdirectories

### PTB-XL
- Source: https://physionet.org/content/ptb-xl/1.0.3/
- Required files: `ptbxl_database.csv`, `scp_statements.csv`, and the `records500/` waveform folder
- Set `PTBXL_ROOT` in `ptbxl_generate.py` to the dataset root

**Neither dataset nor generated ECG images are distributed with this repository.**

---

## Environment setup

The notebook and generator require the `ecgimg` conda environment.

```bash
conda create -n ecgimg python=3.10
conda activate ecgimg
pip install torch==2.0.1 numpy==1.26.4 pandas==2.2.2 opencv-python==4.6.0 \
    scipy==1.13.1 seaborn==0.12.2 matplotlib==3.9.4 \
    scikit-learn==1.4.2 scikit-image==0.21.0
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## External dependency: ECG-Image-Kit

The PTB-XL image generator calls `gen_ecg_images_from_data_batch.py` from the ECG-Image-Kit project:

- Repository: https://github.com/alphanumericslab/ecg-image-kit
- Clone it and set `GEN_ECG_SCRIPT` in `ptbxl_generate.py` to the full path of `gen_ecg_images_from_data_batch.py`

---

## Configuration

### ptbxl_generate.py

Edit the `CONFIGURATION` block at the top of the file:

| Variable | Description |
|---|---|
| `PTBXL_ROOT` | Root folder of the PTB-XL dataset |
| `OUTPUT_DIR` | Empty folder where generated PNG images will be written |
| `GEN_ECG_SCRIPT` | Full path to `gen_ecg_images_from_data_batch.py` from ECG-Image-Kit |

### Notebook (Block 0)

| Variable | Description |
|---|---|
| `MENDELEY_DATASET_PATH` | Root folder of the Mendeley ECG dataset |
| `PTBXL_IMAGES_DIR` | Folder where `ptbxl_generate.py` wrote the generated images (must match `OUTPUT_DIR`) |
| `RETRAIN_MODEL` | `False` to load existing weights, `True` to train from scratch |

`PROJECT_DIR`, `WEIGHTS_PATH`, `PTBXL_MANIFEST`, `FIGURES_DIR`, and `RESULTS_DIR` resolve automatically from the notebook location and do not need to be changed.

---

## PTB-XL image generation

Run once, in the `ecgimg` environment, from the repository root:

```bash
python ptbxl_generate.py
```

This selects 600 records (150 per class, one per patient, seed 42), generates one PNG per record using ECG-Image-Kit, and writes `ptbxl_manifest.csv` with relative image paths. The manifest included in this repository corresponds to a completed run; regenerate only if you change `PTBXL_ROOT`, `OUTPUT_DIR`, or `SEED`.

---

## Notebook execution order

Run all cells top to bottom in one pass:

| Block | Description |
|---|---|
| 0 | Configuration — set your paths here |
| 1 | Imports and reproducibility seed |
| 2 | 1D-ResNet architecture |
| 3 | Digitization pipeline functions |
| 4 | Mendeley dataset class |
| 5 | Stratified train/test split (seed 42) |
| 6 | Training or weight loading |
| 7 | Full Mendeley evaluation + confusion matrices |
| 8 | Multi-component pipeline comparison |
| 9 | Synthetic degradation experiment |
| 10 | Cross-dataset domain-shift analysis (PTB-XL) — requires `ptbxl_generate.py` to have been run |
| 11 | Final summary |

---

## Results

### Mendeley within-dataset performance (N=186 test images)

| Metric | Value |
|---|---|
| Accuracy | 94.62% |
| Balanced accuracy | 95.05% |
| Macro F1 | 0.9475 |
| MCC | 0.9287 |

### Multi-component pipeline comparison (Mendeley test set)

| Method | Accuracy | MCC |
|---|---|---|
| Baseline A: Argmin + fixed scale | 83.33% | 0.7790 |
| Baseline B: Canny + skeletonization | 92.47% | 0.8995 |
| Final: Contour + grid + skeletonization | 94.62% | 0.9287 |

### Synthetic degradation

Rotation (+5°, +15°, +30°), blur (k=3, 7, 15), and brightness changes applied to test images. Rotation and blur caused substantial degradation; brightness had a smaller effect. Rotation levels are positive-only (not ±).

### PTB-XL cross-dataset evaluation

**Analysis A — zero-shot, full cohort (N=600)**

| Experiment | Method | Accuracy |
|---|---|---|
| E1 | Standard pipeline | 25.00% |
| E2 | Grid removal | **25.50%** |
| E3 | Min-max norm | 25.00% |
| E4 | Adaptive threshold | 25.00% |
| E5 | No skeleton (argmax) | 25.00% |
| E6 | Min-max norm + grid removal | 25.50% |

Best full-cohort zero-shot accuracy: **25.50%** (E2, grid removal). All results are near chance level (25%).

**Analysis B — zero-shot, fold-10 test set (N=45)**

Best fold-10 zero-shot accuracy: **28.89%** (E2, grid removal).

**E7 — domain-adapted (train folds 1–8, val fold 9, test fold 10)**

| Metric | Value |
|---|---|
| Accuracy | 20.00% |
| MCC | −0.0510 |
| Frozen layers | layer1, layer2, layer3 |
| Trained layers | layer4, fc |
| Learning rate | 0.0001 |
| Epochs | 20 (early stopping patience 5) |

E7 limited fine-tuning failed to recover performance; accuracy fell below zero-shot E2 and below chance, confirming that the domain gap is not bridged by partial fine-tuning of this depth.

---

## PTB-XL label mapping

Labels are derived from the official `scp_statements.csv` `diagnostic_class` field:

| Label | Class | Criteria |
|---|---|---|
| 0 | Normal | `NORM` in SCP codes, no MI codes, no other codes |
| 1 | MI | MI-coded record with all stadium values in {Stadium I, Stadium I-II, Stadium II} |
| 2 | Abnormal | Not MI-coded and not Normal |
| 3 | History_MI | MI-coded record with all usable stadium values exactly Stadium III |

Records with absent, unknown, transitional, or conflicting stadium values are excluded. Each of the 600 selected records belongs to a unique patient.

---

## Patient-separated PTB-XL folds

PTB-XL stratified 10-fold split used as-is from the dataset metadata:

| Split | Folds | Images | Patients |
|---|---|---|---|
| Train | 1–8 | 502 | 502 |
| Validation | 9 | 53 | 53 |
| Test | 10 | 45 | 45 |

Train/val, val/test, and train/test patient overlaps are all zero (confirmed in `split_summary.json`).

---

## Limitations

- **No Mendeley patient identifiers**: the Mendeley dataset does not include patient IDs; image-level independence is assumed but cannot be verified. Repeated patients, if any, would inflate Mendeley accuracy estimates.
- **Small PTB-XL test fold**: fold 10 contains N=45 images; confidence intervals are wide and point estimates should be interpreted with caution.
- **Chance-level external performance**: all zero-shot PTB-XL results are near or at 25% (four-class chance level). The model trained on Mendeley does not generalise to PTB-XL-derived images.
- **Failed limited fine-tuning**: E7 fine-tuning of layer4 and fc for 20 epochs did not recover performance and produced accuracy below chance on the test fold.

---

## Reproducibility

Set `RETRAIN_MODEL = False` in Block 0 (default) to load the included `mendeley_model_weights.pth` and reproduce all evaluation results without retraining. All random operations use seed 42. PTB-XL record selection is fully deterministic given the same dataset version and seed.

Running the notebook produces:
- All figures in `figures/`
- All metrics in `results/`
- Figures 7 and 8 (domain shift analysis) require PTB-XL generated images

---

## No LICENSE file

This repository does not currently include a LICENSE file. No open-source license has been chosen.
