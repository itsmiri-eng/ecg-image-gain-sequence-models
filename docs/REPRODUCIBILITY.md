# Reproducing the PTB-XL image-model study (updated 10 November 2026)

**Repository contents.** The Git repository holds code, configs, protocols, run records, saved predictions and summaries.
It does **not** hold PTB-XL itself, the rendered pages and image caches, the external Challenge 2021 data, or model
checkpoints (`.pt`, `.npz`, `.joblib`). These are re-creatable with the commands below, or archived separately.
The external test's `infer` step must never be repeated for a run that already has
`results/external/<dataset>/<run_id>/inference_state.json`.

This file is the entry point for a reviewer or for the Zenodo/GitHub archive.

## 1. Environment

- **Machine.** Windows 11, Python 3.12.10, PyTorch 2.9.1+cu128 (CUDA 12.8, cuDNN 9.10.2), NVIDIA RTX 5070 Ti
  (driver 610.88).
- **Exact package versions:** `environment/requirements-lock-pc-2026-11-03.txt`. Top-level requirements are in
  `requirements.txt`.
- **Setup check:** `python -X utf8 -m src.check_pc_setup`. It must end with "READY to train".
- **Seed 42 of the original networks** (CNN-only, CNN-LSTM, temporal conv, hybrid ViT on adaptive pages) was
  trained on a MacBook Air CPU. Every other network run used the GPU. Each run records its device in `run.json`.
- **Windows.** Always run Python with `-X utf8`. The Windows-specific fixes are listed in
  `docs/windows_portability_2026-11-02.md`.

## 2. Data

- **PTB-XL v1.0.3** (PhysioNet, CC BY 4.0) goes in `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`.
  Its `SHA256SUMS.txt` is checked by the cache builders.
- **Derived data**, each bound by SHA-256 to its source and code:
  - Rendered pages: `data/derived/ptbxl_images_v1/`.
  - Learning caches:
    - `data/processed/ptbxl_lead_images_v1/` (adaptive gain);
    - `data/processed/ptbxl_lead_images_fixedgain2p5_v1/` (fixed gain);
    - `data/processed/ptbxl_signals100_v1/` (100 Hz signals).
- **Excluded from version control** (see `.gitignore`): large binaries and checkpoints. They can be rebuilt with
  the commands below, or archived separately on Zenodo.

## 3. Commands (run from the project folder)

```bash
# caches (only if missing; builders refuse to overwrite)
python -X utf8 -m src.ptbxl_fixed_gain build --workers 3
python -X utf8 -m src.ptbxl_fixed_gain verify
python -X utf8 -m src.ptbxl_signals

# classical baselines
python -X utf8 -m src.run_ptbxl_lda --seed 42
python -X utf8 -m src.run_ptbxl_svm --seed 42

# networks: one run = one config + one seed (models cnn_pool, cnn_lstm, temporal_conv, hybrid_vit; fg_* = fixed gain)
python -X utf8 -m src.run_queue --device cuda cnn_pool:43 cnn_lstm:43 temporal_conv:43 hybrid_vit:43
python -X utf8 -m src.run_queue --device cuda fg_cnn_pool:43 fg_cnn_lstm:43 fg_temporal_conv:43 fg_hybrid_vit:43
python -X utf8 -m src.run_ptbxl_wide_head --device cuda --seeds 42 43 44 45 46
python -X utf8 -m src.run_ptbxl_signal --seed 42 --device cuda

# seeds 45/46 and the capacity-matched head (docs/protocol_amendment_2026-11-03.md)
python -X utf8 -m src.run_queue --device cuda cnn_pool:45 cnn_lstm:45 temporal_conv:45 hybrid_vit:45
python -X utf8 -m src.run_queue --device cuda fg_cnn_pool:46 fg_cnn_lstm:46 fg_temporal_conv:46 fg_hybrid_vit:46

# external confirmatory test (docs/external_test_plan.md; frozen in experiments/external/plan_freeze.json)
python -X utf8 -m src.download_external --workers 4                        # Georgia, Chapman-Shaoxing, Ningbo (~6.6 GB, SHA-256 checked)
python -X utf8 -m src.download_external --sets ptbxl-headers --report download_report_ptbxl_headers.json
python -X utf8 -m src.external_label_mapping                               # SNOMED CT -> superclass mapping (rule fixed in the plan)
python -X utf8 -m src.external_test inventory --dataset georgia            # then: render, infer (ONCE per run and dataset), analyze
python -X utf8 -m src.external_test render --dataset georgia --workers 2
python -X utf8 -m src.external_test infer --dataset georgia --device cuda
python -X utf8 -m src.external_test analyze --dataset georgia             # repeat the four steps with --dataset china

# analyses from saved predictions (no retraining)
python -X utf8 -m src.compare_heads_hierarchical                           # five-seed head comparison, Holm-corrected
python -X utf8 -m src.compare_ptbxl_all
python -X utf8 -m src.compare_fixed_gain
python -X utf8 -m src.make_paper_figures                                   # manuscript figures + PTB-XL hierarchical fixed-vs-adaptive
python -X utf8 -m src.evaluate_fixed_gain_transfer_test --split validation
python -X utf8 -m src.analyze_calibration_thresholds
python -X utf8 -m src.analyze_saliency --device cuda
python -X utf8 -m src.analyze_pulse_swap --device cuda
python -X utf8 -m src.decompose_pulse_effect
python -X utf8 -m src.make_model_cards

# tests (synthetic data only)
python -X utf8 -m pytest -q tests
```

## 4. Protocols (all written before the runs they govern)

| Protocol | Covers |
|---|---|
| `experiments/configs/*.json` | Frozen hyperparameters and cache hashes for every run |
| `docs/ptbxl_fixed_gain_protocol.md` | Fixed-gain experiment |
| `docs/secondary_analyses_protocol.md` | Calibration, thresholds, saliency, pulse swap |
| `docs/protocol_amendment_2026-11-03.md` | Seeds 45/46, capacity-matched head, multiple-comparison correction |
| `docs/external_test_plan.md` | External confirmatory test (frozen 5 Nov 2026; amendment 6 Nov in `experiments/external/amendments.json`) |

Post-hoc analyses say so in their file headers, for example `src/decompose_pulse_effect.py`.

## 5. What reproduces exactly and what does not

- **Analyses from saved predictions** reproduce exactly.
- **CPU training** is deterministic on one machine.
- **GPU training** is not bit-reproducible: cuDNN and atomics introduce nondeterminism. A rerun of a GPU seed
  gives a close but not identical model. Seed spread (5 seeds) is the honest uncertainty measure.
- **Re-inference on GPU** of a saved checkpoint matches the saved probabilities within 4e-6 (TF32 off). One
  exception is the Mac-CPU hybrid ViT, at 4.3e-4.
- **Fold 10 (test)** was observed before the models were specified, so PTB-XL test results are exploratory.
  Confirmatory evidence will come from the external test.

## 6. Per-run records

Each run directory `results/<family>/<run_id>/` contains:
- the config, provenance (code and data hashes), and a source snapshot;
- train, validation and test predictions, with metrics;
- the patient bootstrap;
- `run.json`.

The matching `models/<family>/<run_id>/` holds the checkpoint and training curve. `experiments/experiment_log.csv`
is append-only. The model cards are in `docs/model_cards.md`, and the data statement is in `docs/data_statement.md`.
