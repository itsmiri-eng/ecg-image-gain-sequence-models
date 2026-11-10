# Protocol amendment, 3 November 2026: seeds 45/46 and a capacity-matched CNN-only head

Written and frozen **before** any run listed here started. Nothing below changes a completed run,
a hyperparameter or a config used by an existing run. Motivation: `docs/ROADMAP_Q1.md` weaknesses #3
(three seeds give a rough SD; many comparisons without correction) and #4 (CNN-only has 1,705 parameters,
the sequence heads 17,161–23,881).

## A. Additional seeds

- Seeds **45 and 46** for every network, under the unchanged configs:
  - adaptive gain: `ptbxl_v2_{cnn_pool, cnn_lstm, temporal_conv, hybrid_vit}.json`;
  - fixed gain: `ptbxl_v2_fg_{cnn_pool, cnn_lstm, temporal_conv, hybrid_vit}.json`.
- All runs use the PC GPU (`--device cuda`), the unchanged shared trainer (`src/training/ptbxl_trainer.py`)
  and `src/run_ptbxl_experiment.py`. These are the same code and settings as seeds 43/44.
- Seeds 42–46 are all reported. No seed is dropped or selected. A failed run is reported as failed and gets
  no rerun under the same ID.
- Runs execute in up to five parallel lanes. Runtimes are therefore not speed measurements.

## B. Capacity-matched CNN-only head (`cnn_pool_wide`)

- Model: `src/models/ptbxl_cnn_pool_wide.py`, config `experiments/configs/ptbxl_v2_cnn_pool_wide.json`,
  runner `src/run_ptbxl_wide_head.py`.
- Architecture: the shared 1,220-parameter CNN, then one MLP (96 → 96 → 64, ReLU, dropout 0.2) applied to
  each of the 70 feature columns separately, then the column mean and a linear 64 → 5 readout.
  **17,065 parameters** in total, against 17,161 for temporal conv and 18,025 for CNN-LSTM.
- No operation mixes columns before the mean. The head therefore has the capacity of the sequence heads
  without their ability to relate one part of a row to another.
- Hyperparameters are identical to every other network: AdamW lr 1e-3, batch 16, 20 epochs, patience 4,
  cosine schedule, dropout 0.2. There is no tuning.
- Seeds 42, 43, 44, 45, 46, all on the PC GPU, with adaptive-gain (v1) pages only.

## C. Analysis (fixed now)

1. **Main table.** For each network, report the mean ± SD across seeds of test and validation macro AUROC.
   - Primary: seeds 42–46, with Mac-CPU seed 42 pooled, as in `seed_summary_all_seeds.csv`.
   - Sensitivity: GPU-only seeds 43–46. The wide head has GPU seeds 42–46 in both.
2. **Pairwise head comparisons.** Pairs are taken among CNN-only, wide CNN-only, CNN-LSTM, temporal conv and
   hybrid ViT (10 pairs).
   - **Method.** For each pair, run a hierarchical bootstrap (2,000 resamples, seed 2024): resample seeds with
     replacement, then patients with replacement, both shared across the two models. Report the mean
     difference in test macro AUROC with a percentile 95% CI.
   - **p-values.** Two-sided bootstrap p = 2·min(P(Δ ≤ 0), P(Δ ≥ 0)).
   - **Correction.** Holm correction across the 10 pairs at α = 0.05.
   - **Status.** Fold 10 was observed before, so these are labelled exploratory. Validation results are
     reported alongside.
3. **Interpretation rule for the wide head.**
   - Wide head > CNN-only: extra head capacity helps.
   - Temporal conv or CNN-LSTM > wide head: relating columns helps beyond capacity.
   - The ViT is compared too, but it has about 6,800 more parameters than the wide head.
4. **Fixed gain.** `src.compare_fixed_gain` is extended to seeds 45/46. The matched-hardware summary uses GPU
   seeds 43–46, with the same metrics and strata as before.
5. **Limitations stated up front.**
   - GPU training is not bit-reproducible across runs.
   - Seed 42 of the four original networks was trained on a Mac CPU.
   - The wide head also differs in having one extra dropout layer.
