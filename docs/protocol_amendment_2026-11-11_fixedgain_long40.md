# Protocol amendment, 11 November 2026 (evening): fixed-gain training with a 40-epoch budget

Written and frozen **before** any run listed here started (11 Nov 2026, committed 23:10 local time = 19:10 UTC, commit 652b435).
Nothing below changes a completed run, or a config or hyperparameter used by an existing run. This is a
**sensitivity analysis**. It does not replace the primary results.

**Motivation.** The 40-epoch sensitivity of the morning amendment (`docs/protocol_amendment_2026-11-11.md`, part A)
showed that the 20-epoch budget limited every adaptive-gain head except the ViT, by 0.012–0.019 test macro AUROC. The
paper's main finding, that fixed-gain training improves on adaptive-gain training, was measured with 20-epoch models.
- If fixed gain mainly speeds up convergence, the advantage could shrink when both are trained longer.
- This amendment tests that.
- The fixed-gain 40-epoch results have not been seen by anyone. The adaptive-gain 40-epoch results are known
  (Supplementary Tables S16–S17).

## Runs

- **Heads:**
  - CNN-only (`cnn_pool`);
  - CNN-LSTM (`cnn_lstm`);
  - CNN + temporal convolution (`temporal_conv`);
  - CNN + small ViT (`hybrid_vit`).

  These are the four architectures with fixed-gain runs; the capacity-matched head was never trained on fixed gain.
- **Pages and seeds:** fixed-gain (2.5 mm/mV) pages; seeds **42, 43, 44**; all on the PC GPU. 12 runs in total.
- **Configs:** `experiments/configs/ptbxl_v2_long40_fg_<head>.json`. Each is a copy of the head's existing
  fixed-gain config `ptbxl_v2_fg_<head>.json`, with only these changes:
  - `max_epochs` 40 instead of 20, so the cosine schedule spans 40 epochs;
  - a new protocol name, result family `ptbxl_v2_long40` and run IDs `ptbxl_v2_long40_fg_<head>_seed<s>`.
- **Unchanged:** everything else, including patience 4, the selection rule, threshold 0.5, one test inference per
  run, the shared trainer and `src/run_ptbxl_extra.py`. These are the same settings as the adaptive-gain 40-epoch
  runs.

## Analysis (fixed now)

1. **Selected epoch and epochs run** per run.
2. **Primary: fixed − adaptive at 40 epochs.**
   - What is compared: macro AUROC and HYP AUROC on the PTB-XL test fold, with the 40-epoch adaptive-gain runs of
     the same head and seeds.
   - Method: hierarchical seed × patient bootstrap (2,000 resamples, seed 2024), the same as the main fixed-gain
     analysis.
   - Correction: Holm over the 8 comparisons (4 heads × 2 metrics).
3. **Secondary: fixed-gain 40 − 20 epochs** per head (macro AUROC, seeds 42–44, Holm over 4). The 20-epoch
   fixed-gain runs are all GPU-trained.
4. **Descriptive:** the 20-epoch fixed − adaptive differences for the same seeds 42–44, next to the 40-epoch ones.
5. **Interpretation rule.**
   - Macro and HYP differences stay positive for every head, with Holm-adjusted p < 0.05 where they were
     significant at 20 epochs: the gain finding is robust to the training budget.
   - A difference shrinks below significance or turns negative: the manuscript reports this and qualifies the
     gain finding for that head.
   - Either outcome is reported.
6. **External cohorts.** No external inference for these runs. The external tests stay with the 20-epoch models, as
   stated in the paper.

## Execution

- Four hidden lanes at Idle priority, limited to 11 logical CPUs (about 70% of the processor), started 11 Nov
  evening. The PC is free overnight.
- Runs interrupted before checkpoint selection and test inference are moved to `experiments/abandoned_runs/` with a
  README and restarted from scratch under the same ID, as on 1–11 Nov. A run interrupted after its test inference
  started is never repeated.
- **Known limitations:**
  - three seeds;
  - GPU training is not bit-reproducible;
  - seed 42 of the 20-epoch adaptive-gain runs of four heads was CPU-trained. This does not affect the primary
    comparison, because all 40-epoch runs are on the GPU.
