# Protocol amendment, 11 November 2026: 40-epoch sensitivity and a pretrained ResNet-18

Written and frozen **before** any run listed here started (11 Nov 2026, about 01:00 local time, UTC+4 = 10 Nov 21:00 UTC; runs started 21:05 UTC).
Nothing below changes a completed run, or a config or hyperparameter used by an existing run. Both
analyses are **sensitivity analyses**. They do not replace the primary results reported so far.

**Motivation.** On 11 Nov, a reviewer-style read of the manuscript raised two points:

1. **Possible under-training.** 17 of the 25 adaptive-gain network runs (seeds 42–46) selected their checkpoint
   at epoch 18–20 of the 20-epoch budget. CNN-only and the capacity-matched head did so in every run. A reviewer
   could argue that the simpler heads were still improving, which would bias the head comparison. This was
   observed from the selected epochs (validation-based) of existing runs. No new test result has been seen.
2. **Very small models.** All networks have at most 24k parameters. It is open whether the adaptive-gain cue,
   and the benefit of fixed gain, also apply to a standard, ImageNet-pretrained image backbone.

## A. 40-epoch sensitivity (`ptbxl_v2_long40_*`)

- **Heads:**
  - CNN-only (`cnn_pool`);
  - capacity-matched CNN-only (`cnn_pool_wide`);
  - CNN-LSTM (`cnn_lstm`);
  - CNN + temporal convolution (`temporal_conv`);
  - CNN + small ViT (`hybrid_vit`).
- **Pages and seeds:** adaptive-gain pages only (v1 cache); seeds **42, 43, 44**; all on the PC GPU.
- **Configs:** `experiments/configs/ptbxl_v2_long40_<head>.json`. Each is a copy of the head's existing v2 config
  with only these changes:
  - `max_epochs` 40 instead of 20;
  - the cosine schedule therefore spans 40 epochs (`T_max = max_epochs`, as in every run);
  - new protocol, result family (`ptbxl_v2_long40`) and run IDs.
- **Unchanged:** everything else, including patience 4, lr 1e-3, batch 16, the selection rule, threshold 0.5,
  one test inference per run, and the shared trainer and runner.
- **Analysis (fixed now):**
  1. Selected epoch and epochs run per run; how many runs stop early.
  2. For each head, test and validation macro AUROC (mean ± SD over seeds 42–44) at 40 epochs, next to the
     20-epoch runs of the same seeds. Seed 42 of the four original 20-epoch heads was CPU-trained; a GPU-only
     view (seeds 43–44) is also shown.
  3. The 10 pairwise head comparisons repeated on the 40-epoch runs. Method: the same hierarchical bootstrap as
     the main analysis (2,000 resamples, seed 2024, seeds then patients, Holm over 10 pairs), with seeds 42–44.
  4. **Interpretation rule.**
     - If every sequence head still exceeds the capacity-matched head (Holm-adjusted p < 0.05), the claim
       “relating columns helps beyond capacity” is robust to the training budget.
     - If a gap shrinks below significance or reverses, the manuscript reports this and weakens the claim
       accordingly.
     - Either outcome is reported.

## B. Pretrained ResNet-18 on adaptive and fixed gain (`ptbxl_v2_resnet18`, `ptbxl_v2_fg_resnet18`)

- **Weights:** torchvision's official ImageNet ResNet-18, `resnet18-f37072fd.pth`, downloaded on 11 Nov 2026 from
  `https://download.pytorch.org/models/` with the user's permission.
  - SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, stored in `models/pretrained/`
    and not versioned.
  - The network is re-implemented in `src/models/ptbxl_resnet18.py`, with the same layer names as torchvision
    (torchvision is not installed), and the weights are loaded with strict key matching.
- **Input:**
  - The same 12 × 120 × 1120 standardized ink tensor as every other network (same caches, same training-only
    normalization).
  - The 12 lead rows are stacked vertically into one 1440 × 1120 single-channel page image. No resizing.
  - The first convolution's ImageNet RGB filters are summed into one input channel, which is equivalent to giving
    the same image to all three channels.
- **Head:** global average pooling, dropout 0.2, linear 512 → 5. **11.2 million parameters.**
- **Training:**
  - Identical to every other network: AdamW lr 1e-3, weight decay 1e-4, batch 16, max 20 epochs, patience 4,
    cosine to 1e-5, gradient clipping 1.0, unweighted BCE, no augmentation, TF32 off, deterministic algorithms
    requested.
  - **One deviation: bfloat16 mixed precision.** It applies to the forward pass on the GPU (autocast inside the
    model) and uses a channels-last memory layout.
    - Weights, optimizer state and loss stay float32, and logits are returned as float32.
    - Reason: in full float32, one epoch on 1440 × 1120 images took about 19 minutes (measured before any
      training: 1.05 s per batch). With bf16 it takes about 3 minutes (0.17 s per batch). Only with bf16 do the
      six runs fit in the time the shared PC is available.
    - This was decided before any ResNet run. The small networks keep full float32.
  - All weights are fine-tuned, and BatchNorm statistics update in training mode.
  - No hyperparameter is tuned.
- **Seeds:** 42, 43, 44, on adaptive-gain and fixed-gain (2.5 mm/mV) pages, all on the PC GPU.
- **Analysis (fixed now):**
  1. PTB-XL test macro and per-label AUROC (mean ± SD over seeds), next to the small networks.
  2. **Fixed − adaptive** macro AUROC and HYP AUROC on the PTB-XL test fold. Method: the same hierarchical seed ×
     patient bootstrap as the main fixed-gain analysis (seeds 42–44), Holm over these 2 comparisons.
  3. **External cohorts (descriptive extension of the frozen external plan).**
     - Each ResNet run is reloaded and must reproduce its saved validation probabilities.
     - It then performs exactly one inference per cohort (Georgia; Chapman-Shaoxing + Ningbo), with the frozen
       inclusion rules, label mapping and eligibility.
     - Fixed − adaptive macro and HYP AUROC: hierarchical seed × record bootstrap, Holm over the 4 comparisons
       (2 cohorts × 2 metrics).
     - These are added after the plan was frozen, so they are reported as a prespecified-today sensitivity
       analysis, not as part of the confirmatory H1/H2.
  4. **Interpretation.**
     - If fixed gain also improves HYP AUROC for ResNet-18, the gain cue is not specific to tiny models.
     - If not, the manuscript says so and restricts the claim.

## C. Execution and limits

- **Order:** ResNet-18 runs first. They use the GPU heavily, and the PC is shared: the GPU must be free by
  11 Nov 15:00 AST. The 40-epoch runs are light and may continue at low CPU afterwards.
- **Interruptions:**
  - Any run interrupted by the time limit, a sign-out or a restart is moved to `experiments/abandoned_runs/`
    with a README and reported. It is never resumed.
  - A run interrupted after its test inference started is never repeated.
  - **Clarification, 11 Nov 08:10 UTC, after the PC restart.** A run interrupted before checkpoint selection and
    test inference is restarted from scratch under the same ID, as on 1–6 Nov.
    - This applied to three 40-epoch seed-42 runs stopped in epoch 1:
      `experiments/abandoned_runs/2026-11-11_pc_restart_long40/`.
    - It is not a change of analysis.
- **Hardware:** processes run hidden at Idle priority, limited to 8 logical CPUs overnight and 2 from 15:00.
  Runtimes are therefore not speed measurements.
  - **What happened (recorded 11 Nov, evening).** The six ResNet-18 runs finished overnight on 8 logical CPUs, by
    11 Nov 03:00 UTC. The PC was then free until 22:00 local time, so the 40-epoch runs used 10 logical CPUs from
    08:05 UTC and 11 from 08:50 UTC (about 70% of the processor, as the user allowed). Hardware limits do not change
    any result.
- **Known limitations:**
  - three seeds only;
  - the external-test extension is outside the frozen plan;
  - ResNet-18 sees the 2× max-pooled page (4 px/mm), not the full 8 px/mm render;
  - GPU training is not bit-reproducible, and some cuDNN backward kernels (pooling) are non-deterministic.
