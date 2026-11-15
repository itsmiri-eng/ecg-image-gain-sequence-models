# Protocol amendment, 15 November 2026: capacity-matched head on fixed-gain pages (Additional analysis)

Written and frozen **before** any run listed here started (15 Nov 2026, evening, local time; the commit that adds
this file and the config precedes the first run). Nothing below changes a completed run, or a config or
hyperparameter used by an existing run. Everything produced under this amendment is labelled **Additional
analysis** in the manuscript and the Supplementary Information and is kept apart from the earlier analyses.

**Motivation.** The supervisor's review (15 Nov 2026) noted that the Supplementary Information says "the
capacity-matched head was never trained on fixed gain" and asked for it to be run "using the same five seeds and
the same evaluation settings". The request reached us as "the same five C's"; we read this as the five training
seeds 42–46, which is the only five-fold repetition in the protocol. The other possible reading, the five
diagnostic superclasses (NORM, MI, STTC, CD, HYP), is satisfied by the same runs, because they use the same five
labels. Both readings therefore lead to the runs below. The fixed-gain results of this head have not been seen by
anyone.

## Runs

- **Head:** capacity-matched CNN-only (`cnn_pool_wide`, 17,065 parameters), unchanged model file
  `src/models/ptbxl_cnn_pool_wide.py`.
- **Pages and seeds:** fixed-gain (2.5 mm/mV) pages, dataset `ptbxl_images_fixedgain2p5_v1` (same cache and
  SHA-256 binding as every other fixed-gain run); seeds **42, 43, 44, 45, 46**; all on the PC GPU. 5 runs.
- **Config:** `experiments/configs/ptbxl_v2_fg_cnn_pool_wide.json`. It is the head's adaptive-gain config
  `ptbxl_v2_cnn_pool_wide.json` with only the dataset fields (dataset id, cache root, cache hashes, loader,
  `render_gain` note) taken from the existing fixed-gain configs, a new protocol name, result family
  `ptbxl_v2_fixedgain` and run IDs `ptbxl_v2_fg_cnn_pool_wide_seed<s>`. The `training` block is identical
  (checked by script): AdamW, lr 1e-3, weight decay 1e-4, batch 16, at most 20 epochs, cosine to 1e-5, patience 4,
  dropout 0.2, clip 1.0.
- **Unchanged:** selection rule (validation macro AUROC, then macro AP, then −BCE), threshold 0.5, one test
  inference per run after the checkpoint is frozen and reloaded, the shared trainer, and the runner
  `src/run_ptbxl_extra.py` (the same runner as the 11 Nov amendments).
- **Hardware note.** All five adaptive-gain runs of this head were GPU-trained, so this head's comparison has no
  CPU/GPU mix.

## Analysis (fixed now, before any result exists)

Bootstrap: hierarchical seed × patient bootstrap on the PTB-XL test fold (seeds resampled, then patients, both
shared by the two models), seed 2024, percentile 95% CI; seed × record bootstrap on the external cohorts. The
number of resamples and the p-value formula follow the statistical revision
`docs/protocol_amendment_2026-11-15_bootstrap_inference.md`, which is frozen before any of these results is
analysed; the legacy formula (2,000 resamples) is also computed so that the new rows can be read next to the old
tables.

1. **Per run:** selected epoch, validation and test metrics, per-label AUROC (same table format as every run).
2. **Primary: fixed − adaptive for the capacity-matched head** on the PTB-XL test fold, macro AUROC and HYP AUROC,
   seeds 42–46. Holm over its own family of 2. As a sensitivity analysis it is also placed in a family of 10 with
   the eight existing fixed-versus-adaptive comparisons (five heads × two metrics), and both adjusted p-values
   are reported.
3. **Secondary: head pairs on fixed-gain pages** (PTB-XL test, macro AUROC, seeds 42–46): all 10 pairs among the
   five heads, Holm over 10. This asks whether the sequence heads still exceed the capacity-matched head when
   the gain cue is removed.
4. **External cohorts.** One inference per run and cohort (Georgia, China) through
   `src.external_test infer-added`, after the usual reproduction check of the saved validation probabilities.
   Reported: external macro AUROC per run, fixed − adaptive for this head (macro and HYP; Holm over 2 within each
   cohort), and the 10 fixed-gain head pairs (Holm over 10). These are **outside** the frozen confirmatory
   families H1 and H2 of `docs/external_test_plan.md`, which are not changed.
5. **Interpretation rule.**
   - The direction and size of fixed − adaptive are reported whatever they are.
   - A difference is called an improvement only if its 95% CI excludes zero and its Holm-adjusted p < 0.05;
     otherwise it is described as a point estimate that is not statistically supported.
   - If the capacity-matched head is worse with fixed gain on either metric, the manuscript sentences that
     describe fixed gain as raising the point estimates of every architecture are corrected.
   - No non-inferiority claim is made: no margin was prespecified.

## Execution

- Hidden lanes at low priority on at most 8 of 16 logical CPUs (shared PC), started 15 Nov evening.
- Runs interrupted before checkpoint selection and test inference are moved to `experiments/abandoned_runs/` with
  a README and restarted from scratch under the same ID. A run interrupted after its test inference started is
  never repeated.
- **Known limitations:** five seeds; GPU training is not bit-reproducible; 20-epoch budget (this head selected
  epoch 18–20 in every adaptive-gain run, so it is probably budget-limited here as well); the request for this
  analysis came after all earlier results were known, so it is not part of the prospectively frozen plan.
