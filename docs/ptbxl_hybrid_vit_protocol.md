# Prespecified compact hybrid image transformer — 1 November 2026

This is the next ViT-family experiment after completed CNN–LSTM and spatiotemporal CNN+TCN runs. The unresolved “LSM” and remembered convolution name remain unconfirmed; this experiment does not identify or implement them. No prior formal run needs invalidation based on the fresh data/results audit.

## Question and hypothesis

Does global self-attention across the same CNN image-feature sequence improve five-label discrimination relative to a recurrent or temporal-convolution head under the same training budget? Long-range feature interaction could help, but transformers are not assumed to perform better. Architecture, head capacity and dropout topology differ, so a result cannot isolate attention as the sole cause. This is a compact hybrid CNN-feature transformer, following the hybrid option in [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929), not a large pretrained raw-patch ViT.

## Frozen design

- Real PTB-XL v1.0.3 recordings rendered as clean 12-lead images. No AI-generated ECG examples or native waveform values enter the classifier.
- Five overlapping targets: NORM, MI, STTC, CD, HYP. All mapped diagnostic SCP keys are retained, including recorded likelihood zero, under the existing audited target protocol.
- Official patient folds: 17,084 training records (folds 1–8), 2,146 validation (fold 9), 2,158 test (fold 10). No known patient/ECG or exact-pixel overlap.
- Input: 12 × 120 × 1120 full-lead ink images. Same crop, 2×2 maximum-ink pooling and train-only global mean/std as both preceding PTB-XL runs; no augmentation or class weighting.
- Same three depthwise/pointwise CNN stages, independently initialized in the same seeded construction order. Pool to 24 × 4 × 70, reshape to 70 spatial feature positions × 96 features.
- Linear projection 96→32; learned positional embedding (normal initialization, standard deviation 0.02); dropout 0.2.
- Two independently initialized pre-norm Transformer encoder layers: four heads, feedforward width 64, GELU, dropout 0.2. Global attention, no causal mask, no CLS token. Final LayerNorm, mean across positions, dropout 0.2 and five output logits.
- **23,881 trainable parameters**. The 70 positions are image features including calibration/margins, not recovered waveform samples.
- Seed 42, batch 16, four CPU threads, zero loader workers, AdamW learning rate 0.001 and weight decay 0.0001, gradient clipping 1.0, cosine schedule to 0.00001, maximum 20 epochs, patience four. Auto device selection must pass a synthetic deterministic forward/backward probe; this environment resolves to CPU.
- Unweighted BCEWithLogitsLoss. Select strict lexicographic maximum of validation macro AUROC, macro average precision, then negative probability BCE; earliest exact tie. The fixed five independent sigmoid thresholds are 0.5.

## Evaluation and preservation

The runner requires the prespecified source/configuration/evidence hashes before fitting. It keeps all earlier runs and the legacy log's byte prefix intact, refuses existing experiment IDs, saves source snapshots, and records failures without silently restarting. A new unique ID can deliberately reproduce the frozen protocol after later log rows have been appended.

Selected-checkpoint train/validation predictions are saved. Validation metrics must equal the selected epoch, and a reconstructed checkpoint must reproduce 32 training examples exactly on the same device. Only then is one test prediction pass permitted. Saved truth/probabilities supply all subsequent analyses; no repeated checkpoint inference is needed.

Report exact-set accuracy, individual-label accuracy, macro and weighted F1, macro AUROC/AP, per-label confusion counts/precision/recall/specificity/F1/AUC/AP, parameter count, selected epoch and runtimes. Publication summaries additionally report mean binary balanced accuracy across five labels, explicitly distinct from multiclass balanced accuracy. Patient-cluster percentile bootstrap uses 500 draws, seed 2024, 95% intervals. Primary paired comparison is hybrid minus TCN macro AUROC; comparison with CNN–LSTM is secondary. Use identical patient draws for both members of each comparison.

The test cohort was previously observed for two architectures. Comparisons are exploratory and conditional on one fitted seed/configuration; intervals do not establish training-seed robustness, external validity or superiority on real scanned ECGs. HYP/MI failures and any deterioration must be reported alongside aggregate gains. No threshold or architecture change is permitted based on this test outcome.

## Execution

From the project root, the original run command is:

```bash
/opt/anaconda3/bin/python -B -m src.run_ptbxl_hybrid_vit --run-id ptbxl_hybrid_vit_image_v1_seed42
```

A deliberate unchanged reproduction must use a fresh ID, for example `ptbxl_hybrid_vit_image_v1_seed42_reproduction_01`; existing IDs cannot be reused. Source/config changes require a separately prespecified protocol, not edits to the frozen experiment. The current run is a fresh fit, with no checkpoint-resume claim.
