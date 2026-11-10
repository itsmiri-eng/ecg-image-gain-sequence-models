# Final experiment evidence verification

Verified at 2026-09-23T09:33:19.957357+00:00. The completed `lda_v1_seed42` and `cnn_v1_seed42` artifacts passed the checks below. No model was retrained, and no new test-set model inference was performed.

- Both runs use dataset `image_dedup_8e04bd56af2f` and split `grouped_cf5a867a99c3`: **295 training / 98 validation / 98 test images**. Their saved split files are byte-identical to the frozen split; the input-cache hashes agree.
- The source/cache/audit/config integrity verifier passed. Printed report IDs have **0 overlap** across every partition pair; group counts are **187 / 63 / 61**. These are report-ID groups, not verified distinct patients.
- Saved prediction rows match split sample order and class order for all three partitions in both experiments. Probabilities are finite, nonnegative, and sum to one within floating-point tolerance. Recalculated metrics from the existing prediction CSVs match saved metrics to **1e-12**.
- CNN checkpoint **epoch 25** is the minimum of the complete **30-epoch weighted validation-loss history** (loss **1.039684346476449**). Thirty epochs completed; the patience rule did not end training early. The restored checkpoint, rather than epoch 30, supplied final predictions.
- CNN mean **0.06047950372331967** and standard deviation **0.08952904696916278** exactly equal recomputation from training pixels only. Class weights exactly match `295 / (4 × count)` using training counts **[85, 18, 140, 52]**.
- LDA PCA stores **295 fitting samples**, **32 components**, and the training-only pixel mean. Uniform class priors are **[0.25, 0.25, 0.25, 0.25]**.
- Both checkpoints were loaded and evaluated on the **training partition only**. Maximum absolute differences from saved training probabilities: LDA **6.17e-07**, CNN **0**. Both have **0 predicted-label differences across 295 images** (probability tolerance 1e-5). Checkpoint hashes were unchanged.
- Both saved full configurations equal the frozen protocol, which predates both runs. Every source hash recorded at training still matches. Code inspection confirms LDA receives training samples only; CNN receives training and validation samples, selects by validation loss, and receives no test samples in its fitting function.

The scikit-learn probability-sum warning is a precision issue: CNN row sums differ from 1 by at most **1.341104507446289e-7** (validation maximum **1.1362135410308838e-7**). All saved values round-trip through float32 exactly. The evaluator casts these softmax outputs to float64, whose log-loss check uses a much tighter relative tolerance (approximately **1.49e-8**). This is consistent with ordinary float32 rounding. **No scores or probabilities were altered.**

## Limits of this verification

The loader reads and integrity-checks the full frozen cache before fitting; no cohort-wide learned normalization or feature fitting occurs there. The test set is passed to prediction only after model fitting in the recorded runner. Source review and saved artifacts support that ordering; they are not a separately instrumented execution trace.

Transcribed report IDs do not establish patient identity, and some IDs span different folder labels. The small test cohort includes only **6 MI images**. Header/footer cropping and exact-duplicate controls cannot guarantee that every acquisition artifact or near duplicate is absent. These results remain exploratory image baselines and do not establish clinical validity or superiority from accuracy alone.

Detailed observed values and checkpoint hashes: `experiments/logs/checkpoint_verification.json`.
